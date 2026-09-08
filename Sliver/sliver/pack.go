package sliver

import (
	"sync/atomic"
	"unsafe"
)

// PackRecord appends one record to the arena and returns its record ID.
//
// The call performs no allocation whatsoever. It reserves a record slot with a
// single fetch-and-add, reserves payload space in each column with one
// fetch-and-add per column, copies the caller's bytes into the mapping with
// explicit word stores, and publishes each descriptor with a release store. No
// intermediate buffer is built, no interface is boxed, no closure escapes, and
// nothing the garbage collector can see is written.
//
// fields[i] supplies column i. A nil entry, or a fields slice shorter than the
// schema width, marks the remaining columns stateNull rather than failing: that
// is what makes the fabric schema-free at the record level. A non-nil zero
// length slice is a committed empty value and is distinct from null.
//
// PackRecord is safe for concurrent use by any number of goroutines. Two
// packers never touch the same descriptor cell, because the record ID that
// selects the cell is handed out by an atomic increment. Two packers appending
// to different columns never touch the same cache line, because the column
// state blocks are two cache lines apart.
//
// If a column's payload vector is exhausted, that field is marked stateOverflow
// and ErrColumnFull is returned, but the record ID is still valid and every
// field that did fit is readable. A partially packed record is honest about
// which fields are missing rather than silently truncating.
func (a *Arena) PackRecord(fields [][]byte) (uint64, error) {
	if a.isClosed() {
		return 0, ErrClosed
	}
	if uint64(len(fields)) > a.fieldCount {
		atomic.AddUint64(&a.hdr.packFailures.value, 1)
		return 0, ErrFieldCount
	}

	record, err := a.ReserveRecord()
	if err != nil {
		return 0, err
	}

	supplied := uint64(len(fields))
	var firstErr error

	for f := uint64(0); f < a.fieldCount; f++ {
		// The descriptor cell address, computed once and reused for the whole
		// publication sequence:
		//
		//   cell = f*maxRecords + record
		//   d    = descBase + cell*16
		d := (*fieldDescriptor)(unsafe.Add(
			a.descBase,
			uintptr(f*a.maxRecords+record)*descriptorSize,
		))

		if f >= supplied || fields[f] == nil {
			// Explicit absence. Offset and length are zeroed before the state
			// word is released so a reader that observes stateNull can never
			// observe stale geometry from a previous Reset generation.
			atomic.StoreUint64(&d.payloadOffset, 0)
			atomic.StoreUint32(&d.payloadLength, 0)
			atomic.StoreUint32(&d.payloadState, stateNull)
			continue
		}

		if e := a.packInto(d, uint32(f), fields[f]); e != nil && firstErr == nil {
			firstErr = e
		}
	}

	if firstErr != nil {
		atomic.AddUint64(&a.hdr.packFailures.value, 1)
	}
	atomic.AddUint64(&a.hdr.commitCounter.value, 1)
	return record, firstErr
}

// ReserveRecord claims the next record ID without writing any field. The
// returned slot's descriptors are all stateEmpty until PackField fills them, and
// GetField reports ok=false for every one of them in the meantime.
//
// The reservation is a single atomic fetch-and-add on a counter that owns two
// full cache lines, so it is the only point of contention in the pack path and
// it contends on nothing else in the arena.
func (a *Arena) ReserveRecord() (uint64, error) {
	if a.isClosed() {
		return 0, ErrClosed
	}
	next := atomic.AddUint64(&a.hdr.recordCursor.value, 1)
	record := next - 1
	if record >= a.maxRecords {
		// Pull the cursor back to the ceiling so a long run of refused packs
		// cannot drive it toward wraparound. A racing reservation may have
		// already advanced it further; the CAS simply fails in that case and
		// the next refusal retries the clamp.
		atomic.CompareAndSwapUint64(&a.hdr.recordCursor.value, next, a.maxRecords)
		atomic.AddUint64(&a.hdr.packFailures.value, 1)
		return 0, ErrRecordLimit
	}
	return record, nil
}

// PackField writes one field of a previously reserved record.
func (a *Arena) PackField(record uint64, field uint32, payload []byte) error {
	if a.isClosed() {
		return ErrClosed
	}
	if uint64(field) >= a.fieldCount || record >= a.maxRecords {
		return ErrFieldCount
	}
	d := a.descriptor(field, record)
	if payload == nil {
		atomic.StoreUint64(&d.payloadOffset, 0)
		atomic.StoreUint32(&d.payloadLength, 0)
		atomic.StoreUint32(&d.payloadState, stateNull)
		return nil
	}
	return a.packInto(d, field, payload)
}

// packInto is the write half of the fabric: reserve a payload extent in a
// column, copy the bytes in, publish the descriptor.
//
// The address arithmetic, in full:
//
//	need  = (len(payload) + 7) &^ 7          // 8-byte aligned stride
//	end   = atomic.Add(&cs.cursor, need)     // reserve [end-need, end)
//	start = end - need                       // column-relative extent base
//	off   = cs.baseOff + start               // arena-relative extent base
//	dst   = arenaBase + off                  // absolute destination address
//
// cs.baseOff is cache-line aligned by construction and every reservation is a
// multiple of 8, so dst is always 8-byte aligned no matter what lengths the
// caller has packed before. That is the invariant that keeps a field read from
// ever splitting a cache line: an aligned 8-byte load inside an aligned extent
// cannot straddle a 64-byte boundary.
func (a *Arena) packInto(d *fieldDescriptor, field uint32, payload []byte) error {
	n := uint64(len(payload))
	if n > MaxFieldBytes {
		atomic.StoreUint32(&d.payloadState, stateOverflow)
		return ErrFieldTooLarge
	}

	cs := a.column(field)

	// Short-circuit on an already-full column so a sustained overflow cannot
	// drive the cursor toward uint64 wraparound. Drift past the limit is bounded
	// by (concurrent packers x max field size).
	if atomic.LoadUint64(&cs.cursor) > cs.limit {
		atomic.AddUint64(&cs.rejected, 1)
		atomic.StoreUint64(&d.payloadOffset, 0)
		atomic.StoreUint32(&d.payloadLength, 0)
		atomic.StoreUint32(&d.payloadState, stateOverflow)
		return ErrColumnFull
	}

	need := alignUp(n, WordAlign)
	end := atomic.AddUint64(&cs.cursor, need)
	if end > cs.limit || end < need {
		atomic.AddUint64(&cs.rejected, 1)
		atomic.StoreUint64(&d.payloadOffset, 0)
		atomic.StoreUint32(&d.payloadLength, 0)
		atomic.StoreUint32(&d.payloadState, stateOverflow)
		return ErrColumnFull
	}
	start := end - need
	off := cs.baseOff + start

	// The extent belongs exclusively to this call from here on: no other packer
	// can have been handed an overlapping range, because the range came out of
	// an atomic fetch-and-add. The copy therefore needs no further
	// synchronization, only the release store that follows it.
	if n != 0 {
		writeExtent(ptrAt(a.base, off), payload, n)
	}

	// Publish. Geometry first, state last. The state store is the release
	// barrier that makes every byte written above visible to any reader that
	// subsequently observes stateCommitted.
	atomic.StoreUint64(&d.payloadOffset, off)
	atomic.StoreUint32(&d.payloadLength, uint32(n))
	atomic.StoreUint32(&d.payloadState, stateCommitted)

	atomic.AddUint64(&cs.appended, 1)
	return nil
}

// GetField returns a zero-copy view of one field.
//
// The returned slice aliases the arena directly: no bytes are copied, nothing is
// allocated, and the slice header itself lives on the caller's stack. It remains
// valid until Reset or Close. Writing through it corrupts the arena, so treat it
// as read-only.
//
// The address arithmetic, in full:
//
//	cell = fieldIndex*maxRecords + recordID   // column-major descriptor index
//	d    = descBase + cell*16                 // descriptor address
//	ptr  = arenaBase + d.payloadOffset        // payload extent address
//	view = ptr[0 : d.payloadLength]           // the returned slice
//
// ok is false for an out-of-range coordinate, a record that has not been
// reserved, a field that was never written, one explicitly marked null, and one
// whose column overflowed. Those five cases are distinguished by FieldState.
func (a *Arena) GetField(recordID uint64, fieldIndex uint32) ([]byte, bool) {
	if a.isClosed() {
		return nil, false
	}
	if uint64(fieldIndex) >= a.fieldCount || recordID >= a.maxRecords {
		atomic.AddUint64(&a.hdr.readMisses.value, 1)
		return nil, false
	}
	// A record ID at or beyond the cursor was never handed out; its descriptor
	// cell is untouched memory and must not be interpreted.
	if recordID >= atomic.LoadUint64(&a.hdr.recordCursor.value) {
		atomic.AddUint64(&a.hdr.readMisses.value, 1)
		return nil, false
	}

	d := (*fieldDescriptor)(unsafe.Add(
		a.descBase,
		uintptr(uint64(fieldIndex)*a.maxRecords+recordID)*descriptorSize,
	))

	// Acquire. Everything the packer wrote before releasing this word is visible
	// to us once we have observed stateCommitted.
	if atomic.LoadUint32(&d.payloadState) != stateCommitted {
		atomic.AddUint64(&a.hdr.readMisses.value, 1)
		return nil, false
	}

	off := atomic.LoadUint64(&d.payloadOffset)
	length := atomic.LoadUint32(&d.payloadLength)

	// unsafe.Slice fabricates a slice header over the mapping. It allocates
	// nothing: the header is three words returned in registers or on the stack,
	// and the backing array is memory the collector does not own.
	return unsafe.Slice((*byte)(ptrAt(a.base, off)), int(length)), true
}

// FieldState reports the publication state of a descriptor cell without
// materializing the payload: one of StateEmpty, StateCommitted, StateNull, or
// StateOverflow. An out-of-range coordinate reports StateEmpty with ok=false.
func (a *Arena) FieldState(recordID uint64, fieldIndex uint32) (uint32, bool) {
	if a.isClosed() {
		return stateEmpty, false
	}
	if uint64(fieldIndex) >= a.fieldCount || recordID >= a.maxRecords {
		return stateEmpty, false
	}
	if recordID >= atomic.LoadUint64(&a.hdr.recordCursor.value) {
		return stateEmpty, false
	}
	d := a.descriptor(fieldIndex, recordID)
	return atomic.LoadUint32(&d.payloadState), true
}

// Exported descriptor states, for callers that switch on FieldState.
const (
	StateEmpty     = stateEmpty
	StateCommitted = stateCommitted
	StateNull      = stateNull
	StateOverflow  = stateOverflow
)

// FieldLen returns a field's length without touching its payload, so a caller
// sizing a buffer never pulls the payload's cache lines in. The descriptor's
// length and state words share one 8-byte line, so this costs a single load.
func (a *Arena) FieldLen(recordID uint64, fieldIndex uint32) (uint32, bool) {
	if a.isClosed() {
		return 0, false
	}
	if uint64(fieldIndex) >= a.fieldCount || recordID >= a.maxRecords {
		return 0, false
	}
	if recordID >= atomic.LoadUint64(&a.hdr.recordCursor.value) {
		return 0, false
	}
	d := a.descriptor(fieldIndex, recordID)
	if atomic.LoadUint32(&d.payloadState) != stateCommitted {
		return 0, false
	}
	return atomic.LoadUint32(&d.payloadLength), true
}

// CopyField copies a field into a caller-owned buffer instead of aliasing the
// arena, for callers that must hold the bytes across a Reset or Close. It
// returns the number of bytes written; if dst is too small, nothing is copied
// and ok is false.
func (a *Arena) CopyField(dst []byte, recordID uint64, fieldIndex uint32) (int, bool) {
	src, ok := a.GetField(recordID, fieldIndex)
	if !ok {
		return 0, false
	}
	if len(dst) < len(src) {
		return 0, false
	}
	n := uint64(len(src))
	if n == 0 {
		return 0, true
	}
	writeExtent(unsafe.Pointer(unsafe.SliceData(dst)), src, n)
	return int(n), true
}

// writeExtent copies n bytes from a Go slice into the arena using explicit
// pointer arithmetic.
//
// Short payloads take a hand-rolled 8-byte word loop. The destination is
// guaranteed 8-byte aligned by the reservation arithmetic in packInto, so every
// store is an aligned 64-bit store that cannot split a cache line; the source
// may be unaligned, which both amd64 and arm64 handle at full speed for loads.
// Avoiding the memmove call for small fields matters because a columnar record
// is overwhelmingly made of small fields, and the call overhead would otherwise
// dominate the copy.
//
// Long payloads hand off to the runtime's vectorized memmove via copy, which
// beats any word loop once the length is large enough to amortize the call.
// unsafe.Slice builds the destination header on the stack, so this allocates
// nothing either way.
//
// The crossover sits at one cache line: below it the word loop wins, at or above
// it the vectorized path does.
func writeExtent(dst unsafe.Pointer, src []byte, n uint64) {
	if n >= CacheLineSize {
		copy(unsafe.Slice((*byte)(dst), n), src)
		return
	}
	sp := unsafe.Pointer(unsafe.SliceData(src))
	i := uint64(0)
	for ; i+8 <= n; i += 8 {
		// dst[i:i+8] = src[i:i+8], as a single aligned 64-bit store.
		*(*uint64)(unsafe.Add(dst, uintptr(i))) = *(*uint64)(unsafe.Add(sp, uintptr(i)))
	}
	if i+4 <= n {
		*(*uint32)(unsafe.Add(dst, uintptr(i))) = *(*uint32)(unsafe.Add(sp, uintptr(i)))
		i += 4
	}
	if i+2 <= n {
		*(*uint16)(unsafe.Add(dst, uintptr(i))) = *(*uint16)(unsafe.Add(sp, uintptr(i)))
		i += 2
	}
	if i < n {
		*(*byte)(unsafe.Add(dst, uintptr(i))) = *(*byte)(unsafe.Add(sp, uintptr(i)))
	}
}
