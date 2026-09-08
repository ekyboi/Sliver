package sliver

import (
	"sync/atomic"
	"unsafe"
)

// This file holds the read-side tracking loops: the operations that sweep a
// whole column looking for something. They exist because the column-major
// descriptor index makes them cheap, and they are the reason that layout was
// chosen. Filtering column 3 of ten million records touches exactly the
// descriptor run of column 3 and the payload extents it points at; the other
// columns' descriptors and payloads are never brought into cache at all.
//
// Every loop here is allocation-free. Callers supply their own output buffers,
// callbacks receive slices that alias the arena, and no loop builds an
// intermediate collection.

// ScanColumn walks every committed value of one column in record order and
// hands each to visit. Returning false from visit stops the sweep early.
//
// The value slice aliases the arena and is valid only for the duration of the
// call; it must be copied if it is to outlive the sweep.
//
// The loop advances a raw descriptor pointer by a constant 16-byte stride rather
// than recomputing an index each iteration:
//
//	p = descBase + field*maxRecords*16      // the column's run base
//	then p += 16 per record
//
// so the address generation is a single add per record and the access pattern is
// a pure forward stream, which is exactly the shape hardware prefetchers detect.
func (a *Arena) ScanColumn(field uint32, visit func(recordID uint64, value []byte) bool) uint64 {
	if a.isClosed() || uint64(field) >= a.fieldCount {
		return 0
	}
	atomic.AddUint64(&a.hdr.scanCalls.value, 1)

	limit := a.RecordCount()
	p := a.columnRunBase(field)
	base := a.base

	var seen uint64
	for rec := uint64(0); rec < limit; rec++ {
		d := (*fieldDescriptor)(p)
		p = unsafe.Add(p, descriptorSize)

		if atomic.LoadUint32(&d.payloadState) != stateCommitted {
			continue
		}
		off := atomic.LoadUint64(&d.payloadOffset)
		length := atomic.LoadUint32(&d.payloadLength)
		seen++
		if !visit(rec, unsafe.Slice((*byte)(ptrAt(base, off)), int(length))) {
			return seen
		}
	}
	return seen
}

// FilterEqual scans one column for values byte-identical to needle and writes
// the matching record IDs into out. It returns the number written, stopping
// once out is full.
//
// Nothing is allocated: out belongs to the caller, and the comparison reads the
// arena in place. The length check in front of the byte comparison means a
// mismatched candidate costs one descriptor load and no payload access at all,
// so a selective filter never touches the payload region for the rows it
// rejects.
func (a *Arena) FilterEqual(field uint32, needle []byte, out []uint64) int {
	if a.isClosed() || uint64(field) >= a.fieldCount || len(out) == 0 {
		return 0
	}
	atomic.AddUint64(&a.hdr.scanCalls.value, 1)

	want := uint32(len(needle))
	var np unsafe.Pointer
	if want != 0 {
		np = unsafe.Pointer(unsafe.SliceData(needle))
	}

	limit := a.RecordCount()
	p := a.columnRunBase(field)
	base := a.base
	n := 0

	for rec := uint64(0); rec < limit; rec++ {
		d := (*fieldDescriptor)(p)
		p = unsafe.Add(p, descriptorSize)

		// One load covers both the state and the length: they share an 8-byte
		// word, so the length check below is already in cache.
		if atomic.LoadUint32(&d.payloadState) != stateCommitted {
			continue
		}
		if atomic.LoadUint32(&d.payloadLength) != want {
			continue
		}
		if want != 0 {
			off := atomic.LoadUint64(&d.payloadOffset)
			if !extentEqual(ptrAt(base, off), np, want) {
				continue
			}
		}
		out[n] = rec
		n++
		if n == len(out) {
			return n
		}
	}
	return n
}

// CountEqual is FilterEqual without an output buffer, for the case where only
// the cardinality is wanted. It never writes memory outside the arena's
// counters.
func (a *Arena) CountEqual(field uint32, needle []byte) uint64 {
	if a.isClosed() || uint64(field) >= a.fieldCount {
		return 0
	}
	atomic.AddUint64(&a.hdr.scanCalls.value, 1)

	want := uint32(len(needle))
	var np unsafe.Pointer
	if want != 0 {
		np = unsafe.Pointer(unsafe.SliceData(needle))
	}

	limit := a.RecordCount()
	p := a.columnRunBase(field)
	base := a.base
	var n uint64

	for rec := uint64(0); rec < limit; rec++ {
		d := (*fieldDescriptor)(p)
		p = unsafe.Add(p, descriptorSize)

		if atomic.LoadUint32(&d.payloadState) != stateCommitted {
			continue
		}
		if atomic.LoadUint32(&d.payloadLength) != want {
			continue
		}
		if want == 0 {
			n++
			continue
		}
		off := atomic.LoadUint64(&d.payloadOffset)
		if extentEqual(ptrAt(base, off), np, want) {
			n++
		}
	}
	return n
}

// FilterRange scans one column and writes the record IDs whose values compare
// within [lo, hi] under unsigned lexicographic byte ordering. Either bound may
// be nil to leave that side open. Returns the number of IDs written to out.
func (a *Arena) FilterRange(field uint32, lo, hi []byte, out []uint64) int {
	if a.isClosed() || uint64(field) >= a.fieldCount || len(out) == 0 {
		return 0
	}
	atomic.AddUint64(&a.hdr.scanCalls.value, 1)

	limit := a.RecordCount()
	p := a.columnRunBase(field)
	base := a.base
	n := 0

	for rec := uint64(0); rec < limit; rec++ {
		d := (*fieldDescriptor)(p)
		p = unsafe.Add(p, descriptorSize)

		if atomic.LoadUint32(&d.payloadState) != stateCommitted {
			continue
		}
		off := atomic.LoadUint64(&d.payloadOffset)
		length := atomic.LoadUint32(&d.payloadLength)
		value := unsafe.Slice((*byte)(ptrAt(base, off)), int(length))

		if lo != nil && compareBytes(value, lo) < 0 {
			continue
		}
		if hi != nil && compareBytes(value, hi) > 0 {
			continue
		}
		out[n] = rec
		n++
		if n == len(out) {
			return n
		}
	}
	return n
}

// Cursor is a zero-allocation forward iterator over one column. It is a value
// type with no pointers to the heap beyond the arena handle, so constructing one
// costs nothing and it can live entirely in registers inside a tight loop.
//
// A Cursor pins its record limit at construction, so records packed after it was
// created are not visited. That gives a sweep a stable horizon without taking a
// lock or blocking any packer.
type Cursor struct {
	arena *Arena
	p     unsafe.Pointer // rolling descriptor address
	rec   uint64         // next record to inspect
	limit uint64         // pinned horizon
}

// NewCursor pins a horizon and returns an iterator over the given column.
func (a *Arena) NewCursor(field uint32) Cursor {
	if a.isClosed() || uint64(field) >= a.fieldCount {
		return Cursor{}
	}
	return Cursor{
		arena: a,
		p:     a.columnRunBase(field),
		rec:   0,
		limit: a.RecordCount(),
	}
}

// Next advances to the next committed value, returning it, its record ID, and
// true. It returns ok=false once the pinned horizon is reached. Uncommitted,
// null, and overflowed cells are skipped.
//
// The returned slice aliases the arena and is valid until Reset or Close.
func (c *Cursor) Next() (recordID uint64, value []byte, ok bool) {
	if c.arena == nil {
		return 0, nil, false
	}
	for c.rec < c.limit {
		d := (*fieldDescriptor)(c.p)
		c.p = unsafe.Add(c.p, descriptorSize)
		rec := c.rec
		c.rec++

		if atomic.LoadUint32(&d.payloadState) != stateCommitted {
			continue
		}
		off := atomic.LoadUint64(&d.payloadOffset)
		length := atomic.LoadUint32(&d.payloadLength)
		return rec, unsafe.Slice((*byte)(ptrAt(c.arena.base, off)), int(length)), true
	}
	return 0, nil, false
}

// Remaining reports how many record slots the cursor has yet to inspect.
func (c *Cursor) Remaining() uint64 {
	if c.arena == nil || c.rec >= c.limit {
		return 0
	}
	return c.limit - c.rec
}

// ColumnBytes returns the dense, contiguous payload vector of one column as it
// currently stands: every byte ever appended to that column, back to back, in
// append order, including the 8-byte alignment padding between extents.
//
// This is the raw column image. It is the buffer to hand to a checksum, a
// compressor, or a write(2), and it is exactly the shape the arena stores, with
// no repacking step in between. The slice aliases the arena.
func (a *Arena) ColumnBytes(field uint32) ([]byte, bool) {
	if a.isClosed() || uint64(field) >= a.fieldCount {
		return nil, false
	}
	cs := a.column(field)
	used := atomic.LoadUint64(&cs.cursor)
	if used > cs.limit {
		used = cs.limit
	}
	return unsafe.Slice((*byte)(ptrAt(a.base, cs.baseOff)), int(used)), true
}

// DescriptorBytes returns one column's descriptor run as raw bytes: maxRecords
// contiguous 16-byte cells. Handed to a writer alongside ColumnBytes, the pair
// is a complete, self-contained image of the column.
func (a *Arena) DescriptorBytes(field uint32) ([]byte, bool) {
	if a.isClosed() || uint64(field) >= a.fieldCount {
		return nil, false
	}
	return unsafe.Slice((*byte)(a.columnRunBase(field)), int(a.maxRecords*descriptorSize)), true
}

// extentEqual compares n bytes at two addresses using 8-byte word loads.
//
// The arena-side address is always 8-byte aligned by construction; the needle
// side may not be, which is fine on amd64 and arm64. Comparing a word at a time
// means a mismatch in the first eight bytes costs one load and one branch, which
// is the common case for a selective filter over keys that share no prefix.
func extentEqual(a, b unsafe.Pointer, n uint32) bool {
	i := uint32(0)
	for ; i+8 <= n; i += 8 {
		if *(*uint64)(unsafe.Add(a, uintptr(i))) != *(*uint64)(unsafe.Add(b, uintptr(i))) {
			return false
		}
	}
	if i+4 <= n {
		if *(*uint32)(unsafe.Add(a, uintptr(i))) != *(*uint32)(unsafe.Add(b, uintptr(i))) {
			return false
		}
		i += 4
	}
	if i+2 <= n {
		if *(*uint16)(unsafe.Add(a, uintptr(i))) != *(*uint16)(unsafe.Add(b, uintptr(i))) {
			return false
		}
		i += 2
	}
	if i < n {
		if *(*byte)(unsafe.Add(a, uintptr(i))) != *(*byte)(unsafe.Add(b, uintptr(i))) {
			return false
		}
	}
	return true
}

// compareBytes is an unsigned lexicographic comparison, returning -1, 0, or 1.
// It is spelled out here rather than taken from bytes so that the fabric keeps
// its standard-library surface to syscall, unsafe, sync/atomic and errors.
func compareBytes(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			if a[i] < b[i] {
				return -1
			}
			return 1
		}
	}
	switch {
	case len(a) < len(b):
		return -1
	case len(a) > len(b):
		return 1
	}
	return 0
}
