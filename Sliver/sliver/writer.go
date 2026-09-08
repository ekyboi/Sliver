package sliver

import (
	"sync/atomic"
	"unsafe"
)

// Writer is a per-goroutine packing handle that amortizes the arena's atomic
// reservations across a block of records.
//
// Arena.PackRecord is thread-safe on its own, but it pays for that with one
// atomic read-modify-write on the shared record cursor plus one per column, per
// record. Those cursors are the only lines in the fabric that every packer
// touches, so at eight concurrent packers the coherence traffic on them, not the
// copy, becomes the cost of a pack.
//
// A Writer removes that traffic by reserving records and column bytes in blocks:
// one atomic add buys RecordBatch record slots and one buys a run of column
// bytes, after which the writer hands out both from goroutine-local variables
// with no atomics at all. The per-record atomic count drops from (1 + 2*columns)
// to roughly (1 + 2*columns)/batch.
//
// A Writer is NOT safe for concurrent use. Give each goroutine its own; that is
// the entire point. The Arena behind them stays shared and any number of
// Writers, PackRecord callers, and readers may run against it at once.
//
// The cost of blocking is bounded waste: when a writer stops, the unused tail of
// its record block is never filled (those descriptors stay stateEmpty and read
// back as absent) and the unused tail of each column block is never written.
// Total waste is bounded by (writers x RecordBatch) slots and
// (writers x ColumnBatch) bytes.
type Writer struct {
	arena *Arena

	// Goroutine-local record block: IDs [recNext, recEnd) are ours alone.
	recNext uint64
	recEnd  uint64
	batch   uint64

	// Goroutine-local column blocks, one entry per column. colNext[f] and
	// colEnd[f] are arena-relative byte offsets into column f's payload vector.
	colNext  []uint64
	colEnd   []uint64
	colBatch []uint64

	// Local statistics, folded into the arena's shared counters on Flush so the
	// hot path never touches a contended counter line.
	committed uint64
	failures  uint64
	appended  []uint64
	rejected  []uint64

	// Trailing isolation so two Writers living in one slice cannot share a line.
	_pad [CacheLineSize]byte
}

// WriterConfig tunes the block sizes a Writer reserves.
type WriterConfig struct {
	// RecordBatch is how many record IDs to claim per reservation. Larger
	// batches mean fewer atomics and more waste at the tail. 512 is a good
	// default: it cuts the record-cursor traffic by more than two orders of
	// magnitude while wasting at most 512 slots per idle writer.
	RecordBatch uint64

	// ColumnBatch is how many bytes to claim per column per reservation. If
	// zero, it is derived as RecordBatch x 64, which covers a batch of
	// medium-sized fields without a refill.
	ColumnBatch uint64
}

const (
	defaultRecordBatch = 512
	defaultColumnScale = 64
)

// NewWriter creates a packing handle bound to this arena. It allocates its
// bookkeeping slices once, here; every subsequent pack allocates nothing.
func (a *Arena) NewWriter(cfg WriterConfig) (*Writer, error) {
	if a.isClosed() {
		return nil, ErrClosed
	}
	if cfg.RecordBatch == 0 {
		cfg.RecordBatch = defaultRecordBatch
	}
	if cfg.ColumnBatch == 0 {
		cfg.ColumnBatch = cfg.RecordBatch * defaultColumnScale
	}

	f := int(a.fieldCount)
	w := &Writer{
		arena:    a,
		batch:    cfg.RecordBatch,
		colNext:  make([]uint64, f),
		colEnd:   make([]uint64, f),
		colBatch: make([]uint64, f),
		appended: make([]uint64, f),
		rejected: make([]uint64, f),
	}
	for i := 0; i < f; i++ {
		cb := alignUp(cfg.ColumnBatch, WordAlign)
		if lim := a.column(uint32(i)).limit; cb > lim {
			cb = lim
		}
		w.colBatch[i] = cb
	}
	return w, nil
}

// PackRecord appends one record using the writer's private blocks. It has the
// same semantics as Arena.PackRecord and the same zero-allocation guarantee, but
// executes zero atomic operations in the common case.
func (w *Writer) PackRecord(fields [][]byte) (uint64, error) {
	a := w.arena
	if a.isClosed() {
		return 0, ErrClosed
	}
	if uint64(len(fields)) > a.fieldCount {
		w.failures++
		return 0, ErrFieldCount
	}

	// Record slot, from the local block. Only the refill touches an atomic.
	if w.recNext == w.recEnd {
		if err := w.refillRecords(); err != nil {
			w.failures++
			return 0, err
		}
	}
	record := w.recNext
	w.recNext++

	supplied := uint64(len(fields))
	var firstErr error

	for f := uint64(0); f < a.fieldCount; f++ {
		d := (*fieldDescriptor)(unsafe.Add(
			a.descBase,
			uintptr(f*a.maxRecords+record)*descriptorSize,
		))

		if f >= supplied || fields[f] == nil {
			atomic.StoreUint64(&d.payloadOffset, 0)
			atomic.StoreUint32(&d.payloadLength, 0)
			atomic.StoreUint32(&d.payloadState, stateNull)
			continue
		}

		payload := fields[f]
		n := uint64(len(payload))
		if n > MaxFieldBytes {
			atomic.StoreUint32(&d.payloadState, stateOverflow)
			if firstErr == nil {
				firstErr = ErrFieldTooLarge
			}
			continue
		}
		need := alignUp(n, WordAlign)

		// Payload extent, from the local block:
		//
		//   off = colNext[f]; colNext[f] += need
		//
		// a plain add on a goroutine-local variable, no atomic, no contention.
		if w.colNext[f]+need > w.colEnd[f] {
			if err := w.refillColumn(uint32(f), need); err != nil {
				w.rejected[f]++
				atomic.StoreUint64(&d.payloadOffset, 0)
				atomic.StoreUint32(&d.payloadLength, 0)
				atomic.StoreUint32(&d.payloadState, stateOverflow)
				if firstErr == nil {
					firstErr = err
				}
				continue
			}
		}
		off := w.colNext[f]
		w.colNext[f] = off + need

		if n != 0 {
			writeExtent(ptrAt(a.base, off), payload, n)
		}

		// Publish with the same release ordering as Arena.PackRecord: readers
		// cannot tell whether a record came from a Writer or not.
		atomic.StoreUint64(&d.payloadOffset, off)
		atomic.StoreUint32(&d.payloadLength, uint32(n))
		atomic.StoreUint32(&d.payloadState, stateCommitted)
		w.appended[f]++
	}

	w.committed++
	if firstErr != nil {
		w.failures++
	}
	return record, firstErr
}

// refillRecords claims the next block of record IDs with a single atomic add.
func (w *Writer) refillRecords() error {
	a := w.arena
	want := w.batch
	end := atomic.AddUint64(&a.hdr.recordCursor.value, want)
	start := end - want

	if start >= a.maxRecords {
		atomic.CompareAndSwapUint64(&a.hdr.recordCursor.value, end, a.maxRecords)
		return ErrRecordLimit
	}
	if end > a.maxRecords {
		// Partial block at the ceiling: take what exists and clamp the cursor.
		atomic.CompareAndSwapUint64(&a.hdr.recordCursor.value, end, a.maxRecords)
		end = a.maxRecords
	}
	w.recNext, w.recEnd = start, end
	return nil
}

// refillColumn claims the next run of bytes in column f with a single atomic
// add. The remaining tail of the previous block is abandoned.
//
// If the block-sized grab would run past the column's capacity, the grab is
// handed back with a CAS and retried at exactly the size this record needs, so
// a nearly-full column keeps accepting small values right up to its limit
// instead of being closed early by an oversized speculative reservation.
func (w *Writer) refillColumn(f uint32, need uint64) error {
	cs := w.arena.column(f)

	grab := w.colBatch[f]
	if grab < need {
		grab = alignUp(need, WordAlign)
	}

	end := atomic.AddUint64(&cs.cursor, grab)
	if end <= cs.limit && end >= grab {
		w.colNext[f] = cs.baseOff + end - grab
		w.colEnd[f] = cs.baseOff + end
		return nil
	}

	// Give the speculative block back if nobody has moved the cursor since.
	atomic.CompareAndSwapUint64(&cs.cursor, end, end-grab)

	if grab == need {
		return ErrColumnFull
	}
	end = atomic.AddUint64(&cs.cursor, need)
	if end > cs.limit || end < need {
		atomic.CompareAndSwapUint64(&cs.cursor, end, end-need)
		return ErrColumnFull
	}
	w.colNext[f] = cs.baseOff + end - need
	w.colEnd[f] = cs.baseOff + end
	return nil
}

// Flush folds the writer's local statistics into the arena's shared counters. It
// does not affect the visibility of packed records: those were published by the
// release store in PackRecord and were readable the instant it returned.
//
// Call Flush before reading Arena.Stats if the counts matter, and always before
// dropping a Writer.
func (w *Writer) Flush() {
	a := w.arena
	if a.isClosed() {
		return
	}
	if w.committed != 0 {
		atomic.AddUint64(&a.hdr.commitCounter.value, w.committed)
		w.committed = 0
	}
	if w.failures != 0 {
		atomic.AddUint64(&a.hdr.packFailures.value, w.failures)
		w.failures = 0
	}
	for f := uint64(0); f < a.fieldCount; f++ {
		cs := a.column(uint32(f))
		if w.appended[f] != 0 {
			atomic.AddUint64(&cs.appended, w.appended[f])
			w.appended[f] = 0
		}
		if w.rejected[f] != 0 {
			atomic.AddUint64(&cs.rejected, w.rejected[f])
			w.rejected[f] = 0
		}
	}
}

// Reserved reports how many record slots remain in the writer's current block.
func (w *Writer) Reserved() uint64 { return w.recEnd - w.recNext }
