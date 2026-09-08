package sliver

import (
	"sync/atomic"
	"syscall"
	"unsafe"
)

// fabricError is the package's error type. It is a string constant under the
// hood, so every sentinel below is a compile-time value: returning one from a
// hot path boxes nothing and allocates nothing. Defining it here rather than
// calling errors.New keeps the fabric's entire import surface to syscall,
// unsafe and sync/atomic.
type fabricError string

func (e fabricError) Error() string { return string(e) }

// Errors returned by the fabric. All are sentinel values; none carry allocated
// context, so returning one from a hot path costs nothing.
const (
	ErrFieldCount    = fabricError("sliver: field count exceeds arena schema width")
	ErrRecordLimit   = fabricError("sliver: record capacity exhausted")
	ErrColumnFull    = fabricError("sliver: column payload vector exhausted")
	ErrFieldTooLarge = fabricError("sliver: field exceeds 4 GiB length descriptor")
	ErrClosed        = fabricError("sliver: arena is closed")
	ErrGeometry      = fabricError("sliver: invalid arena geometry")
	ErrMisaligned    = fabricError("sliver: kernel returned a misaligned mapping")
	ErrMlock         = fabricError("sliver: mlock refused, arena is not pinned")
)

// Config describes the geometry of an arena. Geometry is fixed for the lifetime
// of the mapping: a single mmap is taken at Open and the arena never grows,
// never remaps, and never copies. That is the whole point. A growable arena
// would have to move payload bytes, which would invalidate every zero-copy
// slice previously handed to a caller.
type Config struct {
	// FieldCount is the schema width: how many columns each record has. Sliver
	// is schema-free in the sense that it attaches no types, names, or tags to
	// a column; it only needs to know how many there are.
	FieldCount uint32

	// MaxRecords is the number of record slots in the descriptor index.
	MaxRecords uint64

	// ColumnCapacity gives each column's payload vector its size in bytes. If
	// nil, every column gets DefaultColumnCapacity. If non-nil its length must
	// equal FieldCount. Sizing columns individually is what keeps the arena
	// dense: a column of 8-byte keys does not reserve the same extent as a
	// column of kilobyte blobs.
	ColumnCapacity []uint64

	// DefaultColumnCapacity is the per-column payload budget used when
	// ColumnCapacity is nil.
	DefaultColumnCapacity uint64

	// RequireMlock makes a failed mlock(2) fatal to Open. Left false, a refused
	// lock (typically RLIMIT_MEMLOCK) is recorded in LockError and the arena is
	// still usable, just pageable.
	RequireMlock bool

	// Prefault forces every page of the arena resident before Open returns, so
	// that the first pass over the data does not pay a storm of minor faults.
	// On Linux this rides MAP_POPULATE; on Darwin the arena is walked one byte
	// per page.
	Prefault bool
}

// Arena is the off-heap columnar memory-packing engine.
//
// The struct itself is a handful of immutable words on the Go heap: a base
// pointer into the mapping and a cached copy of the geometry, so that the hot
// paths resolve an address without chasing through the header. Every piece of
// mutable state lives inside the mapping, cache-line isolated, and is touched
// only through sync/atomic. An Arena is safe for unlimited concurrent use by
// PackRecord, GetField, and the scan family.
//
// Arena contains no Go pointers into the mapping other than base, which points
// to memory the collector does not own; consequently the mark phase has nothing
// to traverse regardless of how many hundreds of millions of records are packed.
type Arena struct {
	base unsafe.Pointer // mapping origin, page aligned
	size uintptr        // mapping length in bytes

	hdr      *arenaHeader // == base
	colBase  unsafe.Pointer
	descBase unsafe.Pointer
	dataBase unsafe.Pointer

	fieldCount uint64
	maxRecords uint64

	locked    bool
	lockError error

	closed uint32 // atomic guard against double Close and use-after-Close
}

// Open computes the arena geometry, takes one contiguous anonymous mapping from
// the kernel, pins it, and writes the header. It is the only allocation the
// fabric ever performs.
func Open(cfg Config) (*Arena, error) {
	if cfg.FieldCount == 0 || cfg.FieldCount > MaxFields {
		return nil, ErrGeometry
	}
	if cfg.MaxRecords == 0 {
		return nil, ErrGeometry
	}
	if cfg.ColumnCapacity != nil && len(cfg.ColumnCapacity) != int(cfg.FieldCount) {
		return nil, ErrGeometry
	}
	if cfg.ColumnCapacity == nil && cfg.DefaultColumnCapacity == 0 {
		return nil, ErrGeometry
	}

	fields := uint64(cfg.FieldCount)
	records := cfg.MaxRecords

	// --- geometry -----------------------------------------------------------
	//
	// Region bases are laid out in ascending order and each is snapped up to a
	// cache line, which is what lets every later offset computation assume
	// alignment instead of re-deriving it.

	columnStateOffset := uint64(arenaHeaderSize)
	columnStateBytes := fields * columnStateSize
	if columnStateBytes/columnStateSize != fields { // multiplication overflow
		return nil, ErrGeometry
	}

	descriptorOffset := alignUp(columnStateOffset+columnStateBytes, CacheLineSize)

	descriptorCells := fields * records
	if descriptorCells/fields != records {
		return nil, ErrGeometry
	}
	descriptorBytes := descriptorCells * descriptorSize
	if descriptorBytes/descriptorSize != descriptorCells {
		return nil, ErrGeometry
	}

	dataOffset := alignUp(descriptorOffset+descriptorBytes, CacheLineSize)
	if dataOffset < descriptorOffset {
		return nil, ErrGeometry
	}

	// Each column's payload vector is a dense extent placed immediately after
	// the previous one, snapped up to a cache line so no two columns ever share
	// a line at their boundary. This pass only totals the extents; the identical
	// walk after the mapping succeeds writes the bases into the arena itself, so
	// no temporary slice of offsets is ever built.
	cursor := dataOffset
	for i := uint64(0); i < fields; i++ {
		capBytes := alignUp(columnCapacity(&cfg, i), CacheLineSize)
		next := cursor + capBytes
		if next < cursor {
			return nil, ErrGeometry
		}
		cursor = next
	}

	pageSize := uint64(syscall.Getpagesize())
	total := alignUp(cursor, pageSize)
	if total == 0 || total < cursor {
		return nil, ErrGeometry
	}

	// --- mapping ------------------------------------------------------------

	base, errno := sysMmap(uintptr(total), cfg.Prefault)
	if errno != 0 {
		return nil, errno
	}

	if !runtimeAssertAlignment(base, columnStateOffset, descriptorOffset, dataOffset) {
		sysMunmap(base, uintptr(total))
		return nil, ErrMisaligned
	}

	a := &Arena{
		base:       base,
		size:       uintptr(total),
		hdr:        (*arenaHeader)(base),
		colBase:    ptrAt(base, columnStateOffset),
		descBase:   ptrAt(base, descriptorOffset),
		dataBase:   ptrAt(base, dataOffset),
		fieldCount: fields,
		maxRecords: records,
	}

	// --- pin ----------------------------------------------------------------
	//
	// mlock keeps the arena resident so a scan can never take a major fault in
	// the middle of a filter, and so the kernel never writes record payloads to
	// swap. The slice handed to Mlock is a transient view over memory the
	// runtime does not manage; it is not retained.
	if err := sysMlock(base, uintptr(total)); err != nil {
		a.lockError = err
		if cfg.RequireMlock {
			sysMunmap(base, uintptr(total))
			return nil, ErrMlock
		}
	} else {
		a.locked = true
	}

	// Darwin has no MAP_POPULATE; fault the arena in by hand when asked.
	if cfg.Prefault {
		a.prefault(pageSize)
	}

	// --- header -------------------------------------------------------------
	//
	// The mapping arrives zero-filled from the kernel, so every counter and
	// every descriptor state word is already stateEmpty. Only the cold geometry
	// block needs writing.
	h := a.hdr
	h.magic = arenaMagic
	h.version = arenaVersion
	h.totalBytes = total
	h.fieldCount = fields
	h.maxRecords = records
	h.columnStateOffset = columnStateOffset
	h.descriptorOffset = descriptorOffset
	h.dataOffset = dataOffset

	// Replay the geometry walk, this time writing each column's base and limit
	// straight into its cache-line-isolated state block inside the mapping.
	place := dataOffset
	for i := uint64(0); i < fields; i++ {
		capBytes := alignUp(columnCapacity(&cfg, i), CacheLineSize)
		cs := a.column(uint32(i))
		cs.cursor = 0
		cs.baseOff = place
		cs.limit = capBytes
		cs.rejected = 0
		cs.appended = 0
		place += capBytes
	}

	return a, nil
}

// prefault walks one byte per page, forcing every page of the arena resident.
// The read is volatile enough that the compiler cannot elide it because the
// result feeds an atomic store into a counter the caller can observe.
func (a *Arena) prefault(pageSize uint64) {
	var acc uint64
	for off := uint64(0); off < uint64(a.size); off += pageSize {
		acc += uint64(*(*byte)(ptrAt(a.base, off)))
	}
	// Publish the accumulator to a sink so the compiler cannot prove the loop
	// dead and delete the faulting reads.
	atomic.StoreUint64(&prefaultSink, acc)
}

// prefaultSink absorbs the result of the prefault walk.
var prefaultSink uint64

// columnCapacity resolves the payload budget of column i from the config.
func columnCapacity(cfg *Config, i uint64) uint64 {
	if cfg.ColumnCapacity != nil {
		return cfg.ColumnCapacity[i]
	}
	return cfg.DefaultColumnCapacity
}

// column resolves the state block of column i.
//
//	address = colBase + i*128
//
// colBase is cache-line aligned and the stride is two cache lines, so cursor
// (offset 0 of the block) is always 64-byte aligned and never shares a line
// with the cursor of any other column.
func (a *Arena) column(i uint32) *columnState {
	return (*columnState)(unsafe.Add(a.colBase, uintptr(i)*columnStateSize))
}

// descriptor resolves the property descriptor for (field, record) in the
// column-major index array.
//
//	cell    = field*maxRecords + record
//	address = descBase + cell*16
//
// Column-major is deliberate: consecutive records of one column are consecutive
// descriptors in memory, so a filter over that column walks a straight line of
// 16-byte cells and the hardware prefetcher keeps up with no help. Row-major
// would stride by fieldCount*16 and touch one cache line per record per column.
func (a *Arena) descriptor(field uint32, record uint64) *fieldDescriptor {
	cell := uint64(field)*a.maxRecords + record
	return (*fieldDescriptor)(unsafe.Add(a.descBase, uintptr(cell)*descriptorSize))
}

// columnRunBase resolves the first descriptor of a column's contiguous run:
//
//	address = descBase + field*maxRecords*16
func (a *Arena) columnRunBase(field uint32) unsafe.Pointer {
	return unsafe.Add(a.descBase, uintptr(uint64(field)*a.maxRecords)*descriptorSize)
}

// FieldCount reports the schema width.
func (a *Arena) FieldCount() uint32 { return uint32(a.fieldCount) }

// MaxRecords reports the record slot capacity.
func (a *Arena) MaxRecords() uint64 { return a.maxRecords }

// SizeBytes reports the total mapped length.
func (a *Arena) SizeBytes() uint64 { return uint64(a.size) }

// Pinned reports whether mlock(2) succeeded.
func (a *Arena) Pinned() bool { return a.locked }

// LockError returns the mlock(2) failure, if the arena is unpinned.
func (a *Arena) LockError() error { return a.lockError }

// RecordCount returns the number of record IDs handed out, clamped to capacity.
// Some of those records may still be mid-pack; use CommittedCount for the count
// of records whose every field has been published.
func (a *Arena) RecordCount() uint64 {
	n := atomic.LoadUint64(&a.hdr.recordCursor.value)
	if n > a.maxRecords {
		return a.maxRecords
	}
	return n
}

// CommittedCount returns the number of records fully packed.
func (a *Arena) CommittedCount() uint64 {
	return atomic.LoadUint64(&a.hdr.commitCounter.value)
}

// ColumnStats is a snapshot of one column's payload vector.
type ColumnStats struct {
	Field    uint32
	BaseOff  uint64
	Used     uint64
	Capacity uint64
	Appended uint64
	Rejected uint64
}

// Stats is a snapshot of the arena's global trackers. It is returned by value
// and contains no pointers, so reading stats allocates nothing.
type Stats struct {
	TotalBytes     uint64
	FieldCount     uint64
	MaxRecords     uint64
	RecordCursor   uint64
	CommittedCount uint64
	PackFailures   uint64
	ReadMisses     uint64
	ScanCalls      uint64
	DescriptorOff  uint64
	DataOff        uint64
	Pinned         bool
}

// Stats reads every global counter. Each load touches a different cache line by
// construction, so this never perturbs a concurrent packer's line ownership.
func (a *Arena) Stats() Stats {
	return Stats{
		TotalBytes:     uint64(a.size),
		FieldCount:     a.fieldCount,
		MaxRecords:     a.maxRecords,
		RecordCursor:   atomic.LoadUint64(&a.hdr.recordCursor.value),
		CommittedCount: atomic.LoadUint64(&a.hdr.commitCounter.value),
		PackFailures:   atomic.LoadUint64(&a.hdr.packFailures.value),
		ReadMisses:     atomic.LoadUint64(&a.hdr.readMisses.value),
		ScanCalls:      atomic.LoadUint64(&a.hdr.scanCalls.value),
		DescriptorOff:  a.hdr.descriptorOffset,
		DataOff:        a.hdr.dataOffset,
		Pinned:         a.locked,
	}
}

// ColumnStats snapshots one column, or reports ok=false for an out-of-range
// field index.
func (a *Arena) ColumnStats(field uint32) (ColumnStats, bool) {
	if uint64(field) >= a.fieldCount {
		return ColumnStats{}, false
	}
	cs := a.column(field)
	return ColumnStats{
		Field:    field,
		BaseOff:  cs.baseOff,
		Used:     atomic.LoadUint64(&cs.cursor),
		Capacity: cs.limit,
		Appended: atomic.LoadUint64(&cs.appended),
		Rejected: atomic.LoadUint64(&cs.rejected),
	}, true
}

// Reset rewinds the arena to empty without unmapping or reallocating: every
// column cursor returns to zero and the descriptor cells that were in use are
// cleared back to stateEmpty. Only the used prefix of each column's descriptor
// run is touched, so resetting an arena that held a thousand records out of a
// hundred million slots costs a thousand cells per column, not a hundred million.
//
// Reset requires quiescence. It is not safe against a concurrent PackRecord,
// GetField, or scan; every slice previously returned by GetField dangles into
// space that the next pack will overwrite.
func (a *Arena) Reset() {
	used := a.RecordCount()

	for f := uint64(0); f < a.fieldCount; f++ {
		// Clear the descriptor run for this column: cells [0, used).
		run := a.columnRunBase(uint32(f))
		for r := uint64(0); r < used; r++ {
			d := (*fieldDescriptor)(unsafe.Add(run, uintptr(r)*descriptorSize))
			atomic.StoreUint32(&d.payloadState, stateEmpty)
			atomic.StoreUint32(&d.payloadLength, 0)
			atomic.StoreUint64(&d.payloadOffset, 0)
		}
		cs := a.column(uint32(f))
		atomic.StoreUint64(&cs.cursor, 0)
		atomic.StoreUint64(&cs.appended, 0)
		atomic.StoreUint64(&cs.rejected, 0)
	}

	atomic.StoreUint64(&a.hdr.commitCounter.value, 0)
	atomic.StoreUint64(&a.hdr.packFailures.value, 0)
	atomic.StoreUint64(&a.hdr.readMisses.value, 0)
	atomic.StoreUint64(&a.hdr.scanCalls.value, 0)
	// recordCursor last: it is the gate every reader consults first.
	atomic.StoreUint64(&a.hdr.recordCursor.value, 0)
}

// Close unpins and unmaps the arena. Every slice ever returned by GetField or
// handed to a scan callback points into the mapping and becomes invalid the
// instant Close returns; dereferencing one afterwards faults.
//
// Close is idempotent and returns ErrClosed on a second call.
func (a *Arena) Close() error {
	if !atomic.CompareAndSwapUint32(&a.closed, 0, 1) {
		return ErrClosed
	}
	base := a.base
	length := a.size

	if a.locked {
		_ = sysMunlock(base, length)
		a.locked = false
	}

	// Poison the handle before the mapping goes away so a use-after-Close hits
	// the nil check in the hot paths instead of faulting on freed address space.
	a.base = nil
	a.hdr = nil
	a.colBase = nil
	a.descBase = nil
	a.dataBase = nil

	if errno := sysMunmap(base, length); errno != 0 {
		return errno
	}
	return nil
}

// isClosed is the single-word guard the hot paths consult.
func (a *Arena) isClosed() bool { return atomic.LoadUint32(&a.closed) != 0 }
