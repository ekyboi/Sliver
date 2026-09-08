// Package sliver implements the memory-packing engine of a schema-free
// columnar serialization fabric.
//
// Sliver stores tabular records in a single contiguous, off-heap arena obtained
// straight from the kernel with mmap(2) and pinned with mlock(2). The Go garbage
// collector never scans, marks, or moves a single byte of that arena: it holds no
// Go pointers, it is not part of any span, and it is invisible to the mark phase.
// Encoding and decoding are performed exclusively with explicit unsafe.Pointer
// byte-offset arithmetic. The reflect package is never imported, no code is
// generated, and the steady-state pack/get/scan paths allocate zero bytes.
//
// Arena geography, in one contiguous mapping:
//
//	byte 0                            arenaHeader        (768 B, cache-line isolated)
//	byte hdr.columnStateOffset        columnState[F]     (128 B stride, one per column)
//	byte hdr.descriptorOffset         fieldDescriptor[F][R]  column-major index array
//	byte hdr.dataOffset               column-packed payload vectors, dense, back to back
//
// The descriptor index array is column-major: every descriptor belonging to one
// logical column occupies one uninterrupted run of memory, so a filter over a
// single column streams the hardware prefetcher in a straight line and never
// pulls in a byte of any other column. The payload vectors mirror that decision;
// column i's bytes are one dense extent, disjoint from every other column's.
package sliver

import "unsafe"

// Structural constants of the fabric. These are the numbers the compile-time
// assertions in asserts.go are checked against; changing one without changing
// the corresponding struct is a build failure, not a runtime surprise.
const (
	// CacheLineSize is the coherence granule on amd64 and arm64. Every mutable
	// counter in the arena is isolated to its own multiple of this.
	CacheLineSize = 64

	// WordAlign is the mandatory alignment of every column base, every payload
	// extent, and every length descriptor. Nothing in the arena is permitted to
	// straddle an 8-byte boundary, which is what guarantees that a field seek
	// never costs a split-line access.
	WordAlign = 8

	// MaxFields caps the descriptor fan-out of a single arena.
	MaxFields = 4096

	// MaxFieldBytes is the largest single field payload, bounded by the uint32
	// length descriptor.
	MaxFieldBytes = 1<<32 - 1

	// arenaMagic tags the header so a mapping can be sanity-checked cheaply.
	arenaMagic   = 0x53_4C_49_56_45_52_01_00 // "SLIVER\x01\x00"
	arenaVersion = 1
)

// Descriptor publication states. A descriptor is only readable once its state
// word has been released with a store; that store is the linearization point
// that makes the payload bytes written before it visible to every reader.
const (
	stateEmpty     uint32 = 0 // never written
	stateCommitted uint32 = 1 // payload bytes are complete and visible
	stateNull      uint32 = 2 // field explicitly absent for this record
	stateOverflow  uint32 = 3 // column ran out of space; payload was dropped
)

// fieldDescriptor is the property descriptor: one entry per (column, record).
//
// Layout, exactly 16 bytes, so descriptor n begins at 16*n and every field
// inside it lands on its natural alignment:
//
//	+0  payloadOffset uint64  arena-relative byte offset of the payload extent
//	+8  payloadLength uint32  payload length in bytes (the variable-length descriptor)
//	+12 payloadState  uint32  publication state, the release/acquire gate
//
// payloadOffset sits at offset 0 of a 16-byte-strided array whose base is
// 64-byte aligned, so it is always 8-byte aligned and therefore a legal target
// for a 64-bit atomic on every architecture Go supports. payloadLength and
// payloadState share the second 8-byte word; both are 4-byte aligned.
type fieldDescriptor struct {
	payloadOffset uint64
	payloadLength uint32
	payloadState  uint32
}

const (
	descriptorSize         = 16
	descriptorOffsetOffset = 0
	descriptorLengthOffset = 8
	descriptorStateOffset  = 12
)

// columnState is the mutable, contended head of one column's payload vector.
//
// Concurrent packers hammer cursor with atomic fetch-and-add. If two columns'
// cursors shared a cache line, two threads appending to two different columns
// would serialize on the coherence protocol despite touching disjoint data.
// The hot words are therefore grouped into the first line, and an explicit
// 64-byte trailing padding block follows, giving a 128-byte stride: one line
// for the counters, one dead line behind them. The dead line also absorbs the
// adjacent-line prefetcher on Apple silicon and on Intel's spatial prefetcher,
// both of which pull line pairs and would otherwise reintroduce the sharing
// that a bare 64-byte stride is supposed to eliminate.
type columnState struct {
	cursor   uint64 // +0  bytes handed out, monotonically increasing
	baseOff  uint64 // +8  arena-relative base of this column's payload vector
	limit    uint64 // +16 capacity in bytes of that vector
	rejected uint64 // +24 appends refused because the column was full
	appended uint64 // +32 payload extents successfully committed
	_hotPad  [CacheLineSize - 40]byte
	_trail   [CacheLineSize]byte // explicit 64-byte isolation block
}

const columnStateSize = 2 * CacheLineSize // 128

// paddedCounter is a single global tracker owning two full cache lines, for the
// same reason columnState does. The value word is first so that the containing
// array's 64-byte-aligned base makes every value 64-byte aligned.
type paddedCounter struct {
	value  uint64
	_hot   [CacheLineSize - 8]byte
	_trail [CacheLineSize]byte // explicit 64-byte isolation block
}

const paddedCounterSize = 2 * CacheLineSize // 128

// arenaHeader is the first object in the mapping. Its cold half is written once
// during Open and never again; its hot half is a run of mutually isolated
// counters. The cold half is fenced off from the counters by its own trailing
// padding block so that a reader touching immutable geometry never shares a line
// with a writer incrementing recordCursor.
type arenaHeader struct {
	// Cold: exactly one cache line of immutable geometry.
	magic             uint64 // +0
	version           uint64 // +8
	totalBytes        uint64 // +16
	fieldCount        uint64 // +24
	maxRecords        uint64 // +32
	columnStateOffset uint64 // +40
	descriptorOffset  uint64 // +48
	dataOffset        uint64 // +56
	_coldTrail        [CacheLineSize]byte

	// Hot: one counter per line pair.
	recordCursor  paddedCounter // record IDs handed out
	commitCounter paddedCounter // records fully packed
	packFailures  paddedCounter // packs refused or partially dropped
	readMisses    paddedCounter // GetField calls that found nothing readable
	scanCalls     paddedCounter // column scans started
}

const (
	arenaHeaderSize        = 2*CacheLineSize + 5*paddedCounterSize // 768
	headerColdTrailOffset  = CacheLineSize                         // 64
	headerRecordCursorOff  = 2 * CacheLineSize                     // 128
	headerCommitCounterOff = headerRecordCursorOff + paddedCounterSize
	headerPackFailuresOff  = headerCommitCounterOff + paddedCounterSize
	headerReadMissesOff    = headerPackFailuresOff + paddedCounterSize
	headerScanCallsOff     = headerReadMissesOff + paddedCounterSize
)

// alignUp rounds v up to the next multiple of a, which must be a power of two.
// Used for every offset the arena hands out, which is what keeps the 8-byte
// alignment matrix an invariant rather than an aspiration.
func alignUp(v, a uint64) uint64 { return (v + a - 1) &^ (a - 1) }

// ptrAt performs the one and only address computation in the fabric: arena base
// plus an arena-relative byte offset. Every read, write, and slice in Sliver
// bottoms out here.
func ptrAt(base unsafe.Pointer, off uint64) unsafe.Pointer {
	return unsafe.Add(base, uintptr(off))
}
