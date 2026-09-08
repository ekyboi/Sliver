package sliver

import "unsafe"

// Compile-time structural verification of the 8-byte alignment matrix.
//
// Every declaration below is a constant expression evaluated by the compiler.
// unsafe.Sizeof, unsafe.Offsetof and unsafe.Alignof are constant expressions in
// Go, so the arithmetic here runs at build time and costs nothing at run time.
//
// The technique: convert a difference to an unsigned type. If the difference is
// non-zero in either direction, one of the two conversions in the pair is a
// negative constant, and a negative untyped constant converted to uint64 is a
// build failure ("constant -8 overflows uint64"). A pair of conversions is
// therefore an exact equality assertion, and a single conversion of a negated
// remainder is an exact divisibility assertion.
//
// If any invariant below is violated the package does not compile. There is no
// configuration under which Sliver can be built with a misaligned descriptor, a
// counter that shares a cache line, or a header whose Go layout has drifted
// from the byte offsets the pointer arithmetic assumes.

// --- exact equality: sizes and offsets --------------------------------------

const (
	// fieldDescriptor must be exactly 16 bytes, or the column-major descriptor
	// stride (index * descriptorSize) addresses the wrong entry.
	_ = uint64(descriptorSize - unsafe.Sizeof(fieldDescriptor{}))
	_ = uint64(unsafe.Sizeof(fieldDescriptor{}) - descriptorSize)

	// Field offsets inside the descriptor must match the documented layout,
	// because scan loops address the state and length words directly.
	_ = uint64(descriptorOffsetOffset - unsafe.Offsetof(fieldDescriptor{}.payloadOffset))
	_ = uint64(unsafe.Offsetof(fieldDescriptor{}.payloadOffset) - descriptorOffsetOffset)
	_ = uint64(descriptorLengthOffset - unsafe.Offsetof(fieldDescriptor{}.payloadLength))
	_ = uint64(unsafe.Offsetof(fieldDescriptor{}.payloadLength) - descriptorLengthOffset)
	_ = uint64(descriptorStateOffset - unsafe.Offsetof(fieldDescriptor{}.payloadState))
	_ = uint64(unsafe.Offsetof(fieldDescriptor{}.payloadState) - descriptorStateOffset)

	// columnState must be exactly two cache lines: one hot, one dead. A smaller
	// stride puts two columns' cursors in one coherence granule.
	_ = uint64(columnStateSize - unsafe.Sizeof(columnState{}))
	_ = uint64(unsafe.Sizeof(columnState{}) - columnStateSize)

	// The contended word must be at offset 0 of the stride, so that a
	// 64-byte-aligned array base makes every cursor 64-byte aligned.
	_ = uint64(0 - unsafe.Offsetof(columnState{}.cursor))

	// paddedCounter, same contract.
	_ = uint64(paddedCounterSize - unsafe.Sizeof(paddedCounter{}))
	_ = uint64(unsafe.Sizeof(paddedCounter{}) - paddedCounterSize)
	_ = uint64(0 - unsafe.Offsetof(paddedCounter{}.value))

	// arenaHeader size and the offset of every hot counter inside it.
	_ = uint64(arenaHeaderSize - unsafe.Sizeof(arenaHeader{}))
	_ = uint64(unsafe.Sizeof(arenaHeader{}) - arenaHeaderSize)

	_ = uint64(headerRecordCursorOff - unsafe.Offsetof(arenaHeader{}.recordCursor))
	_ = uint64(unsafe.Offsetof(arenaHeader{}.recordCursor) - headerRecordCursorOff)
	_ = uint64(headerCommitCounterOff - unsafe.Offsetof(arenaHeader{}.commitCounter))
	_ = uint64(unsafe.Offsetof(arenaHeader{}.commitCounter) - headerCommitCounterOff)
	_ = uint64(headerPackFailuresOff - unsafe.Offsetof(arenaHeader{}.packFailures))
	_ = uint64(unsafe.Offsetof(arenaHeader{}.packFailures) - headerPackFailuresOff)
	_ = uint64(headerReadMissesOff - unsafe.Offsetof(arenaHeader{}.readMisses))
	_ = uint64(unsafe.Offsetof(arenaHeader{}.readMisses) - headerReadMissesOff)
	_ = uint64(headerScanCallsOff - unsafe.Offsetof(arenaHeader{}.scanCalls))
	_ = uint64(unsafe.Offsetof(arenaHeader{}.scanCalls) - headerScanCallsOff)

	// The cold geometry block must be fenced from the first hot counter by a
	// full isolation line.
	_ = uint64(headerColdTrailOffset - unsafe.Offsetof(arenaHeader{}._coldTrail))
	_ = uint64(unsafe.Offsetof(arenaHeader{}._coldTrail) - headerColdTrailOffset)
)

// --- divisibility: the 8-byte alignment matrix ------------------------------

const (
	// Every structural stride must be a whole number of 8-byte words, so that
	// stepping by the stride can never move a pointer off an aligned boundary.
	_ = uint64(0 - (descriptorSize % WordAlign))
	_ = uint64(0 - (columnStateSize % WordAlign))
	_ = uint64(0 - (paddedCounterSize % WordAlign))
	_ = uint64(0 - (arenaHeaderSize % WordAlign))

	// Every field offset inside a descriptor must be 4-byte aligned, and the
	// 64-bit offset word must be 8-byte aligned to be a legal atomic target.
	_ = uint64(0 - (descriptorOffsetOffset % WordAlign))
	_ = uint64(0 - (descriptorLengthOffset % 4))
	_ = uint64(0 - (descriptorStateOffset % 4))

	// The regions that follow the header and the column-state array must begin
	// on a cache line, which is what makes every counter and every descriptor
	// run inherit 64-byte alignment from its region base.
	_ = uint64(0 - (arenaHeaderSize % CacheLineSize))
	_ = uint64(0 - (columnStateSize % CacheLineSize))
	_ = uint64(0 - (paddedCounterSize % CacheLineSize))

	// The isolation stride must be at least one full coherence granule.
	_ = uint64(columnStateSize - CacheLineSize)
	_ = uint64(paddedCounterSize - CacheLineSize)

	// The Go compiler must agree that a uint64 wants 8-byte alignment; if it
	// ever did not, the atomic accesses in this package would be illegal.
	_ = uint64(WordAlign - unsafe.Alignof(uint64(0)))
	_ = uint64(unsafe.Alignof(uint64(0)) - WordAlign)
)

// runtimeAssertAlignment re-verifies at Open time the one property the compiler
// cannot see: that the address the kernel returned, plus each computed region
// offset, is actually aligned. mmap is required to return page-aligned memory,
// and every offset is built with alignUp, so this can only fire if the kernel
// or the geometry computation is broken. It is cheap and runs once per arena.
func runtimeAssertAlignment(base unsafe.Pointer, offsets ...uint64) bool {
	if uintptr(base)%CacheLineSize != 0 {
		return false
	}
	for _, off := range offsets {
		if off%CacheLineSize != 0 {
			return false
		}
		if (uintptr(base)+uintptr(off))%WordAlign != 0 {
			return false
		}
	}
	return true
}
