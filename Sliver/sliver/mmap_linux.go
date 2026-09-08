//go:build linux && (amd64 || arm64)

package sliver

import (
	"syscall"
	"unsafe"
)

// mapPopulateFlag is MAP_POPULATE: pre-fault the whole mapping inside the
// kernel so the first pack does not take a storm of minor faults. The constant
// is spelled out because syscall does not export it on every Go release.
const mapPopulateFlag = 0x8000

// sysMmap acquires an anonymous, private, read-write mapping of exactly length
// bytes directly from the kernel via the raw mmap(2) trap. Nothing about this
// memory is known to the Go runtime: it is not a heap span, it is not in any
// size class, and the garbage collector's mark phase will never walk it.
//
// Linux amd64/arm64 argument order for SYS_MMAP:
//
//	a1 addr, a2 length, a3 prot, a4 flags, a5 fd, a6 offset
//
// addr = 0 lets the kernel choose; fd = -1 with MAP_ANONYMOUS means no backing
// file. offset must be 0 for anonymous mappings.
func sysMmap(length uintptr, populate bool) (unsafe.Pointer, syscall.Errno) {
	flags := uintptr(syscall.MAP_PRIVATE | syscall.MAP_ANONYMOUS)
	if populate {
		flags |= mapPopulateFlag
	}
	addr, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0,      // a1: let the kernel place it
		length, // a2: byte length
		uintptr(syscall.PROT_READ|syscall.PROT_WRITE), // a3: prot
		flags,       // a4: flags
		^uintptr(0), // a5: fd == -1
		0,           // a6: offset
	)
	if errno != 0 {
		return nil, errno
	}
	// The kernel returned a fresh mapping. Converting that uintptr to a Pointer
	// is the one place in the fabric where an integer becomes a pointer, and it
	// is sound because the address names live, page-aligned memory that the Go
	// runtime neither owns nor may move.
	return unsafe.Pointer(addr), 0 //nolint:govet // mmap result, see above
}

// sysMunmap releases the mapping through the raw munmap(2) trap.
func sysMunmap(base unsafe.Pointer, length uintptr) syscall.Errno {
	_, _, errno := syscall.Syscall(syscall.SYS_MUNMAP, uintptr(base), length, 0)
	return errno
}

// sysMlock pins the mapping in physical memory so that no page of the arena can
// be evicted to swap. A filter sweeping a pinned column can never take a major
// fault mid-scan, and record payloads never reach backing store.
func sysMlock(base unsafe.Pointer, length uintptr) error {
	return syscall.Mlock(unsafe.Slice((*byte)(base), length))
}

// sysMunlock releases the pin.
func sysMunlock(base unsafe.Pointer, length uintptr) error {
	return syscall.Munlock(unsafe.Slice((*byte)(base), length))
}
