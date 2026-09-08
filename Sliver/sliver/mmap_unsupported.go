//go:build !((linux || darwin) && (amd64 || arm64))

package sliver

import (
	"syscall"
	"unsafe"
)

// Sliver's arena is defined in terms of the raw mmap(2)/munmap(2) traps and a
// 64-bit address space. On any other platform the package still compiles, but
// Open fails cleanly rather than fabricating a heap-backed imitation that would
// silently reintroduce GC pressure.

const mapPopulateFlag = 0

func sysMmap(length uintptr, populate bool) (unsafe.Pointer, syscall.Errno) {
	return nil, syscall.ENOSYS
}

func sysMunmap(base unsafe.Pointer, length uintptr) syscall.Errno {
	return syscall.ENOSYS
}

func sysMlock(base unsafe.Pointer, length uintptr) error   { return syscall.ENOSYS }
func sysMunlock(base unsafe.Pointer, length uintptr) error { return syscall.ENOSYS }
