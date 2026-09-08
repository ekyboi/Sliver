# Sliver

A schema-free columnar serialization fabric for Go. Sliver packs millions of
tabular records into a single contiguous, off-heap, page-locked memory arena and
reads them back with pure pointer arithmetic — no reflection, no code
generation, no allocation, and nothing for the garbage collector to mark.

```
Arena.PackRecord      31.6 ns/op   3.42 GB/s   0 allocs      6-field record
Writer.PackRecord     22.8 ns/op   4.74 GB/s   0 allocs      same record
GetField               1.77 ns/op               0 allocs
FilterEqual            1.63 ns/row              0 allocs      1M-row column sweep
encoding/json          927 ns/op   0.12 GB/s   8 allocs      same record, round trip
```

## The problem

Sorting and filtering millions of tabular records through ordinary Go
serialization is dominated by costs that have nothing to do with the data:

- **Reflection.** `encoding/json` and friends walk type metadata at runtime, per
  field, per record.
- **Cache fragmentation.** A `[]Record` of structs holding `[]byte` and `string`
  scatters payloads across the heap. Filtering one column drags every other
  column's bytes through L1 alongside it.
- **GC mark pressure.** Ten million records with four `[]byte` fields each is
  forty million pointers the collector must trace on every cycle. The mark phase
  cost scales with your working set whether or not any of it is garbage.

Sliver removes all three. Records live in memory the collector cannot see,
columns are stored separately so a filter touches only what it reads, and every
field access is an address computation.

## Guarantees

| | |
|---|---|
| **Dependencies** | `syscall`, `unsafe`, `sync/atomic`. Nothing else, including no `errors`. |
| **Allocation** | Zero on every pack, read, filter, and scan path. Enforced by `AllocsPerRun` tests. |
| **GC** | The arena holds no Go pointers. 400k records / 51 MiB of payload measured a **0**-object heap delta. |
| **Reflection** | None. No `reflect` import, no `go:generate`, no build step. |
| **Alignment** | Every column base, payload extent, and length descriptor is 8-byte aligned; every mutable counter is 64-byte aligned. Verified at compile time. |
| **Concurrency** | `Arena` is safe for unlimited concurrent packers and readers. Race-detector clean. |

## Requirements

Go 1.21+. Linux or Darwin on amd64 or arm64. The package compiles everywhere,
but `Open` returns `ENOSYS` on other platforms rather than falling back to a
heap-backed imitation that would quietly reintroduce the GC pressure it exists
to remove.

```
go get github.com/sliver/sliver
```

The package lives at `github.com/sliver/sliver/sliver`. If you would rather
import it as `github.com/sliver/sliver`, move the `.go` files to the repository
root; nothing in the code depends on the directory.

## Quickstart

```go
package main

import (
	"fmt"

	"github.com/sliver/sliver/sliver"
)

func main() {
	// One mmap, taken once. Geometry is fixed for the arena's lifetime.
	a, err := sliver.Open(sliver.Config{
		FieldCount:            3,               // three columns
		MaxRecords:            1 << 20,         // one million record slots
		DefaultColumnCapacity: 64 << 20,        // 64 MiB of payload per column
	})
	if err != nil {
		panic(err)
	}
	defer a.Close()

	// Pack. The fields slice is reused; nothing is allocated per record.
	fields := make([][]byte, 3)
	for i := 0; i < 100000; i++ {
		fields[0] = []byte(fmt.Sprintf("user-%d", i%1000))
		fields[1] = []byte("us-east-1")
		fields[2] = []byte(fmt.Sprintf("%d", i))
		if _, err := a.PackRecord(fields); err != nil {
			panic(err)
		}
	}

	// Read. The returned slice aliases the arena directly — zero copy.
	if v, ok := a.GetField(42, 0); ok {
		fmt.Printf("record 42, column 0 = %s\n", v)
	}

	// Filter one column. Matching record IDs land in a caller-owned buffer.
	out := make([]uint64, 256)
	n := a.FilterEqual(0, []byte("user-7"), out)
	fmt.Printf("%d records match, first: %v\n", n, out[:min(n, 5)])

	// Sweep a column with no intermediate collection.
	var bytes int
	a.ScanColumn(2, func(recordID uint64, value []byte) bool {
		bytes += len(value)
		return true // false stops early
	})
	fmt.Printf("column 2 holds %d payload bytes\n", bytes)
}
```

## Concurrency: pick the right packer

`Arena.PackRecord` is thread-safe on its own, but it pays for that with one
atomic read-modify-write on the shared record cursor plus two per column, per
record. Those cursors are the only cache lines every packer touches, so past a
couple of cores the coherence traffic on them — not the copy — becomes the cost
of a pack. Padding cannot fix this; the lines are genuinely shared.

`Writer` is a per-goroutine handle that reserves records and column bytes in
**blocks**. One atomic buys 512 record slots, after which it hands them out from
goroutine-local variables with no atomics at all.

| Packer | 1 core | 8 cores | ops in the same wall time (8 cores) |
|---|---|---|---|
| `Arena.PackRecord` | 31.9 ns/op | **279.5 ns/op** | 1,281,134 |
| `Writer.PackRecord` | 24.0 ns/op | **27.6 ns/op** | 59,202,813 |

```go
// One Writer per goroutine. Never share one.
w, err := a.NewWriter(sliver.WriterConfig{RecordBatch: 512})
if err != nil {
	return err
}
defer w.Flush() // folds local counters into Arena.Stats

for _, row := range rows {
	if _, err := w.PackRecord(row); err != nil {
		return err
	}
}
```

Use `Arena.PackRecord` for low-concurrency or occasional writes. Use `Writer`
whenever more than one goroutine packs. The two are fully interoperable — a
reader cannot tell which produced a record.

The cost of blocking is bounded waste: an idle writer never fills the tail of
its block. Those slots read back as `StateEmpty`, never as garbage.

## Memory layout

One contiguous `mmap` region, `mlock`ed, laid out in four regions. Every region
base is snapped to a cache line, so every offset inside it inherits alignment
instead of re-deriving it.

```
byte 0            ┌────────────────────────────────────────────┐
                  │ arenaHeader                        768 B   │
                  │  ├─ cold geometry (magic, offsets)  64 B   │
                  │  ├─ 64 B isolation block                   │
                  │  └─ 5 counters, 128 B apart                │  ← one per
                  │     recordCursor, commitCounter, …         │    cache line
hdr.columnStateOffset ────────────────────────────────────────┤
                  │ columnState[F]        128 B stride         │  ← cursor,
                  │  cursor │ base │ limit │ stats │ 64 B pad   │    own line
hdr.descriptorOffset ─────────────────────────────────────────┤
                  │ fieldDescriptor[F][R]   COLUMN-MAJOR       │
                  │  column 0: rec 0,1,2,…R    16 B each       │  ← one dense
                  │  column 1: rec 0,1,2,…R                    │    run per
                  │  …                                         │    column
hdr.dataOffset  ──┼────────────────────────────────────────────┤
                  │ column 0 payload vector (dense, 8-aligned) │
                  │ column 1 payload vector                    │
                  │ …                                          │
                  └────────────────────────────────────────────┘
```

**The descriptor index is column-major, and that is the load-bearing decision.**
All `R` descriptors of one column occupy one uninterrupted run, so filtering
column 3 of ten million records walks a straight 16-byte stride and the hardware
prefetcher keeps up unaided. It never pulls in a byte of any other column.
Row-major would stride by `fieldCount × 16` and touch one cache line per record
per column.

A descriptor is exactly 16 bytes:

```
+0   payloadOffset  uint64   arena-relative byte offset of the payload
+8   payloadLength  uint32   the variable-length descriptor
+12  payloadState   uint32   publication gate (release store / acquire load)
```

The two address computations that the entire fabric reduces to:

```
descriptor = descBase  + (fieldIndex*maxRecords + recordID) * 16
payload    = arenaBase + descriptor.payloadOffset
```

### False-sharing defeat

`columnState` and `paddedCounter` are 128 bytes: one line of hot words followed
by an explicit 64-byte trailing padding block. The second line matters — a bare
64-byte stride is defeated by the adjacent-line prefetcher on Apple silicon and
by Intel's spatial prefetcher, both of which pull line pairs and would reintroduce
exactly the sharing the padding is meant to eliminate.

### The 8-byte alignment matrix

Every payload reservation is rounded up to 8 bytes from a 64-byte-aligned column
base, so no extent can ever land off an aligned boundary regardless of what
lengths were packed before it. `asserts.go` checks this at **compile time** using
constant offset arithmetic — a wrong layout is a build failure, not a runtime
surprise:

```go
// Exact equality: if the size differs in either direction, one of these
// is a negative constant, and a negative constant does not convert to uint64.
_ = uint64(descriptorSize - unsafe.Sizeof(fieldDescriptor{}))
_ = uint64(unsafe.Sizeof(fieldDescriptor{}) - descriptorSize)
```

Verified by deliberately breaking it. Shrinking the `columnState` isolation pad:

```
sliver/asserts.go:43:13: unsafe.Sizeof(columnState{}) - columnStateSize
                         (constant -8 of type uintptr) overflows uintptr
```

There is no configuration in which Sliver builds with a misaligned descriptor or
a counter that shares a cache line.

## Sizing the arena

```
total = 768                                   header
      + FieldCount × 128                      column state blocks
      + FieldCount × MaxRecords × 16          descriptor index
      + Σ column capacities                   payload vectors (each 64 B aligned)
      → rounded up to the page size
```

The descriptor index is a fixed 16 bytes per field per record and is sized to
`MaxRecords`, not to how many records you actually pack. At 6 columns and 10M
record slots that is 960 MB of index before a single byte of payload.

**Because the arena is `mlock`ed, all of it is resident immediately.** Measured
on darwin/arm64: a 512 MiB anonymous mapping costs ~96 KB of RSS on its own
(pages are demand-zero and faulted lazily), but the same mapping pinned costs the
full 524,288 KB. Size `MaxRecords` to what you will actually use, or set
`RequireMlock: false` and accept a pageable arena.

If `mlock` is refused — usually `RLIMIT_MEMLOCK`, see `ulimit -l` — `Open`
succeeds anyway and records the failure. Check it:

```go
if !a.Pinned() {
	log.Printf("sliver: arena is pageable: %v", a.LockError())
}
```

Set `Config.RequireMlock: true` to make that fatal instead.

## API

**Lifecycle** — `Open(Config) (*Arena, error)`, `Close`, `Reset`

**Packing** — `PackRecord`, `ReserveRecord`, `PackField`, `NewWriter`

**Reading** — `GetField` (zero-copy), `CopyField` (into your buffer),
`FieldLen`, `FieldState`

**Scanning** — `ScanColumn`, `FilterEqual`, `CountEqual`, `FilterRange`,
`NewCursor`

**Bulk export** — `ColumnBytes`, `DescriptorBytes`. Together these are a
complete, self-contained image of a column in exactly the shape the arena stores
it — hand them to a checksum, a compressor, or `write(2)` with no repacking step.

**Introspection** — `Stats`, `ColumnStats`, `RecordCount`, `CommittedCount`,
`Pinned`, `LockError`

### Field states

`GetField` returns `ok == false` for four distinguishable reasons. Use
`FieldState` to tell them apart:

| State | Meaning |
|---|---|
| `StateCommitted` | Payload is complete and readable |
| `StateEmpty` | Slot reserved but never written (e.g. a `Writer` block tail) |
| `StateNull` | Field explicitly absent for this record |
| `StateOverflow` | The column ran out of space; the payload was dropped |

A zero-length field is `StateCommitted` and distinct from `StateNull` — this is
what makes the fabric schema-free at the record level. Passing fewer fields than
the schema width marks the remainder `StateNull` rather than failing.

## Testing

```bash
go test ./sliver/                    # correctness
go test ./sliver/ -race              # concurrency
go test ./sliver/ -bench . -benchtime=300ms
```

The suite covers region and payload alignment, column-major descriptor
contiguity, round-trip fidelity across the copy-path crossover, null vs
zero-length, record and column exhaustion, `Reset` reuse, `Close` idempotence and
handle poisoning, eight-way concurrent pack-while-scanning with unique-ID and
torn-read checks, `Writer` block semantics, and the zero-allocation and
zero-heap-growth claims above.

## Benchmarks

Apple M4, darwin/arm64, Go 1.27. Six-field record, 108 payload bytes.

| Benchmark | Result | Allocs |
|---|---|---|
| `PackRecord` | 31.6 ns/op · 3.42 GB/s | 0 |
| `PackRecordParallel` (8 cores) | 279.5 ns/op | 0 |
| `WriterPackRecord` | 22.8 ns/op · 4.74 GB/s | 0 |
| `WriterPackRecordParallel` (8 cores) | 27.6 ns/op | 0 |
| `GetField` | 1.77 ns/op | 0 |
| `ScanColumn` (1,048,576 rows) | 1.15 ms · ~910M rows/s | 0 |
| `FilterEqual` (1,048,576 rows) | 1.71 ms · ~614M rows/s · 9.82 GB/s | 0 |
| `BaselineJSONRoundTrip` | 927.5 ns/op · 0.12 GB/s | 8 (472 B) |

Against `encoding/json` marshal + unmarshal of the same record: **~41× faster,
8 allocations to 0.**

`FilterEqual` rejects a non-matching row on the length check alone, so a
selective filter never touches the payload region for rows it discards.

## Sharp edges

- **`GetField` slices alias the arena.** They are valid until `Reset` or
  `Close`, and writing through one corrupts the arena. Use `CopyField` if the
  bytes must outlive the arena.
- **`Reset` requires quiescence.** It is not safe against a concurrent
  `PackRecord`, `GetField`, or scan, and it invalidates every outstanding slice.
  This is documented but not enforced.
- **`Close` poisons the handle.** Subsequent calls return `ErrClosed` or a miss
  rather than faulting, but any slice you still hold points into unmapped
  address space.
- **Geometry is fixed.** The arena never grows. Growing would require moving
  payload bytes, which would invalidate every zero-copy slice already handed out.
- **A `Writer` is not thread-safe.** One per goroutine. Call `Flush` before
  reading `Stats` and before dropping it.
- **Column overflow is partial, not atomic.** If one column fills mid-record,
  that field is marked `StateOverflow` and `ErrColumnFull` is returned, but the
  record ID stays valid and every field that fit remains readable. A partially
  packed record is honest about what is missing rather than silently truncating.
- **`go vet` reports one warning**, at `mmap_darwin.go:42` / `mmap_linux.go:47`:
  `possible misuse of unsafe.Pointer`. This is the `uintptr`→`Pointer`
  conversion of the `mmap` result. It is unavoidable and `golang.org/x/sys/unix`
  carries the identical warning. It is isolated to one annotated line per
  platform rather than suppressed.
