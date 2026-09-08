package sliver

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

func openTest(t testing.TB, fields uint32, records uint64, colCap uint64) *Arena {
	t.Helper()
	a, err := Open(Config{
		FieldCount:            fields,
		MaxRecords:            records,
		DefaultColumnCapacity: colCap,
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	return a
}

// --- geometry and the 8-byte alignment matrix -------------------------------

func TestRegionGeometryIsCacheLineAligned(t *testing.T) {
	a := openTest(t, 7, 4096, 1<<20)

	if uintptr(a.base)%CacheLineSize != 0 {
		t.Fatalf("arena base %#x is not cache-line aligned", uintptr(a.base))
	}
	for name, p := range map[string]unsafe.Pointer{
		"header":      a.base,
		"columnState": a.colBase,
		"descriptors": a.descBase,
		"data":        a.dataBase,
	} {
		if uintptr(p)%CacheLineSize != 0 {
			t.Errorf("%s region at %#x is not 64-byte aligned", name, uintptr(p))
		}
	}

	// Every column state block must sit on its own cache line.
	for f := uint32(0); f < a.FieldCount(); f++ {
		cs := a.column(f)
		if uintptr(unsafe.Pointer(cs))%CacheLineSize != 0 {
			t.Errorf("column %d state block is not cache-line aligned", f)
		}
		if cs.baseOff%CacheLineSize != 0 {
			t.Errorf("column %d payload base %d is not cache-line aligned", f, cs.baseOff)
		}
	}

	// Every global counter must sit on its own cache line, and no two may share.
	counters := []uintptr{
		uintptr(unsafe.Pointer(&a.hdr.recordCursor.value)),
		uintptr(unsafe.Pointer(&a.hdr.commitCounter.value)),
		uintptr(unsafe.Pointer(&a.hdr.packFailures.value)),
		uintptr(unsafe.Pointer(&a.hdr.readMisses.value)),
		uintptr(unsafe.Pointer(&a.hdr.scanCalls.value)),
	}
	for i, c := range counters {
		if c%CacheLineSize != 0 {
			t.Errorf("counter %d at %#x is not cache-line aligned", i, c)
		}
		for j := i + 1; j < len(counters); j++ {
			if c/CacheLineSize == counters[j]/CacheLineSize {
				t.Errorf("counters %d and %d share cache line %#x", i, j, c/CacheLineSize)
			}
		}
	}
}

func TestEveryPayloadExtentIsEightByteAligned(t *testing.T) {
	a := openTest(t, 4, 2048, 1<<20)

	// Deliberately ragged lengths: 1..37 bytes, which would misalign every
	// subsequent extent if the reservation stride were not rounded up to 8.
	fields := make([][]byte, 4)
	for i := 0; i < 500; i++ {
		for f := range fields {
			n := 1 + (i*7+f*3)%37
			buf := make([]byte, n)
			for k := range buf {
				buf[k] = byte(i + f + k)
			}
			fields[f] = buf
		}
		if _, err := a.PackRecord(fields); err != nil {
			t.Fatalf("pack %d: %v", i, err)
		}
	}

	for rec := uint64(0); rec < a.RecordCount(); rec++ {
		for f := uint32(0); f < a.FieldCount(); f++ {
			v, ok := a.GetField(rec, f)
			if !ok {
				t.Fatalf("record %d field %d missing", rec, f)
			}
			addr := uintptr(unsafe.Pointer(unsafe.SliceData(v)))
			if addr%WordAlign != 0 {
				t.Fatalf("record %d field %d payload at %#x is not 8-byte aligned", rec, f, addr)
			}
			// An 8-byte-aligned extent can never straddle a 64-byte line
			// within its first word, which is the split-line case we exclude.
			if addr%CacheLineSize+8 > CacheLineSize && addr%8 != 0 {
				t.Fatalf("record %d field %d first word splits a cache line", rec, f)
			}
		}
	}
}

func TestDescriptorRunsAreColumnMajorAndContiguous(t *testing.T) {
	a := openTest(t, 3, 64, 4096)
	for f := uint32(0); f < 3; f++ {
		want := a.columnRunBase(f)
		for r := uint64(0); r < 64; r++ {
			got := unsafe.Pointer(a.descriptor(f, r))
			expect := unsafe.Add(want, uintptr(r)*descriptorSize)
			if got != expect {
				t.Fatalf("descriptor(%d,%d) = %p, want %p", f, r, got, expect)
			}
		}
		// The run must be exactly maxRecords*16 bytes and butt against the next.
		if f+1 < 3 {
			next := a.columnRunBase(f + 1)
			gap := uintptr(next) - uintptr(want)
			if gap != uintptr(64*descriptorSize) {
				t.Fatalf("column %d run is %d bytes, want %d", f, gap, 64*descriptorSize)
			}
		}
	}
}

// --- correctness ------------------------------------------------------------

func TestPackGetRoundTrip(t *testing.T) {
	a := openTest(t, 5, 1024, 1<<20)

	type row [5][]byte
	want := make([]row, 0, 300)
	fields := make([][]byte, 5)

	for i := 0; i < 300; i++ {
		var r row
		for f := 0; f < 5; f++ {
			// Lengths spanning the word-loop path, the crossover, and memmove.
			n := (i*13 + f*29) % 200
			buf := make([]byte, n)
			for k := range buf {
				buf[k] = byte(i*3 + f*7 + k)
			}
			r[f] = buf
			fields[f] = buf
		}
		id, err := a.PackRecord(fields)
		if err != nil {
			t.Fatalf("pack %d: %v", i, err)
		}
		if id != uint64(i) {
			t.Fatalf("record id = %d, want %d", id, i)
		}
		want = append(want, r)
	}

	for i, r := range want {
		for f := 0; f < 5; f++ {
			got, ok := a.GetField(uint64(i), uint32(f))
			if !ok {
				t.Fatalf("record %d field %d missing", i, f)
			}
			if len(got) != len(r[f]) {
				t.Fatalf("record %d field %d len = %d, want %d", i, f, len(got), len(r[f]))
			}
			for k := range got {
				if got[k] != r[f][k] {
					t.Fatalf("record %d field %d byte %d = %d, want %d", i, f, k, got[k], r[f][k])
				}
			}
		}
	}

	if a.CommittedCount() != 300 {
		t.Errorf("CommittedCount = %d, want 300", a.CommittedCount())
	}
}

func TestZeroLengthIsDistinctFromNull(t *testing.T) {
	a := openTest(t, 3, 16, 4096)

	id, err := a.PackRecord([][]byte{[]byte{}, nil, []byte("x")})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}

	v, ok := a.GetField(id, 0)
	if !ok || len(v) != 0 {
		t.Errorf("field 0: got (%v, %v), want empty committed", v, ok)
	}
	if st, _ := a.FieldState(id, 0); st != StateCommitted {
		t.Errorf("field 0 state = %d, want StateCommitted", st)
	}

	if _, ok := a.GetField(id, 1); ok {
		t.Error("field 1 should not be readable")
	}
	if st, _ := a.FieldState(id, 1); st != StateNull {
		t.Errorf("field 1 state = %d, want StateNull", st)
	}
}

func TestShortRecordFillsTrailingNulls(t *testing.T) {
	a := openTest(t, 4, 16, 4096)
	id, err := a.PackRecord([][]byte{[]byte("a"), []byte("b")})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	for f := uint32(2); f < 4; f++ {
		st, ok := a.FieldState(id, f)
		if !ok || st != StateNull {
			t.Errorf("field %d state = %d ok=%v, want StateNull", f, st, ok)
		}
	}
}

func TestUnreservedAndOutOfRangeReadsMiss(t *testing.T) {
	a := openTest(t, 2, 16, 4096)
	if _, err := a.PackRecord([][]byte{[]byte("a"), []byte("b")}); err != nil {
		t.Fatal(err)
	}
	if _, ok := a.GetField(1, 0); ok {
		t.Error("unreserved record 1 should miss")
	}
	if _, ok := a.GetField(0, 2); ok {
		t.Error("out-of-range field 2 should miss")
	}
	if _, ok := a.GetField(1<<40, 0); ok {
		t.Error("out-of-range record should miss")
	}
	if a.Stats().ReadMisses != 3 {
		t.Errorf("ReadMisses = %d, want 3", a.Stats().ReadMisses)
	}
}

// --- limits -----------------------------------------------------------------

func TestRecordLimitRefusesCleanly(t *testing.T) {
	a := openTest(t, 1, 4, 4096)
	f := [][]byte{[]byte("v")}
	for i := 0; i < 4; i++ {
		if _, err := a.PackRecord(f); err != nil {
			t.Fatalf("pack %d: %v", i, err)
		}
	}
	for i := 0; i < 3; i++ {
		if _, err := a.PackRecord(f); err != ErrRecordLimit {
			t.Fatalf("overflow pack %d: err = %v, want ErrRecordLimit", i, err)
		}
	}
	if got := a.RecordCount(); got != 4 {
		t.Errorf("RecordCount = %d, want 4", got)
	}
	// The cursor must be clamped, not running away.
	if got := a.Stats().RecordCursor; got != 4 {
		t.Errorf("RecordCursor = %d, want 4 (clamped)", got)
	}
	for i := uint64(0); i < 4; i++ {
		if _, ok := a.GetField(i, 0); !ok {
			t.Errorf("record %d should still be readable", i)
		}
	}
}

func TestColumnOverflowIsPartialNotCorrupting(t *testing.T) {
	// Column 0 gets one cache line of payload; column 1 gets plenty.
	a, err := Open(Config{
		FieldCount:     2,
		MaxRecords:     64,
		ColumnCapacity: []uint64{64, 1 << 16},
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer a.Close()

	big := make([]byte, 32)
	small := []byte("ok")

	var overflowAt uint64
	found := false
	for i := uint64(0); i < 10; i++ {
		id, err := a.PackRecord([][]byte{big, small})
		if err == ErrColumnFull && !found {
			overflowAt, found = id, true
		}
	}
	if !found {
		t.Fatal("expected column 0 to overflow")
	}

	// The overflowed field is unreadable and flagged; the healthy field on the
	// very same record is intact.
	if _, ok := a.GetField(overflowAt, 0); ok {
		t.Error("overflowed field should not be readable")
	}
	if st, _ := a.FieldState(overflowAt, 0); st != StateOverflow {
		t.Errorf("field 0 state = %d, want StateOverflow", st)
	}
	v, ok := a.GetField(overflowAt, 1)
	if !ok || string(v) != "ok" {
		t.Errorf("field 1 = (%q, %v), want (\"ok\", true)", v, ok)
	}

	cs, _ := a.ColumnStats(0)
	if cs.Rejected == 0 {
		t.Error("column 0 should report rejections")
	}
	if cs.Used > cs.Capacity+uint64(len(big)) {
		t.Errorf("column cursor drifted to %d past capacity %d", cs.Used, cs.Capacity)
	}
}

func TestFieldCountTooWideIsRefused(t *testing.T) {
	a := openTest(t, 2, 8, 4096)
	_, err := a.PackRecord([][]byte{[]byte("a"), []byte("b"), []byte("c")})
	if err != ErrFieldCount {
		t.Fatalf("err = %v, want ErrFieldCount", err)
	}
	if a.RecordCount() != 0 {
		t.Error("a refused pack must not consume a record slot")
	}
}

// --- scanning ---------------------------------------------------------------

func TestScanFilterAndCursor(t *testing.T) {
	a := openTest(t, 2, 512, 1<<18)

	fields := make([][]byte, 2)
	for i := 0; i < 200; i++ {
		fields[0] = []byte(fmt.Sprintf("k%03d", i%10))
		fields[1] = []byte(fmt.Sprintf("payload-%d", i))
		if _, err := a.PackRecord(fields); err != nil {
			t.Fatal(err)
		}
	}

	var seen int
	got := a.ScanColumn(0, func(rec uint64, v []byte) bool {
		seen++
		return true
	})
	if got != 200 || seen != 200 {
		t.Errorf("ScanColumn = %d, visited %d, want 200/200", got, seen)
	}

	out := make([]uint64, 64)
	n := a.FilterEqual(0, []byte("k007"), out)
	if n != 20 {
		t.Errorf("FilterEqual matched %d, want 20", n)
	}
	for _, rec := range out[:n] {
		if rec%10 != 7 {
			t.Errorf("record %d does not have key k007", rec)
		}
	}
	if c := a.CountEqual(0, []byte("k007")); c != 20 {
		t.Errorf("CountEqual = %d, want 20", c)
	}
	if c := a.CountEqual(0, []byte("nope")); c != 0 {
		t.Errorf("CountEqual(miss) = %d, want 0", c)
	}

	// Early termination.
	stopped := a.ScanColumn(0, func(rec uint64, v []byte) bool { return rec < 4 })
	if stopped != 5 {
		t.Errorf("early-stop scan visited %d, want 5", stopped)
	}

	// Range filter over the key column.
	rng := make([]uint64, 512)
	rn := a.FilterRange(0, []byte("k002"), []byte("k004"), rng)
	if rn != 60 {
		t.Errorf("FilterRange matched %d, want 60", rn)
	}

	// Cursor covers exactly the same ground as ScanColumn.
	cur := a.NewCursor(1)
	count := 0
	for {
		rec, v, ok := cur.Next()
		if !ok {
			break
		}
		if want := fmt.Sprintf("payload-%d", rec); string(v) != want {
			t.Fatalf("cursor record %d = %q, want %q", rec, v, want)
		}
		count++
	}
	if count != 200 {
		t.Errorf("cursor visited %d, want 200", count)
	}
}

func TestColumnBytesIsDenseAndPacked(t *testing.T) {
	a := openTest(t, 1, 32, 4096)
	for i := 0; i < 4; i++ {
		if _, err := a.PackRecord([][]byte{[]byte("abcdefgh")}); err != nil {
			t.Fatal(err)
		}
	}
	raw, ok := a.ColumnBytes(0)
	if !ok {
		t.Fatal("ColumnBytes failed")
	}
	// Four 8-byte extents, each already a multiple of 8: exactly 32 dense bytes.
	if len(raw) != 32 {
		t.Fatalf("column image is %d bytes, want 32", len(raw))
	}
	if string(raw) != "abcdefghabcdefghabcdefghabcdefgh" {
		t.Fatalf("column image = %q", raw)
	}

	desc, ok := a.DescriptorBytes(0)
	if !ok || len(desc) != 32*descriptorSize {
		t.Fatalf("descriptor image is %d bytes, want %d", len(desc), 32*descriptorSize)
	}
}

// --- lifecycle --------------------------------------------------------------

func TestResetRewindsWithoutRemapping(t *testing.T) {
	a := openTest(t, 2, 64, 4096)
	before := a.base

	for i := 0; i < 10; i++ {
		if _, err := a.PackRecord([][]byte{[]byte("aa"), []byte("bb")}); err != nil {
			t.Fatal(err)
		}
	}
	a.Reset()

	if a.base != before {
		t.Error("Reset must not remap")
	}
	if a.RecordCount() != 0 || a.CommittedCount() != 0 {
		t.Errorf("after Reset: records=%d committed=%d, want 0/0", a.RecordCount(), a.CommittedCount())
	}
	if cs, _ := a.ColumnStats(0); cs.Used != 0 {
		t.Errorf("column cursor = %d after Reset, want 0", cs.Used)
	}
	if _, ok := a.GetField(0, 0); ok {
		t.Error("stale record readable after Reset")
	}

	// The arena is reusable and hands out IDs from zero again.
	id, err := a.PackRecord([][]byte{[]byte("cc"), []byte("dd")})
	if err != nil || id != 0 {
		t.Fatalf("post-reset pack: id=%d err=%v", id, err)
	}
	if v, _ := a.GetField(0, 0); string(v) != "cc" {
		t.Errorf("post-reset read = %q, want \"cc\"", v)
	}
}

func TestCloseIsIdempotentAndPoisonsTheHandle(t *testing.T) {
	a, err := Open(Config{FieldCount: 1, MaxRecords: 8, DefaultColumnCapacity: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.PackRecord([][]byte{[]byte("x")}); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := a.Close(); err != ErrClosed {
		t.Fatalf("second Close = %v, want ErrClosed", err)
	}
	if _, ok := a.GetField(0, 0); ok {
		t.Error("GetField after Close must miss, not fault")
	}
	if _, err := a.PackRecord([][]byte{[]byte("x")}); err != ErrClosed {
		t.Errorf("PackRecord after Close = %v, want ErrClosed", err)
	}
	if n := a.ScanColumn(0, func(uint64, []byte) bool { return true }); n != 0 {
		t.Errorf("ScanColumn after Close = %d, want 0", n)
	}
}

func TestGeometryValidation(t *testing.T) {
	cases := []Config{
		{FieldCount: 0, MaxRecords: 4, DefaultColumnCapacity: 64},
		{FieldCount: 1, MaxRecords: 0, DefaultColumnCapacity: 64},
		{FieldCount: MaxFields + 1, MaxRecords: 4, DefaultColumnCapacity: 64},
		{FieldCount: 1, MaxRecords: 4},
		{FieldCount: 2, MaxRecords: 4, ColumnCapacity: []uint64{64}},
		{FieldCount: 2, MaxRecords: 1 << 62, DefaultColumnCapacity: 64},
	}
	for i, cfg := range cases {
		if a, err := Open(cfg); err == nil {
			a.Close()
			t.Errorf("case %d: Open succeeded, want failure", i)
		}
	}
}

func TestMlockIsRequestedAndReported(t *testing.T) {
	a := openTest(t, 1, 8, 1<<16)
	if !a.Pinned() && a.LockError() == nil {
		t.Fatal("arena is unpinned but reports no lock error")
	}
	if a.Pinned() && a.LockError() != nil {
		t.Fatal("arena is pinned but reports a lock error")
	}
	t.Logf("pinned=%v lockError=%v", a.Pinned(), a.LockError())
}

// --- concurrency ------------------------------------------------------------

func TestConcurrentPackAndRead(t *testing.T) {
	const (
		writers = 8
		perW    = 2000
		fields  = 4
	)
	a := openTest(t, fields, writers*perW+16, 1<<22)

	var wg sync.WaitGroup
	ids := make([][]uint64, writers)

	for w := 0; w < writers; w++ {
		ids[w] = make([]uint64, perW)
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			buf := make([][]byte, fields)
			scratch := make([][]byte, fields)
			for f := range scratch {
				scratch[f] = make([]byte, 24)
			}
			for i := 0; i < perW; i++ {
				for f := 0; f < fields; f++ {
					v := scratch[f][:8+(i+f)%16]
					for k := range v {
						v[k] = byte(w*31 + i*7 + f*13 + k)
					}
					buf[f] = v
				}
				id, err := a.PackRecord(buf)
				if err != nil {
					t.Errorf("writer %d pack %d: %v", w, i, err)
					return
				}
				ids[w][i] = id
			}
		}(w)
	}

	// Concurrent readers sweeping while writers append. They must never observe
	// a torn value: any committed descriptor points at complete bytes.
	var stop sync.WaitGroup
	done := make(chan struct{})
	for r := 0; r < 4; r++ {
		stop.Add(1)
		go func() {
			defer stop.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				a.ScanColumn(0, func(rec uint64, v []byte) bool {
					if len(v) < 8 || len(v) > 24 {
						t.Errorf("torn value at record %d: len %d", rec, len(v))
						return false
					}
					return true
				})
			}
		}()
	}

	wg.Wait()
	close(done)
	stop.Wait()

	// Every record ID must be unique across all writers.
	seen := make(map[uint64]bool, writers*perW)
	for w := 0; w < writers; w++ {
		for _, id := range ids[w] {
			if seen[id] {
				t.Fatalf("record id %d handed out twice", id)
			}
			seen[id] = true
		}
	}
	if len(seen) != writers*perW {
		t.Fatalf("got %d unique ids, want %d", len(seen), writers*perW)
	}

	// Every record must read back exactly what its writer wrote.
	for w := 0; w < writers; w++ {
		for i := 0; i < perW; i++ {
			id := ids[w][i]
			for f := 0; f < fields; f++ {
				v, ok := a.GetField(id, uint32(f))
				if !ok {
					t.Fatalf("writer %d record %d field %d missing", w, i, f)
				}
				want := 8 + (i+f)%16
				if len(v) != want {
					t.Fatalf("writer %d record %d field %d len %d, want %d", w, i, f, len(v), want)
				}
				for k := range v {
					if want := byte(w*31 + i*7 + f*13 + k); v[k] != want {
						t.Fatalf("writer %d record %d field %d byte %d = %d, want %d", w, i, f, k, v[k], want)
					}
				}
			}
		}
	}

	if a.CommittedCount() != writers*perW {
		t.Errorf("CommittedCount = %d, want %d", a.CommittedCount(), writers*perW)
	}
}

// --- the allocation and GC claims -------------------------------------------

func TestPackAndGetAllocateNothing(t *testing.T) {
	a := openTest(t, 4, 300000, 1<<24)

	fields := make([][]byte, 4)
	for f := range fields {
		fields[f] = make([]byte, 12+f*20) // spans the word loop and memmove paths
	}

	if got := testing.AllocsPerRun(20000, func() {
		if _, err := a.PackRecord(fields); err != nil {
			t.Fatal(err)
		}
	}); got != 0 {
		t.Errorf("PackRecord allocates %v objects per call, want 0", got)
	}

	if got := testing.AllocsPerRun(20000, func() {
		if _, ok := a.GetField(17, 2); !ok {
			t.Fatal("read miss")
		}
	}); got != 0 {
		t.Errorf("GetField allocates %v objects per call, want 0", got)
	}

	out := make([]uint64, 1024)
	needle := fields[1]
	if got := testing.AllocsPerRun(50, func() {
		a.FilterEqual(1, needle, out)
	}); got != 0 {
		t.Errorf("FilterEqual allocates %v objects per call, want 0", got)
	}

	if got := testing.AllocsPerRun(50, func() {
		cur := a.NewCursor(0)
		for {
			if _, _, ok := cur.Next(); !ok {
				return
			}
		}
	}); got != 0 {
		t.Errorf("Cursor iteration allocates %v objects per call, want 0", got)
	}

	dst := make([]byte, 128)
	if got := testing.AllocsPerRun(20000, func() {
		a.CopyField(dst, 5, 3)
	}); got != 0 {
		t.Errorf("CopyField allocates %v objects per call, want 0", got)
	}
}

func TestPackingDoesNotGrowTheGoHeap(t *testing.T) {
	const n = 400000
	a := openTest(t, 4, n+16, 1<<26)

	fields := make([][]byte, 4)
	for f := range fields {
		fields[f] = make([]byte, 32)
	}

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	for i := 0; i < n; i++ {
		if _, err := a.PackRecord(fields); err != nil {
			t.Fatalf("pack %d: %v", i, err)
		}
	}

	runtime.ReadMemStats(&after)

	// 400k records x 4 fields x 32 bytes is 51 MiB of payload. If any of it had
	// landed on the Go heap, HeapObjects would have moved by millions.
	objDelta := int64(after.HeapObjects) - int64(before.HeapObjects)
	if objDelta > 1024 {
		t.Errorf("packing %d records created %d heap objects, want ~0", n, objDelta)
	}
	if after.NumGC != before.NumGC {
		t.Logf("note: %d GC cycles ran during the pack loop (not caused by the arena)",
			after.NumGC-before.NumGC)
	}
	t.Logf("heap objects delta=%d, arena bytes=%d, records=%d",
		objDelta, a.SizeBytes(), a.RecordCount())
}

// --- writer -----------------------------------------------------------------

func TestWriterRoundTripAndUniqueIDs(t *testing.T) {
	const (
		writers = 8
		perW    = 5000
		fields  = 3
	)
	a := openTest(t, fields, writers*perW+8192, 1<<22)

	var wg sync.WaitGroup
	ids := make([][]uint64, writers)
	for w := 0; w < writers; w++ {
		ids[w] = make([]uint64, perW)
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			wr, err := a.NewWriter(WriterConfig{RecordBatch: 64})
			if err != nil {
				t.Error(err)
				return
			}
			defer wr.Flush()
			buf := make([][]byte, fields)
			scratch := make([][]byte, fields)
			for f := range scratch {
				scratch[f] = make([]byte, 32)
			}
			for i := 0; i < perW; i++ {
				for f := 0; f < fields; f++ {
					v := scratch[f][:4+(i+f)%20]
					for k := range v {
						v[k] = byte(w*17 + i*5 + f*11 + k)
					}
					buf[f] = v
				}
				id, err := wr.PackRecord(buf)
				if err != nil {
					t.Errorf("writer %d pack %d: %v", w, i, err)
					return
				}
				ids[w][i] = id
			}
		}(w)
	}
	wg.Wait()

	seen := make(map[uint64]bool, writers*perW)
	for w := 0; w < writers; w++ {
		for _, id := range ids[w] {
			if seen[id] {
				t.Fatalf("record id %d handed out twice", id)
			}
			seen[id] = true
		}
	}
	if len(seen) != writers*perW {
		t.Fatalf("%d unique ids, want %d", len(seen), writers*perW)
	}

	for w := 0; w < writers; w++ {
		for i := 0; i < perW; i++ {
			id := ids[w][i]
			for f := 0; f < fields; f++ {
				v, ok := a.GetField(id, uint32(f))
				if !ok {
					t.Fatalf("writer %d rec %d field %d missing", w, i, f)
				}
				if want := 4 + (i+f)%20; len(v) != want {
					t.Fatalf("writer %d rec %d field %d len %d want %d", w, i, f, len(v), want)
				}
				for k := range v {
					if want := byte(w*17 + i*5 + f*11 + k); v[k] != want {
						t.Fatalf("writer %d rec %d field %d byte %d = %d want %d", w, i, f, k, v[k], want)
					}
				}
			}
		}
	}

	if got := a.CommittedCount(); got != writers*perW {
		t.Errorf("CommittedCount = %d, want %d", got, writers*perW)
	}
}

func TestWriterUnfilledBlockTailReadsAsAbsent(t *testing.T) {
	a := openTest(t, 1, 1024, 1<<16)
	w, err := a.NewWriter(WriterConfig{RecordBatch: 128})
	if err != nil {
		t.Fatal(err)
	}
	// Pack three records out of a 128-slot block, then stop.
	for i := 0; i < 3; i++ {
		if _, err := w.PackRecord([][]byte{[]byte("v")}); err != nil {
			t.Fatal(err)
		}
	}
	w.Flush()

	if got := a.RecordCount(); got != 128 {
		t.Fatalf("RecordCount = %d, want 128 (the whole reserved block)", got)
	}
	for i := uint64(0); i < 3; i++ {
		if _, ok := a.GetField(i, 0); !ok {
			t.Errorf("record %d should be readable", i)
		}
	}
	// The abandoned tail must read as absent, never as garbage.
	for i := uint64(3); i < 128; i++ {
		if _, ok := a.GetField(i, 0); ok {
			t.Fatalf("unfilled slot %d must not be readable", i)
		}
		if st, _ := a.FieldState(i, 0); st != StateEmpty {
			t.Fatalf("unfilled slot %d state = %d, want StateEmpty", i, st)
		}
	}
	// A scan sees only the real records.
	if n := a.ScanColumn(0, func(uint64, []byte) bool { return true }); n != 3 {
		t.Errorf("ScanColumn = %d, want 3", n)
	}
}

func TestWriterPacksAllocateNothing(t *testing.T) {
	a := openTest(t, 4, 200000, 1<<24)
	w, err := a.NewWriter(WriterConfig{})
	if err != nil {
		t.Fatal(err)
	}
	fields := make([][]byte, 4)
	for f := range fields {
		fields[f] = make([]byte, 16+f*8)
	}
	if got := testing.AllocsPerRun(20000, func() {
		if _, err := w.PackRecord(fields); err != nil {
			t.Fatal(err)
		}
	}); got != 0 {
		t.Errorf("Writer.PackRecord allocates %v per call, want 0", got)
	}
	w.Flush()
}

func TestWriterHonorsColumnCapacity(t *testing.T) {
	a, err := Open(Config{
		FieldCount:     1,
		MaxRecords:     4096,
		ColumnCapacity: []uint64{512},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	// Batch far larger than the column: the oversized speculative grab must be
	// handed back so exact-size reservations can still fill the column.
	w, err := a.NewWriter(WriterConfig{RecordBatch: 8, ColumnBatch: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	val := make([]byte, 8)
	ok := 0
	for i := 0; i < 200; i++ {
		if _, err := w.PackRecord([][]byte{val}); err == nil {
			ok++
		}
	}
	w.Flush()

	if ok != 64 {
		t.Errorf("committed %d records, want 64 (512 bytes / 8)", ok)
	}
	cs, _ := a.ColumnStats(0)
	if cs.Used > cs.Capacity {
		t.Errorf("column used %d exceeds capacity %d", cs.Used, cs.Capacity)
	}
}
