package sliver

import (
	"encoding/json"
	"strconv"
	"testing"
)

const benchFields = 6

func benchArena(b *testing.B, records uint64) *Arena {
	b.Helper()
	a, err := Open(Config{
		FieldCount:            benchFields,
		MaxRecords:            records,
		DefaultColumnCapacity: records * 48,
		Prefault:              true,
	})
	if err != nil {
		b.Fatalf("Open: %v", err)
	}
	b.Cleanup(func() { _ = a.Close() })
	return a
}

func benchRow() [][]byte {
	row := make([][]byte, benchFields)
	for f := range row {
		buf := make([]byte, 8+f*4)
		for k := range buf {
			buf[k] = byte(f*7 + k)
		}
		row[f] = buf
	}
	return row
}

func BenchmarkPackRecord(b *testing.B) {
	a := benchArena(b, uint64(b.N)+16)
	row := benchRow()
	var bytes int64
	for _, f := range row {
		bytes += int64(len(f))
	}

	b.SetBytes(bytes)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := a.PackRecord(row); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPackRecordParallel(b *testing.B) {
	a := benchArena(b, uint64(b.N)+4096)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		row := benchRow() // per-goroutine, outside the timed loop
		for pb.Next() {
			if _, err := a.PackRecord(row); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkGetField(b *testing.B) {
	const n = 1 << 16
	a := benchArena(b, n)
	row := benchRow()
	for i := 0; i < n; i++ {
		if _, err := a.PackRecord(row); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		v, ok := a.GetField(uint64(i)&(n-1), 3)
		if !ok {
			b.Fatal("miss")
		}
		sink += len(v)
	}
	if sink == 0 {
		b.Fatal("optimized away")
	}
}

func BenchmarkFilterEqual(b *testing.B) {
	const n = 1 << 20
	a := benchArena(b, n)
	row := benchRow()
	for i := 0; i < n; i++ {
		row[0] = []byte(strconv.Itoa(i % 1000))
		if _, err := a.PackRecord(row); err != nil {
			b.Fatal(err)
		}
	}
	needle := []byte("777")
	out := make([]uint64, 4096)

	// Records with i%1000 == 777 in [0, n): 777, 1777, ... under n.
	want := (n - 777 + 999) / 1000

	b.SetBytes(int64(n) * descriptorSize) // descriptor bytes swept per op
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := a.FilterEqual(0, needle, out); got != want {
			b.Fatalf("matched %d, want %d", got, want)
		}
	}
}

func BenchmarkScanColumn(b *testing.B) {
	const n = 1 << 20
	a := benchArena(b, n)
	row := benchRow()
	for i := 0; i < n; i++ {
		if _, err := a.PackRecord(row); err != nil {
			b.Fatal(err)
		}
	}
	var total int
	visit := func(rec uint64, v []byte) bool {
		total += len(v)
		return true
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.ScanColumn(2, visit)
	}
	if total == 0 {
		b.Fatal("optimized away")
	}
}

// --- the baseline the fabric exists to beat ---------------------------------

type jsonRow struct {
	F0, F1, F2, F3, F4, F5 []byte
}

// BenchmarkBaselineJSONRoundTrip is the cost of the same record through
// encoding/json: reflection-driven, heap-allocating, GC-visible.
func BenchmarkBaselineJSONRoundTrip(b *testing.B) {
	row := benchRow()
	r := jsonRow{row[0], row[1], row[2], row[3], row[4], row[5]}
	var bytes int64
	for _, f := range row {
		bytes += int64(len(f))
	}

	b.SetBytes(bytes)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blob, err := json.Marshal(&r)
		if err != nil {
			b.Fatal(err)
		}
		var out jsonRow
		if err := json.Unmarshal(blob, &out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWriterPackRecordParallel(b *testing.B) {
	a := benchArena(b, uint64(b.N)+1<<16)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		w, err := a.NewWriter(WriterConfig{})
		if err != nil {
			b.Fatal(err)
		}
		row := benchRow()
		for pb.Next() {
			if _, err := w.PackRecord(row); err != nil {
				b.Fatal(err)
			}
		}
		w.Flush()
	})
}

func BenchmarkWriterPackRecord(b *testing.B) {
	a := benchArena(b, uint64(b.N)+1<<16)
	w, err := a.NewWriter(WriterConfig{})
	if err != nil {
		b.Fatal(err)
	}
	row := benchRow()
	var bytes int64
	for _, f := range row {
		bytes += int64(len(f))
	}
	b.SetBytes(bytes)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := w.PackRecord(row); err != nil {
			b.Fatal(err)
		}
	}
	w.Flush()
}
