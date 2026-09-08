# Sliver
A schema-free columnar serialization fabric for Go. Sliver packs millions of tabular records into a single contiguous, off-heap, page-locked memory arena and reads them back with pure pointer arithmetic — no reflection, no code generation, no allocation, and nothing for the garbage collector to mark.
