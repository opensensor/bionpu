# methylation_context v0

Sparse scanner for cytosine methylation contexts:

- `CG`: C followed by G
- `CHG`: C followed by A/C/T then G
- `CHH`: C followed by A/C/T then A/C/T

Both strands are emitted. Minus-strand cytosines are represented by the
forward-reference `G` at the same 0-based position.

## ABI

Input is the standard genomics packed-2bit stream with the 8-byte
per-chunk header used by the existing sparse scanners.

Output is a 32 KiB partial buffer:

- bytes `[0:4]`: little-endian `uint32 n_records`
- records: `uint32 pos | uint8 strand | uint8 context | uint16 pad`
- `strand`: `0=+`, `1=-`
- `context`: `0=CG`, `1=CHG`, `2=CHH`

The Python host op converts these records back to
`MethylationContextHit` and reconstructs the oriented motif from the
packed input.
