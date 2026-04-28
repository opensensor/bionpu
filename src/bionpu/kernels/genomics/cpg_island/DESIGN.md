# bionpu_cpg_island — Design v0

`cpg_island` is the fifth CRISPR-shape genomics primitive. It keeps the
same silicon division of labor as `kmer_count`, `minimizer`, and
`primer_scan`: AIE2P performs per-base streaming arithmetic and sparse
emits; the host performs aggregation into the final biological object.

## Algorithm

For every length-200 window, the tile maintains rolling counters:

- `n_C`
- `n_G`
- `n_CG` dinucleotides fully inside the window

It emits the window-start position when both fixed-point thresholds pass:

```text
2 * (n_C + n_G) >= W
5 * W * n_CG >= 3 * n_C * n_G
```

The tile emits candidate starts, not final islands. The host sorts and
deduplicates candidate starts, then applies the oracle merge rule:
contiguous candidate runs with length at least `W` become half-open
intervals `(run_start, run_end + W)`.

## Wire Format

Input is MSB-first packed 2-bit DNA with an 8-byte per-chunk header:

```text
bytes [0..3]  uint32 actual_payload_bytes
bytes [4..7]  int32 owned_start_offset_bases
bytes [8..]   packed DNA payload
```

Output slot layout:

```text
bytes [0..3]                 uint32 emit_count
bytes [4 .. 4 + 4*n)         n x uint32 candidate positions
```

The runner translates chunk-local positions to global positions and
writes a binary blob:

```text
uint64 n_records
n x uint32 candidate_position
```

## Scope

v0 is source-complete and oracle-locked. Silicon validation should build
the `wide4` artifact and compare `BionpuCpgIsland(...)(packed_seq)` to
`find_cpg_islands_packed(...)` on synthetic and chr22 fixtures.

Known v1 candidate: cap-fire in a fully CpG-rich 16 Kbp chunk can exceed
`CI_MAX_EMIT_IDX=8190`. If observed on real references, use the same
lever as minimizer: smaller chunks or wider partial output.
