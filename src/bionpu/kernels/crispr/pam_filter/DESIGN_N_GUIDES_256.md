# PAM filter N_GUIDES=256 exploratory topology

## Attempted direct topology

This workspace branch changes the PAM filter shape to:

| field | value |
|---|---:|
| `N_GUIDES` | 256 |
| `N_MATCH_TILES` | 4 |
| `GUIDES_PER_TILE` | 64 |
| guide BO size | 1280 bytes |
| Tile Z sparse slot | 1024 records / 8192 bytes |

The attempted direct dataflow is:

```text
shim -> guides broadcast -> match_0..3
shim -> windows_in -> Tile A -> windows_out broadcast -> match_0..3
Tile A -> pam_meta -> Tile Z
match_0..3 -> partial_0..3 -> Tile Z
Tile Z -> sparse_out -> shim
```

The C++ match tile remains the existing 64-lane vectorized kernel. Tile Z
was extended to accept four partial buffers and emit guide indices 0..255.
The Python wrapper points 256-guide artifacts at
`crispr_pam_filter_early_n256` / `crispr_pam_filter_late_n256` so old
128-guide xclbins are not accidentally used with the wider ABI.

## Build result

The direct topology is not buildable with the current IRON lowering:

```text
aiecc --aie-generate-xclbin ... build/early/aie.mlir
<unknown>:0: error: 'aie.tile' op number of input DMA channel exceeded!
<unknown>:0: note: see current operation: %2 = "aie.tile"() <{col = 1 : i32, row = 3 : i32}> : () -> index
Error: Resource allocation pipeline failed
```

Tile Z has five logical input streams in the direct design:

1. `partial_0`
2. `partial_1`
3. `partial_2`
4. `partial_3`
5. `pam_meta`

That exceeds the compute-tile input DMA channel budget.

## Required buildable topology

A buildable 256-guide variant needs one of these changes:

1. **Memtile fan-in before Tile Z:** keep four 64-guide match tiles, but join
   or packetize the four partial streams in a memtile so Tile Z consumes at
   most two partial streams plus `pam_meta`.
2. **Move PAM metadata into partial packets:** have match tiles carry or
   reproduce the PAM-pass bit so Tile Z does not need a separate `pam_meta`
   ObjectFifo. This requires changing the partial record layout and Tile Z
   parser.
3. **Two-stage Tile Z reduction:** two joiner tiles each consume two partials,
   then a final emit tile consumes two joined streams. This adds one tile and
   another inter-tile stream stage.

Option 1 is the closest to the existing artifact/wrapper pattern, but it is
not a small constant-only edit: it requires a new IRON memtile placement and
join/split layout, updated Tile Z input types, and fresh silicon smoke.

