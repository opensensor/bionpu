# pam_filter_iupac — Design notes (Track A v0)

## Goal

Multi-PAM IUPAC scan kernel for base editor design. A single xclbin
serves every Cas9 PAM variant in the supported zoo (SpCas9 wt, NG,
SpRY, SaCas9-KKH, ...) via runtime header args.

This is a NEW silicon kernel (NOT a modification of the locked
`crispr/pam_filter` kernel). The locked kernel is reference-frozen at
v0 with a hardcoded NGG check; `pam_filter_iupac` is the BE-design
extension and ships independently.

## Topology

Mirrors `genomics/primer_scan` v0:

```
shim ─seq_in─ broadcast ──▶ tile_0 .. tile_{N_TILES-1}
                                  │
                            (per-position IUPAC mask check;
                             emit on match. Forward strand only;
                             host does RC.)
                                  │
                            partial_pam_<i>
                                  │
                      memtile .join(N_TILES, ...)
                                  │
                                  ▼
                              shim drain
```

Every tile receives the same input (broadcast), so every tile produces
the same match output. Tile 0 is authoritative; siblings 1..N-1 are
duplicates that the host runner ignores.

## Wire format

24-byte chunk header (mirrors primer_scan with PAM mask in place of
primer canonical pair):

| bytes | field | type |
|------|------|------|
| `[0..3]`  | `pam_mask` (packed 4-bit IUPAC nibbles) | uint32 LE |
| `[4]`     | `pam_length` | uint8 |
| `[5..7]`  | `_pad0` | 3 bytes |
| `[8..15]` | `_reserved` | uint64 LE |
| `[16..19]`| `actual_payload_bytes` | uint32 LE |
| `[20..23]`| `owned_start_offset_bases` | int32 LE |
| `[24..]` | payload | packed 2-bit DNA |

Output records: 16 bytes per record, identical layout to primer_scan
(`uint32 query_pos | uint8 strand | uint8 _pad | uint16 _pad2 |
uint64 _pad3`).

## IUPAC nibble encoding

| code | bases | nibble |
|------|-------|--------|
| A | A | 0x1 |
| C | C | 0x2 |
| G | G | 0x4 |
| T | T | 0x8 |
| R | A,G | 0x5 |
| Y | C,T | 0xA |
| S | G,C | 0x6 |
| W | A,T | 0x9 |
| K | G,T | 0xC |
| M | A,C | 0x3 |
| B | C,G,T | 0xE |
| D | A,G,T | 0xD |
| H | A,C,T | 0xB |
| V | A,C,G | 0x7 |
| N | A,C,G,T | 0xF |

Per-position match check (kernel inner loop):

```cpp
uint8_t base_2bit  = (win >> (2 * (pam_length - 1 - p))) & 0x3;
uint8_t base_onehot = (uint8_t)(1u << base_2bit);  // A=1 C=2 G=4 T=8
uint8_t pos_mask   = (uint8_t)((pam_mask >> (4 * p)) & 0xF);
if ((base_onehot & pos_mask) == 0) { match = false; break; }
```

Position 0 of the PAM mask corresponds to the FIRST PAM base
(5'-most), which lives at the highest 2-bit lane of the rolling
window after a left-shift fill. Cross-check the oracle in
`bionpu/data/pam_iupac_oracle.py` — both byte-equal by construction.

## Strand handling

Mirrors locked `crispr/pam_filter`: forward only on tile, host runs RC
as a second dispatch on a pre-flipped buffer. Avoids per-base 2-bit
complement on tile (saves DM budget).

## Build

```
make NPU2=1 experiment=wide4 all
```

Builds a single per-(n_tiles) xclbin. PAM is provided per-dispatch via
the chunk header — no per-PAM rebuild required.

## v1 deferred

* In-process pyxrt path (currently only subprocess; mirrors primer_scan v0 where pyxrt was added in v2).
* Reverse-complement on tile (host pre-flip is the cheap path).
* Off-target match scoring composition (the BE design ranker stays
  CPU-only in v0; the locked `crispr/match_multitile_memtile`
  kernel handles off-target match arithmetic in v1+).

## Phase relationship

Per `PRDs/PRD-crispr-state-of-the-art-roadmap.md` §3.1:

* Phase 1 (v0, this kernel + ranker): SpCas9 wt + NG; BE4max + ABE7.10.
* Phase 2 (v1+): SpRY, SaCas9-KKH; ABE8e; bystander-rate scoring.
* Phase 3 (v2+): BE-Hive-class neural scorer (CPU per Track D's lessons).
