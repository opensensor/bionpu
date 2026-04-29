# bionpu_pam_filter_iupac_n4 — silicon artifacts (Track A v0)

Multi-PAM IUPAC scan kernel; n_tiles=4. A SINGLE xclbin serves every
Cas9 PAM variant via runtime header args (pam_mask + pam_length).

## Build status

Scaffold directory. The xclbin / insts.bin / host_runner are produced
by:

```
cd src/bionpu/kernels/crispr/pam_filter_iupac
make NPU2=1 experiment=wide4 all
```

then vendored under this directory. The build is bounded scope (mirrors
primer_scan v0 build wall ~30s) and produces:

* `final.xclbin` (xclbin for n_tiles=4)
* `insts.bin` (NPU instruction binary)
* `host_runner` (XRT-linked host runner)

## Pinned shape

| field              | value                                            |
|--------------------|--------------------------------------------------|
| PAM length max     | 8 (PFI_PAM_LEN_MAX; covers SaCas9-KKH NNNRRT)    |
| n_tiles            | 4 (broadcast topology)                           |
| MAX_EMIT_IDX       | 2046 / 32 KiB pass-slot                          |
| header             | 24 bytes (4 mask + 1 length + 11 pad + 4 actual + 4 owned) |

See `kernels/crispr/pam_filter_iupac/DESIGN.md` and `MANIFEST.md` for
build invocations and shape pins.

## SHA-256 sums

(populated post-build by the vendor step)
