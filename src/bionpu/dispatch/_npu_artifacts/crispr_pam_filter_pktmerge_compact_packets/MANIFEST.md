# crispr_pam_filter_pktmerge_compact_packets

Built from `bionpu/kernels/crispr/pam_filter_pktmerge` with:

```sh
make NPU2=1 COMPACT_PACKETS=1 build/final.xclbin build/insts.bin crispr_pam_filter_pktmerge
```

This artifact keeps Tile A's valid-window payload compacted in a counted
ObjectFifo instead of using the direct AIE stream fanout. The current compact
ABI uses `memref<65xi64>`: one count word plus two logical 32-bit words per
valid window. The local window index is packed into the high byte of the second
word and carried into compact partial records.
