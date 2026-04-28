# CRISPR PAM filter pktMerge direct-stream artifacts

These artifacts are the opt-in packetized replacement build for
`crispr_pam_filter_pktmerge_direct_stream`.

Build command:

```bash
cd bionpu/kernels/crispr/pam_filter_pktmerge
make NPU2=1 DIRECT_STREAM=1 build/final.xclbin build/insts.bin
XILINX_XRT=/opt/xilinx/xrt make NPU2=1 DIRECT_STREAM=1 crispr_pam_filter_pktmerge
```

The Python registration points this variant at:

```text
bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge_direct_stream/
```

The default `crispr_pam_filter_pktmerge` artifact remains separate as the
byte-equal ObjectFifo rollback path.
