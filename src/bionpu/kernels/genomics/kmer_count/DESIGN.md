# bionpu_kmer_count — k-mer counting on AIE2P (DESIGN)

Per `state/kmer_count_interface_contract.md` (T1) — symbols, ObjectFifo
names, constants, streaming chunk + overlap protocol, and the
emit-on-evict overflow policy are pinned there. This document is a
T4-stage skeleton (section headers only); T17 fills the body once T5–T11
have produced measurements and chr22 byte-equal-vs-Jellyfish data is in.

## Topology

## §1 — Streaming chunk + overlap protocol

## §2 — 2-bit wire format (MSB-first within byte)

## §3 — Rolling canonical k-mer (forward + reverse-complement)

## §4 — Per-tile open-addressed hash table (geometry, sizing, headroom)

## §5 — Hash collision overflow policy: emit-on-evict

## §6 — Aggregator fan-in and dedup-merge

## §7 — Host runner re-aggregation (canonical-u64 sum)

## §8 — Reference-equivalence vs Jellyfish (chr22 byte-equal)

## §9 — Performance: shim-DMA bandwidth ceiling, tile-table scaling

## §10 — Gaps surfaced (cross-reference `gaps.yaml`)
