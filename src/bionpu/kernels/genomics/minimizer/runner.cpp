// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// runner.cpp — Host runner for the (w, k) minimizer kernel.
//
// v1 host-side dataflow:
//   1. Load N_PASSES xclbins (one per pass_idx; xclbin/insts paths are
//      derived from --xclbin / --instr templates by appending _p{idx}).
//   2. Plan input chunks (4096 + per-(k,w) overlap, chunk-aligned).
//   3. v1.2 (b) batched dispatch: each silicon dispatch processes
//      n_chunks_per_launch chunks (BO sized accordingly). Tail batch
//      is padded with zero-actual-bytes chunks.
//   4. v1.3 depth-4 ring of (seq_in, sparse_out) BOs to enable
//      pipelined dispatch. While silicon is computing dispatch i, host
//      accumulates dispatch i-1's output and stages dispatch i+1.
//   5. Parse tile_0's slot per chunk per pass; translate positions to
//      global; append to merged record list.
//   6. After all chunks × all passes, parallel-sort by (position asc,
//      canonical asc), de-duplicate, apply --top, emit binary blob.
//
// Wire format (per minimizer_constants.h):
//   - Input: packed-2-bit DNA, MSB-first.
//   - Per-chunk in-band header (8 bytes):
//       [0..3] uint32 LE actual_payload_bytes
//       [4..7] int32  LE owned_start_offset_bases
//     Payload starts at byte 8.
//   - Per-pass output: N_TILES × 32 KiB pass-slots. Slot 0 holds the
//     authoritative output (every tile sees identical input under the
//     v0 broadcast topology, so slots 1..N_TILES-1 are duplicates and
//     ignored host-side).
//   - Per-slot layout:
//       [0..3]: uint32 LE emit_count
//       [4 .. 4+16*emit_count]: emit_count × { uint64 canonical_LE,
//                                              uint32 position_LE,
//                                              uint32 _pad }
//
// Output formats:
//   - tsv (default): "<position>\t<canonical_acgt>\n" per record.
//   - binary: [uint64 n_records LE][n_records × { uint64 canonical_LE,
//                                                 uint32 position_LE,
//                                                 uint32 _pad }].

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;  // v0/v1 unchanged
constexpr int32_t MZ_OVERLAP_K15_W10 = 8;
constexpr int32_t MZ_OVERLAP_K21_W11 = 8;
constexpr int32_t PARTIAL_OUT_BYTES_PADDED = 32768;
constexpr int32_t MZ_RECORD_BYTES = 16;
constexpr int32_t MZ_MAX_EMIT_IDX = 2046;
constexpr size_t  HEADER_BYTES = 8;

constexpr uint64_t MASK_K15 = (1ULL << 30) - 1ULL;
constexpr uint64_t MASK_K21 = (1ULL << 42) - 1ULL;

int overlap_bytes_for_kw(int k, int w) {
    if (k == 15 && w == 10) return MZ_OVERLAP_K15_W10;
    if (k == 21 && w == 11) return MZ_OVERLAP_K21_W11;
    throw std::runtime_error(
        "unsupported (k=" + std::to_string(k) + ", w=" + std::to_string(w) +
        "); pinned: (15,10), (21,11)");
}

uint64_t kmer_mask_for_k(int k) {
    switch (k) {
        case 15: return MASK_K15;
        case 21: return MASK_K21;
        default:
            throw std::runtime_error(
                "unsupported k=" + std::to_string(k));
    }
}

void trace(const std::string &msg) {
    std::cerr << "[minimizer_runner] " << msg << std::endl;
}

std::vector<uint32_t> read_instr_binary(const std::string &path) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh)
        throw std::runtime_error("failed to open instructions file: " + path);
    std::vector<uint32_t> out;
    uint32_t word = 0;
    while (fh.read(reinterpret_cast<char *>(&word), sizeof(word)))
        out.push_back(word);
    return out;
}

std::vector<uint8_t> read_packed_input(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open packed-2-bit input: " + path);
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz < 0)
        throw std::runtime_error("bad seek on input: " + path);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

char base_of_2bit(uint64_t code) {
    switch (static_cast<unsigned>(code & 0x3ULL)) {
        case 0: return 'A';
        case 1: return 'C';
        case 2: return 'G';
        case 3: return 'T';
    }
    return 'N';
}

std::string decode_canonical_to_acgt(uint64_t canonical, int k) {
    uint64_t masked = canonical & kmer_mask_for_k(k);
    std::string out;
    out.resize(static_cast<size_t>(k));
    for (int pos = 0; pos < k; ++pos) {
        int shift = 2 * (k - 1 - pos);
        uint64_t code = (masked >> shift) & 0x3ULL;
        out[static_cast<size_t>(pos)] = base_of_2bit(code);
    }
    return out;
}

struct MinimizerRecord {
    uint64_t canonical;
    uint32_t position;
    uint32_t _pad;
};

struct Args {
    // Comma-separated list of xclbin paths NOT used; instead, --xclbin
    // points at the pass-0 artifact and the runner derives _p<i> paths
    // for additional passes (mirrors kmer_count).
    std::string xclbin_template;
    std::string instr_template;
    std::string kernel = "MLIR_AIE";
    std::string input_path;
    std::string output_path;
    std::string output_format = "tsv";  // {tsv, binary}
    int k = 15;
    int w = 10;
    int top = 0;            // 0 = all
    int launch_chunks = 4;
    int n_passes = 1;
    int n_chunks_per_launch = 1;
    int iters = 1;
    int warmup = 0;
};

void print_usage(std::ostream &os) {
    os <<
"Usage: minimizer_runner [options]\n"
"\n"
"Required:\n"
"  -x, --xclbin <path>         xclbin file (final.xclbin OR final_p0.xclbin)\n"
"                              When --n-passes>1, the runner derives pass-i\n"
"                              xclbin by replacing the final '_p0' token\n"
"                              with '_p<i>' (mirrors kmer_count convention).\n"
"  -i, --instr <path>          NPU instructions binary (insts*.bin) — same\n"
"                              _p<i> substitution as --xclbin.\n"
"  --input <packed_2bit>       packed-2-bit input (.2bit.bin)\n"
"  --output <path>             output file (TSV or binary)\n"
"  --k {15,21}                 minimizer k\n"
"  --w {10,11}                 minimizer w (10 for k=15; 11 for k=21)\n"
"\n"
"Optional:\n"
"  -k, --kernel <name>         kernel name (default MLIR_AIE)\n"
"  --launch-chunks {1,2,4,8}   tile fan-out (default 4)\n"
"  --n-passes {1,4,8,16}       hash-slice partition count (default 1)\n"
"  --n-chunks-per-launch {1,2,4,8}\n"
"                              chunks processed per silicon dispatch\n"
"                              (default 1; v1.2 (b) batched-dispatch knob.\n"
"                              MUST match the value baked into the xclbin\n"
"                              by IRON Python at build time).\n"
"  --top <N>                   top-N records by position (default 0=all)\n"
"  --iters <N>                 timed iterations (default 1)\n"
"  --warmup <N>                untimed warmup iters (default 0)\n"
"  --output-format {tsv,binary}\n"
"                              tsv: '<pos>\\t<acgt>\\n' per record.\n"
"                              binary: [uint64 n][n × {u64 canonical,\n"
"                              u32 position, u32 _pad}].\n";
}

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "-h" || key == "--help") {
            print_usage(std::cout);
            std::exit(0);
        }
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };
        if (key == "-x" || key == "--xclbin") a.xclbin_template = next();
        else if (key == "-i" || key == "--instr") a.instr_template = next();
        else if (key == "-k" || key == "--kernel") a.kernel = next();
        else if (key == "--input") a.input_path = next();
        else if (key == "--output") a.output_path = next();
        else if (key == "--k") a.k = std::stoi(next());
        else if (key == "--w") a.w = std::stoi(next());
        else if (key == "--top") a.top = std::stoi(next());
        else if (key == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (key == "--n-passes") a.n_passes = std::stoi(next());
        else if (key == "--n-chunks-per-launch") a.n_chunks_per_launch = std::stoi(next());
        else if (key == "--iters") a.iters = std::stoi(next());
        else if (key == "--warmup") a.warmup = std::stoi(next());
        else if (key == "--output-format") a.output_format = next();
        else throw std::runtime_error("unknown arg: " + key);
    }
    if (a.xclbin_template.empty() || a.instr_template.empty() ||
        a.input_path.empty() || a.output_path.empty())
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --input <path> "
            "--output <path> --k {15,21} --w {10,11}");
    if (!((a.k == 15 && a.w == 10) || (a.k == 21 && a.w == 11)))
        throw std::runtime_error(
            "(--k, --w) must be (15,10) or (21,11)");
    if (a.launch_chunks != 1 && a.launch_chunks != 2 &&
        a.launch_chunks != 4 && a.launch_chunks != 8)
        throw std::runtime_error("--launch-chunks must be one of {1,2,4,8}");
    if (a.n_passes != 1 && a.n_passes != 4 &&
        a.n_passes != 8 && a.n_passes != 16)
        throw std::runtime_error("--n-passes must be one of {1, 4, 8, 16}");
    if (a.n_chunks_per_launch != 1 && a.n_chunks_per_launch != 2 &&
        a.n_chunks_per_launch != 4 && a.n_chunks_per_launch != 8)
        throw std::runtime_error(
            "--n-chunks-per-launch must be one of {1, 2, 4, 8}");
    if (a.iters <= 0) throw std::runtime_error("--iters must be > 0");
    if (a.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (a.top < 0) throw std::runtime_error("--top must be >= 0");
    if (a.output_format != "tsv" && a.output_format != "binary")
        throw std::runtime_error(
            "--output-format must be one of {tsv, binary}");
    return a;
}

// ---------------------------------------------------------------------
// Per-pass artifact path resolver. Convention: --xclbin and --instr
// point to either:
//   - pass-0 artifacts (filename contains '_p0') — runner derives
//     _p<i> paths for additional passes.
//   - single-pass artifacts (filename WITHOUT '_p0') — only valid for
//     --n-passes 1; passed through unchanged.
// ---------------------------------------------------------------------
std::string resolve_pass_path(const std::string &template_path,
                              int pass_idx, int n_passes) {
    if (n_passes == 1) {
        return template_path;  // single pass; use template path as-is
    }
    std::string out = template_path;
    std::string from = "_p0";
    std::string to = "_p" + std::to_string(pass_idx);
    auto pos = out.rfind(from);
    if (pos != std::string::npos) {
        out.replace(pos, from.size(), to);
        return out;
    }
    auto dot = template_path.rfind('.');
    if (dot != std::string::npos) {
        out = template_path.substr(0, dot) + "_p" + std::to_string(pass_idx) +
              template_path.substr(dot);
        return out;
    }
    return template_path + "_p" + std::to_string(pass_idx);
}

struct ChunkPlan {
    size_t src_offset;     // first byte of this chunk in source buffer
    size_t bytes;          // bytes copied into the BO
};

std::vector<ChunkPlan> plan_chunks(size_t input_bytes, int overlap_bytes) {
    std::vector<ChunkPlan> plan;
    if (input_bytes == 0) return plan;
    const size_t total_chunk = static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE)
                             + static_cast<size_t>(overlap_bytes);
    const size_t payload_cap = total_chunk - HEADER_BYTES;
    const size_t advance = payload_cap - static_cast<size_t>(overlap_bytes);
    size_t off = 0;
    while (off < input_bytes) {
        size_t end = std::min(off + payload_cap, input_bytes);
        ChunkPlan p;
        p.src_offset = off;
        p.bytes = end - off;
        plan.push_back(p);
        if (end >= input_bytes) break;
        off += advance;
    }
    return plan;
}

// Parse one tile_0 slot into the records vector. The slot is at the
// start of the per-chunk-per-batch joined output blob; slots 1..N_TILES-1
// are duplicates (broadcast topology) and ignored.
//
// `chunk_global_base_bases` is the CHUNK's first-base global offset (in
// bases). Records' `position` is chunk-local; we add this to translate
// to global.
void parse_chunk_tile0(const uint8_t *blob,
                       size_t chunk_global_base_bases,
                       std::vector<MinimizerRecord> &out,
                       uint32_t &max_emit_observed_io) {
    uint32_t emit_count = 0;
    std::memcpy(&emit_count, blob, 4);
    if (emit_count > max_emit_observed_io) {
        max_emit_observed_io = emit_count;
    }
    if (emit_count > static_cast<uint32_t>(MZ_MAX_EMIT_IDX)) {
        emit_count = static_cast<uint32_t>(MZ_MAX_EMIT_IDX);
    }
    const size_t payload_off = 4u;
    out.reserve(out.size() + emit_count);
    for (uint32_t e = 0; e < emit_count; ++e) {
        const uint8_t *rec = blob + payload_off +
                             (size_t)e * (size_t)MZ_RECORD_BYTES;
        MinimizerRecord r{};
        std::memcpy(&r.canonical, rec, 8);
        std::memcpy(&r.position,  rec + 8, 4);
        // Translate to global position.
        r.position += static_cast<uint32_t>(chunk_global_base_bases);
        r._pad = 0;
        out.push_back(r);
    }
}

void emit_tsv(const std::string &path,
              const std::vector<MinimizerRecord> &v, int k) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("cannot open output for write: " + path);
    for (const auto &r : v) {
        f << r.position << '\t'
          << decode_canonical_to_acgt(r.canonical, k) << '\n';
    }
}

void emit_binary_blob(const std::string &path,
                      const std::vector<MinimizerRecord> &v) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open output for write: " + path);
    uint64_t n = static_cast<uint64_t>(v.size());
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &r : v) {
        f.write(reinterpret_cast<const char *>(&r.canonical), 8);
        f.write(reinterpret_cast<const char *>(&r.position),  4);
        f.write(reinterpret_cast<const char *>(&r._pad),      4);
    }
}

// Per-pass loaded artifacts.
struct PassArtifacts {
    xrt::xclbin xclbin;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo bo_instr;
    std::vector<uint32_t> instr;
};

}  // namespace

int main(int argc, char **argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return 2;
    }

    const int overlap_bytes = overlap_bytes_for_kw(args.k, args.w);
    trace("k=" + std::to_string(args.k) +
          " w=" + std::to_string(args.w) +
          " overlap_bytes=" + std::to_string(overlap_bytes) +
          " launch_chunks=" + std::to_string(args.launch_chunks) +
          " n_passes=" + std::to_string(args.n_passes) +
          " n_chunks_per_launch=" + std::to_string(args.n_chunks_per_launch));

    auto input_buf = read_packed_input(args.input_path);
    trace("input bytes=" + std::to_string(input_buf.size()));

    auto chunks = plan_chunks(input_buf.size(), overlap_bytes);
    const size_t n_chunks_host = chunks.size();
    trace("planned chunks=" + std::to_string(n_chunks_host));

    const size_t seq_in_slot_bytes =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes);

    // Per-batch BO sizes: each silicon dispatch processes
    // n_chunks_per_launch chunks. seq_in BO = N_BATCH × slot bytes;
    // partial_out BO = N_BATCH × (N_TILES × 32 KiB).
    const int n_batch = args.n_chunks_per_launch;
    const size_t per_batch_seq_in_bytes =
        static_cast<size_t>(n_batch) * seq_in_slot_bytes;
    const size_t per_pass_out_bytes =
        static_cast<size_t>(n_batch) *
        static_cast<size_t>(args.launch_chunks) *
        static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);

    const size_t n_batches =
        (n_chunks_host + static_cast<size_t>(n_batch) - 1u) /
        static_cast<size_t>(n_batch);
    trace("planned batches=" + std::to_string(n_batches));

    // ------------------------------------------------------------
    // XRT setup — load N_PASSES xclbins.
    // ------------------------------------------------------------
    trace("xrt device open");
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    std::vector<PassArtifacts> per_pass(static_cast<size_t>(args.n_passes));
    for (int p = 0; p < args.n_passes; ++p) {
        std::string xclbin_path =
            resolve_pass_path(args.xclbin_template, p, args.n_passes);
        std::string instr_path =
            resolve_pass_path(args.instr_template, p, args.n_passes);
        trace("pass=" + std::to_string(p) +
              " xclbin=" + xclbin_path +
              " instr=" + instr_path);

        per_pass[p].xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(per_pass[p].xclbin);
        per_pass[p].context = xrt::hw_context(
            device, per_pass[p].xclbin.get_uuid());
        per_pass[p].kernel = xrt::kernel(per_pass[p].context, args.kernel);
        per_pass[p].instr = read_instr_binary(instr_path);

        per_pass[p].bo_instr = xrt::bo(
            device, per_pass[p].instr.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, per_pass[p].kernel.group_id(1));
        auto *p_instr = per_pass[p].bo_instr.map<uint32_t *>();
        std::memcpy(p_instr, per_pass[p].instr.data(),
                    per_pass[p].instr.size() * sizeof(uint32_t));
        per_pass[p].bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // v1.3 ring-of-BOs for pipelined dispatch (mirrors kmer_count v1.3).
    // RING=4 gives XRT 3-deep submit-ahead while host accumulates.
    // seq_in is staged once per BATCH (re-used across all n_passes
    // dispatches of that batch); ping-pongs at batch granularity.
    // sparse_out ping-pongs PER DISPATCH.
    const auto &k0 = per_pass[0].kernel;
    constexpr int RING = 4;
    xrt::bo bo_seq_in_ring[RING];
    xrt::bo bo_sparse_out_ring[RING];
    uint8_t *p_seq_in_ring[RING];
    uint8_t *p_sparse_out_ring[RING];
    for (int s = 0; s < RING; ++s) {
        bo_seq_in_ring[s] = xrt::bo(device, per_batch_seq_in_bytes,
                                    XRT_BO_FLAGS_HOST_ONLY, k0.group_id(3));
        bo_sparse_out_ring[s] = xrt::bo(device, per_pass_out_bytes,
                                        XRT_BO_FLAGS_HOST_ONLY, k0.group_id(4));
        p_seq_in_ring[s] = bo_seq_in_ring[s].map<uint8_t *>();
        p_sparse_out_ring[s] = bo_sparse_out_ring[s].map<uint8_t *>();
    }

    // Host-side accumulator.
    std::vector<MinimizerRecord> all_records;
    all_records.reserve(64 * 1024);

    const unsigned int opcode = 3;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;
    int timed_dispatches = 0;
    uint32_t max_emit_observed = 0;

    struct InflightSlot {
        xrt::run run;
        int seq_slot;
        int out_slot;
        size_t batch_idx;
        int pass_idx;
        int dispatch_idx;
        bool valid;
        std::chrono::high_resolution_clock::time_point t_dispatch;
    };

    // Accumulate one drained dispatch's blob into all_records.
    auto accumulate = [&](const uint8_t *blob, size_t batch_idx,
                          bool keep) {
        if (!keep) return;
        // Per-iteration block layout: N_BATCH × (N_TILES × PARTIAL_PADDED).
        const size_t per_chunk_block_bytes =
            static_cast<size_t>(args.launch_chunks) *
            static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);
        for (int slot = 0; slot < n_batch; ++slot) {
            const size_t ci = batch_idx * static_cast<size_t>(n_batch) +
                              static_cast<size_t>(slot);
            if (ci >= n_chunks_host) break;  // tail-padded slot; skip
            const ChunkPlan &c = chunks[ci];
            // Tile 0 slot is the first PARTIAL_PADDED bytes of the
            // chunk's per-tile-block. Slots 1..N_TILES-1 are duplicates
            // and ignored.
            const uint8_t *chunk_blob =
                blob + static_cast<size_t>(slot) * per_chunk_block_bytes;
            size_t chunk_global_base_bases = c.src_offset * 4u;
            parse_chunk_tile0(chunk_blob,
                              chunk_global_base_bases,
                              all_records,
                              max_emit_observed);
        }
    };

    const int total_iters = args.warmup + args.iters;
    for (int it = 0; it < total_iters; ++it) {
        const bool keep = (it >= args.warmup);
        if (keep) {
            all_records.clear();
            max_emit_observed = 0;
        }

        InflightSlot inflight[RING];
        for (int s = 0; s < RING; ++s) inflight[s].valid = false;

        size_t last_seq_batch[RING];
        for (int s = 0; s < RING; ++s) {
            last_seq_batch[s] = static_cast<size_t>(-1);
        }

        // Linearise (batch, pass) into single dispatch index d.
        const size_t total_dispatches =
            n_batches * static_cast<size_t>(args.n_passes);

        for (size_t d = 0; d < total_dispatches; ++d) {
            const size_t bi = d / static_cast<size_t>(args.n_passes);
            const int p = static_cast<int>(d % static_cast<size_t>(args.n_passes));
            const int out_slot = static_cast<int>(d % static_cast<size_t>(RING));
            const int seq_slot = static_cast<int>(bi % static_cast<size_t>(RING));

            // (1) Drain ring slot we're about to overwrite.
            if (inflight[out_slot].valid) {
                inflight[out_slot].run.wait();
                bo_sparse_out_ring[inflight[out_slot].out_slot].sync(
                    XCL_BO_SYNC_BO_FROM_DEVICE);
                auto t_drain1 = std::chrono::high_resolution_clock::now();

                accumulate(
                    p_sparse_out_ring[inflight[out_slot].out_slot],
                    inflight[out_slot].batch_idx,
                    keep);

                if (keep) {
                    auto td = std::chrono::duration_cast<
                        std::chrono::microseconds>(
                            t_drain1 - inflight[out_slot].t_dispatch).count();
                    float us = static_cast<float>(td);
                    total_us += us;
                    if (us < min_us) min_us = us;
                    if (us > max_us) max_us = us;
                    timed_dispatches++;
                }
                inflight[out_slot].valid = false;
            }

            // (2) Stage seq_in for batch bi if not already staged.
            if (last_seq_batch[seq_slot] != bi) {
                // Drain any other in-flight dispatch reading from this
                // seq_slot before we overwrite it.
                for (int s = 0; s < RING; ++s) {
                    if (inflight[s].valid && inflight[s].seq_slot == seq_slot) {
                        inflight[s].run.wait();
                        bo_sparse_out_ring[inflight[s].out_slot].sync(
                            XCL_BO_SYNC_BO_FROM_DEVICE);
                        accumulate(
                            p_sparse_out_ring[inflight[s].out_slot],
                            inflight[s].batch_idx,
                            keep);
                        if (keep) {
                            auto td = std::chrono::duration_cast<
                                std::chrono::microseconds>(
                                    std::chrono::high_resolution_clock::now() -
                                    inflight[s].t_dispatch).count();
                            float us = static_cast<float>(td);
                            total_us += us;
                            if (us < min_us) min_us = us;
                            if (us > max_us) max_us = us;
                            timed_dispatches++;
                        }
                        inflight[s].valid = false;
                    }
                }

                const size_t batch_first_chunk =
                    bi * static_cast<size_t>(n_batch);
                uint8_t *p_seq_in = p_seq_in_ring[seq_slot];
                std::memset(p_seq_in, 0, per_batch_seq_in_bytes);
                for (int slot = 0; slot < n_batch; ++slot) {
                    const size_t ci = batch_first_chunk +
                                      static_cast<size_t>(slot);
                    uint8_t *slot_base = p_seq_in +
                                         static_cast<size_t>(slot) *
                                             seq_in_slot_bytes;
                    if (ci >= n_chunks_host) continue;  // tail-pad: zero
                    const ChunkPlan &c = chunks[ci];
                    uint32_t actual_bytes_le = static_cast<uint32_t>(c.bytes);
                    std::memcpy(slot_base, &actual_bytes_le, 4);
                    // owned_start_offset_bases for minimizers covers
                    // (k + w - 1) prior bases (window seed). Formula:
                    //   chunk 0: 0 (full payload owned)
                    //   chunk i>0: overlap_bases - (k + w - 2)
                    int32_t owned_start_offset_bases =
                        (ci == 0)
                            ? 0
                            : (overlap_bytes * 4) - (args.k + args.w - 2);
                    std::memcpy(slot_base + 4, &owned_start_offset_bases, 4);
                    std::memcpy(slot_base + HEADER_BYTES,
                                input_buf.data() + c.src_offset,
                                c.bytes);
                }
                bo_seq_in_ring[seq_slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                last_seq_batch[seq_slot] = bi;
            }

            // (3) Skip the per-dispatch zero-output (kernel writes
            //     emit_count prefix unconditionally; parser reads only
            //     exactly emit_count records). Restore via
            //     BIONPU_MZ_OUTPUT_ZERO=1 env if a regression suspected.
            const bool output_zero_enabled =
                (std::getenv("BIONPU_MZ_OUTPUT_ZERO") != nullptr) &&
                std::string(std::getenv("BIONPU_MZ_OUTPUT_ZERO")) == "1";
            if (output_zero_enabled) {
                std::memset(p_sparse_out_ring[out_slot], 0, per_pass_out_bytes);
                bo_sparse_out_ring[out_slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }

            // (4) Issue the kernel — non-blocking.
            auto t_disp = std::chrono::high_resolution_clock::now();
            xrt::run run = per_pass[p].kernel(
                opcode,
                per_pass[p].bo_instr,
                static_cast<unsigned>(per_pass[p].instr.size()),
                bo_seq_in_ring[seq_slot],
                bo_sparse_out_ring[out_slot],
                static_cast<int>(per_batch_seq_in_bytes));

            inflight[out_slot].run = std::move(run);
            inflight[out_slot].seq_slot = seq_slot;
            inflight[out_slot].out_slot = out_slot;
            inflight[out_slot].batch_idx = bi;
            inflight[out_slot].pass_idx = p;
            inflight[out_slot].dispatch_idx = static_cast<int>(d);
            inflight[out_slot].valid = true;
            inflight[out_slot].t_dispatch = t_disp;
        }

        // (5) Drain remaining in-flight slots in dispatch-issue order.
        {
            std::vector<int> drain_order;
            for (int s = 0; s < RING; ++s) {
                if (inflight[s].valid) drain_order.push_back(s);
            }
            std::sort(drain_order.begin(), drain_order.end(),
                      [&](int a, int b) {
                          return inflight[a].dispatch_idx <
                                 inflight[b].dispatch_idx;
                      });
            for (int s : drain_order) {
                inflight[s].run.wait();
                bo_sparse_out_ring[inflight[s].out_slot].sync(
                    XCL_BO_SYNC_BO_FROM_DEVICE);
                auto t_drain1 = std::chrono::high_resolution_clock::now();
                accumulate(
                    p_sparse_out_ring[inflight[s].out_slot],
                    inflight[s].batch_idx,
                    keep);
                if (keep) {
                    auto td = std::chrono::duration_cast<
                        std::chrono::microseconds>(
                            t_drain1 - inflight[s].t_dispatch).count();
                    float us = static_cast<float>(td);
                    total_us += us;
                    if (us < min_us) min_us = us;
                    if (us > max_us) max_us = us;
                    timed_dispatches++;
                }
                inflight[s].valid = false;
            }
        }
    }

    if (max_emit_observed >= static_cast<uint32_t>(MZ_MAX_EMIT_IDX)) {
        trace("WARN: max_emit_observed=" +
              std::to_string(max_emit_observed) +
              " hit cap MZ_MAX_EMIT_IDX=" +
              std::to_string(MZ_MAX_EMIT_IDX) +
              " — kernel dropped emits; increase --n-passes or use smaller chunks.");
    }
    trace("total minimizer records=" + std::to_string(all_records.size()));

    // Sort by (position asc, canonical asc); de-duplicate adjacent
    // records (defensive — owned-range gate should already prevent
    // overlap dupes; multi-pass is partition-disjoint by design).
    auto t_sort0 = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par_unseq,
              all_records.begin(), all_records.end(),
              [](const MinimizerRecord &a, const MinimizerRecord &b) {
                  if (a.position != b.position) return a.position < b.position;
                  return a.canonical < b.canonical;
              });
    {
        auto last = std::unique(
            all_records.begin(), all_records.end(),
            [](const MinimizerRecord &a, const MinimizerRecord &b) {
                return a.position == b.position &&
                       a.canonical == b.canonical;
            });
        all_records.erase(last, all_records.end());
    }
    auto t_sort1 = std::chrono::high_resolution_clock::now();
    auto sort_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t_sort1 - t_sort0).count();
    trace("unique minimizer records=" + std::to_string(all_records.size()) +
          " sort+dedup wall=" + std::to_string(sort_us) + "us");

    if (args.top > 0 && static_cast<size_t>(args.top) < all_records.size()) {
        all_records.resize(static_cast<size_t>(args.top));
    }

    if (args.output_format == "binary") {
        emit_binary_blob(args.output_path, all_records);
        trace("wrote binary blob (" + std::to_string(all_records.size()) +
              " records) to " + args.output_path);
    } else {
        emit_tsv(args.output_path, all_records, args.k);
        trace("wrote TSV (" + std::to_string(all_records.size()) +
              " records) to " + args.output_path);
    }

    float avg_us = (timed_dispatches > 0) ?
                   (total_us / timed_dispatches) : 0.0f;
    if (timed_dispatches == 0) {
        min_us = 0.0f;
        max_us = 0.0f;
    }
    std::cout << "Avg NPU time: " << avg_us << "us." << std::endl;
    std::cout << "Min NPU time: " << min_us << "us." << std::endl;
    std::cout << "Max NPU time: " << max_us << "us." << std::endl;
    std::cout << "\nPASS!\n\n";
    return 0;
}
