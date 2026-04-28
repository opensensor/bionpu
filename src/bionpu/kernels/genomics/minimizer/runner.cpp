// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// runner.cpp — Host runner for the v0 minimizer kernel.
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
//     Position is the chunk-local 0-indexed start offset (in bases).
//     The runner translates to a global offset by adding chunk's
//     ``src_offset * 4`` (src_offset is in bytes; ×4 for bases).
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

constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;
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
    std::string xclbin_path;
    std::string instr_path;
    std::string kernel = "MLIR_AIE";
    std::string input_path;
    std::string output_path;
    std::string output_format = "tsv";  // {tsv, binary}
    int k = 15;
    int w = 10;
    int top = 0;            // 0 = all
    int launch_chunks = 4;
    int iters = 1;
    int warmup = 0;
};

void print_usage(std::ostream &os) {
    os <<
"Usage: minimizer_runner [options]\n"
"\n"
"Required:\n"
"  -x, --xclbin <path>         xclbin file (final.xclbin)\n"
"  -i, --instr <path>          NPU instructions binary (insts.bin)\n"
"  --input <packed_2bit>       packed-2-bit input (.2bit.bin)\n"
"  --output <path>             output file (TSV or binary)\n"
"  --k {15,21}                 minimizer k\n"
"  --w {10,11}                 minimizer w (10 for k=15; 11 for k=21)\n"
"\n"
"Optional:\n"
"  -k, --kernel <name>         kernel name (default MLIR_AIE)\n"
"  --launch-chunks {1,2,4,8}   tile fan-out (default 4)\n"
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
        if (key == "-x" || key == "--xclbin") a.xclbin_path = next();
        else if (key == "-i" || key == "--instr") a.instr_path = next();
        else if (key == "-k" || key == "--kernel") a.kernel = next();
        else if (key == "--input") a.input_path = next();
        else if (key == "--output") a.output_path = next();
        else if (key == "--k") a.k = std::stoi(next());
        else if (key == "--w") a.w = std::stoi(next());
        else if (key == "--top") a.top = std::stoi(next());
        else if (key == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (key == "--iters") a.iters = std::stoi(next());
        else if (key == "--warmup") a.warmup = std::stoi(next());
        else if (key == "--output-format") a.output_format = next();
        else throw std::runtime_error("unknown arg: " + key);
    }
    if (a.xclbin_path.empty() || a.instr_path.empty() ||
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
    if (a.iters <= 0) throw std::runtime_error("--iters must be > 0");
    if (a.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (a.top < 0) throw std::runtime_error("--top must be >= 0");
    if (a.output_format != "tsv" && a.output_format != "binary")
        throw std::runtime_error(
            "--output-format must be one of {tsv, binary}");
    return a;
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

// Parse one chunk's tile_0 slot from the joined BO; append global-
// position records to `out`. `chunk_global_base_bases` is the
// chunk's first-base global offset (in bases).
void parse_chunk_tile0(const uint8_t *blob, size_t blob_size,
                       size_t chunk_global_base_bases,
                       int n_tiles,
                       std::vector<MinimizerRecord> &out,
                       uint32_t &max_emit_observed_io) {
    (void)n_tiles;
    if (blob_size < 4) return;
    uint32_t emit_count = 0;
    std::memcpy(&emit_count, blob, 4);
    if (emit_count > max_emit_observed_io) {
        max_emit_observed_io = emit_count;
    }
    if (emit_count > static_cast<uint32_t>(MZ_MAX_EMIT_IDX)) {
        emit_count = static_cast<uint32_t>(MZ_MAX_EMIT_IDX);
    }
    const size_t payload_off = 4u;
    const size_t need = static_cast<size_t>(emit_count) *
                        static_cast<size_t>(MZ_RECORD_BYTES);
    if (payload_off + need > blob_size) return;
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
          " launch_chunks=" + std::to_string(args.launch_chunks));

    auto input_buf = read_packed_input(args.input_path);
    trace("input bytes=" + std::to_string(input_buf.size()));

    auto chunks = plan_chunks(input_buf.size(), overlap_bytes);
    const size_t n_chunks_host = chunks.size();
    trace("planned chunks=" + std::to_string(n_chunks_host));

    const size_t seq_in_slot_bytes =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes);
    const size_t per_dispatch_seq_in_bytes = seq_in_slot_bytes;
    const size_t per_dispatch_out_bytes =
        static_cast<size_t>(args.launch_chunks) *
        static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);

    trace("xrt device open");
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    auto xclbin = xrt::xclbin(args.xclbin_path);
    device.register_xclbin(xclbin);
    auto context = xrt::hw_context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, args.kernel);
    auto instr = read_instr_binary(args.instr_path);

    auto bo_instr = xrt::bo(
        device, instr.size() * sizeof(uint32_t),
        XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    {
        auto *p_instr = bo_instr.map<uint32_t *>();
        std::memcpy(p_instr, instr.data(), instr.size() * sizeof(uint32_t));
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    auto bo_seq_in = xrt::bo(device, per_dispatch_seq_in_bytes,
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_sparse_out = xrt::bo(device, per_dispatch_out_bytes,
                                 XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto *p_seq_in = bo_seq_in.map<uint8_t *>();
    auto *p_sparse_out = bo_sparse_out.map<uint8_t *>();

    std::vector<MinimizerRecord> all_records;
    all_records.reserve(64 * 1024);

    const unsigned int opcode = 3;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;
    int timed_dispatches = 0;
    uint32_t max_emit_observed = 0;

    const int total_iters = args.warmup + args.iters;
    for (int it = 0; it < total_iters; ++it) {
        const bool keep = (it >= args.warmup);
        if (keep) {
            all_records.clear();
            max_emit_observed = 0;
        }

        for (size_t ci = 0; ci < n_chunks_host; ++ci) {
            const ChunkPlan &c = chunks[ci];

            // Stage chunk into seq_in BO.
            std::memset(p_seq_in, 0, per_dispatch_seq_in_bytes);
            uint32_t actual_bytes_le = static_cast<uint32_t>(c.bytes);
            std::memcpy(p_seq_in, &actual_bytes_le, 4);
            // owned_start_offset_bases:
            //   chunk 0: 0 (full payload owned)
            //   chunk i (i>0): overlap_bases - (k+w-2)
            int32_t owned_start_offset_bases =
                (ci == 0)
                    ? 0
                    : (overlap_bytes * 4) - (args.k + args.w - 2);
            std::memcpy(p_seq_in + 4, &owned_start_offset_bases, 4);
            std::memcpy(p_seq_in + HEADER_BYTES,
                        input_buf.data() + c.src_offset, c.bytes);
            bo_seq_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Zero the output BO (defensive; v0 small fixture).
            std::memset(p_sparse_out, 0, per_dispatch_out_bytes);
            bo_sparse_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Dispatch.
            auto t0 = std::chrono::high_resolution_clock::now();
            xrt::run run = kernel(
                opcode,
                bo_instr,
                static_cast<unsigned>(instr.size()),
                bo_seq_in,
                bo_sparse_out,
                static_cast<int>(per_dispatch_seq_in_bytes));
            run.wait();
            bo_sparse_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto t1 = std::chrono::high_resolution_clock::now();

            if (keep) {
                // Tile 0 slot starts at offset 0 of the joined output.
                size_t chunk_global_base_bases = c.src_offset * 4u;
                parse_chunk_tile0(p_sparse_out,
                                  static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED),
                                  chunk_global_base_bases,
                                  args.launch_chunks,
                                  all_records,
                                  max_emit_observed);
                auto td = std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count();
                float us = static_cast<float>(td);
                total_us += us;
                if (us < min_us) min_us = us;
                if (us > max_us) max_us = us;
                timed_dispatches++;
            }
        }
    }

    if (max_emit_observed >= static_cast<uint32_t>(MZ_MAX_EMIT_IDX)) {
        trace("WARN: max_emit_observed=" +
              std::to_string(max_emit_observed) +
              " hit cap MZ_MAX_EMIT_IDX=" +
              std::to_string(MZ_MAX_EMIT_IDX) +
              " — kernel dropped emits; use smaller chunks.");
    }
    trace("total minimizer records=" + std::to_string(all_records.size()));

    // De-duplicate adjacent records that may straddle chunk boundaries
    // (the owned-range gate should prevent these but we belt-and-brace).
    // Sort by position; equal-position equal-canonical records collapse.
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
    trace("unique minimizer records=" + std::to_string(all_records.size()));

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
