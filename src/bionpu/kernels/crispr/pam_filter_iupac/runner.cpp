// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// runner.cpp — Host runner for the v0 IUPAC PAM filter kernel
// (Track A v0 — base editor design).
//
// The PAM mask + length live in the per-chunk 24-byte header. One
// xclbin handles every Cas9 PAM variant via runtime header args.
//
// Output formats:
//   - tsv: "<global_pos>\t<strand>\n" per record.
//   - binary: [u64 n_records][n × {u32 pos, u8 strand, u8 _pad,
//                                  u16 _pad2, u64 _pad3}].

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
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;
constexpr int32_t PFI_OVERLAP_BYTES = 8;
constexpr int32_t PARTIAL_OUT_BYTES_PADDED = 32768;
constexpr int32_t PFI_RECORD_BYTES = 16;
constexpr int32_t PFI_MAX_EMIT_IDX = 2046;
constexpr size_t  HEADER_BYTES = 24;
constexpr int32_t PFI_PAM_LEN_MAX = 8;

void trace(const std::string &msg) {
    std::cerr << "[pam_filter_iupac_runner] " << msg << std::endl;
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

// Encode an IUPAC PAM string into (pam_mask u32, pam_length u8).
// Bit map per nibble: A=1, C=2, G=4, T=8.
void encode_pam_iupac(const std::string &pam,
                      uint32_t &mask_out, uint8_t &len_out) {
    int p = static_cast<int>(pam.size());
    if (p < 1 || p > PFI_PAM_LEN_MAX)
        throw std::runtime_error(
            "PAM length must be 1.." + std::to_string(PFI_PAM_LEN_MAX) +
            "; got " + std::to_string(p));
    uint32_t mask = 0u;
    for (int i = 0; i < p; ++i) {
        char c = pam[i];
        uint8_t nib;
        switch (c) {
            case 'A': case 'a': nib = 0x1; break;
            case 'C': case 'c': nib = 0x2; break;
            case 'G': case 'g': nib = 0x4; break;
            case 'T': case 't': nib = 0x8; break;
            case 'R': case 'r': nib = 0x5; break;  // A|G
            case 'Y': case 'y': nib = 0xA; break;  // C|T
            case 'S': case 's': nib = 0x6; break;  // G|C
            case 'W': case 'w': nib = 0x9; break;  // A|T
            case 'K': case 'k': nib = 0xC; break;  // G|T
            case 'M': case 'm': nib = 0x3; break;  // A|C
            case 'B': case 'b': nib = 0xE; break;  // C|G|T
            case 'D': case 'd': nib = 0xD; break;  // A|G|T
            case 'H': case 'h': nib = 0xB; break;  // A|C|T
            case 'V': case 'v': nib = 0x7; break;  // A|C|G
            case 'N': case 'n': nib = 0xF; break;  // A|C|G|T
            default:
                throw std::runtime_error(
                    std::string("non-IUPAC PAM base: ") + c);
        }
        mask |= ((uint32_t)nib & 0xFu) << (4 * i);
    }
    mask_out = mask;
    len_out = (uint8_t)p;
}

struct PamHit {
    uint32_t position;     // global, in bases
    uint8_t  strand;       // 0 = fwd (v0 only fwd; host does RC)
    uint8_t  _pad;
    uint16_t _pad2;
    uint64_t _pad3;
};

struct Args {
    std::string xclbin_path;
    std::string instr_path;
    std::string kernel = "MLIR_AIE";
    std::string input_path;
    std::string output_path;
    std::string output_format = "tsv";  // {tsv, binary}
    std::string pam_iupac;              // IUPAC string e.g. "NGG"
    int top = 0;
    int launch_chunks = 4;
    int n_chunks_per_launch = 1;
    int iters = 1;
    int warmup = 0;
};

void print_usage(std::ostream &os) {
    os <<
"Usage: pam_filter_iupac_runner [options]\n"
"\n"
"Required:\n"
"  -x, --xclbin <path>         xclbin file (final.xclbin)\n"
"  -i, --instr <path>          NPU instructions binary (insts.bin)\n"
"  --input <packed_2bit>       packed-2-bit input (.2bit.bin)\n"
"  --output <path>             output file (TSV or binary)\n"
"  --pam <IUPAC>               IUPAC PAM string (e.g. NGG, NRN, NNNRRT)\n"
"\n"
"Optional:\n"
"  -k, --kernel <name>         kernel name (default MLIR_AIE)\n"
"  --launch-chunks {1,2,4,8}   tile fan-out (default 4)\n"
"  --n-chunks-per-launch {1,2,4,8} (default 1)\n"
"  --top <N>                   top-N records by position (default 0=all)\n"
"  --iters <N>                 timed iterations (default 1)\n"
"  --warmup <N>                untimed warmup iters (default 0)\n"
"  --output-format {tsv,binary}\n"
"                              tsv: '<pos>\\t<strand>\\n' per record.\n"
"                              binary: [u64 n][n × 16-byte records].\n";
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
        else if (key == "--top") a.top = std::stoi(next());
        else if (key == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (key == "--n-chunks-per-launch") a.n_chunks_per_launch = std::stoi(next());
        else if (key == "--iters") a.iters = std::stoi(next());
        else if (key == "--warmup") a.warmup = std::stoi(next());
        else if (key == "--output-format") a.output_format = next();
        else if (key == "--pam") a.pam_iupac = next();
        else throw std::runtime_error("unknown arg: " + key);
    }
    if (a.xclbin_path.empty() || a.instr_path.empty() ||
        a.input_path.empty() || a.output_path.empty() ||
        a.pam_iupac.empty())
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --input <path> "
            "--output <path> --pam <IUPAC>");
    if (a.launch_chunks != 1 && a.launch_chunks != 2 &&
        a.launch_chunks != 4 && a.launch_chunks != 8)
        throw std::runtime_error("--launch-chunks must be one of {1,2,4,8}");
    if (a.n_chunks_per_launch != 1 && a.n_chunks_per_launch != 2 &&
        a.n_chunks_per_launch != 4 && a.n_chunks_per_launch != 8)
        throw std::runtime_error("--n-chunks-per-launch must be one of {1,2,4,8}");
    if (a.iters <= 0) throw std::runtime_error("--iters must be > 0");
    if (a.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (a.top < 0) throw std::runtime_error("--top must be >= 0");
    if (a.output_format != "tsv" && a.output_format != "binary")
        throw std::runtime_error("--output-format must be {tsv, binary}");
    return a;
}

struct ChunkPlan {
    size_t src_offset;     // first byte of this chunk in source buffer
    size_t bytes;          // bytes copied into the BO (payload)
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

void parse_chunk_tile0(const uint8_t *blob,
                       size_t chunk_global_base_bases,
                       std::vector<PamHit> &out,
                       uint32_t &max_emit_observed_io) {
    uint32_t emit_count = 0;
    std::memcpy(&emit_count, blob, 4);
    if (emit_count > max_emit_observed_io) {
        max_emit_observed_io = emit_count;
    }
    if (emit_count > static_cast<uint32_t>(PFI_MAX_EMIT_IDX)) {
        emit_count = static_cast<uint32_t>(PFI_MAX_EMIT_IDX);
    }
    out.reserve(out.size() + emit_count);
    for (uint32_t e = 0; e < emit_count; ++e) {
        const uint8_t *rec = blob + 4 +
                             (size_t)e * (size_t)PFI_RECORD_BYTES;
        PamHit r{};
        uint32_t pos_local = 0;
        std::memcpy(&pos_local, rec, 4);
        r.position = pos_local + static_cast<uint32_t>(chunk_global_base_bases);
        r.strand   = rec[4];
        r._pad     = rec[5];
        r._pad2 = 0;
        r._pad3 = 0;
        out.push_back(r);
    }
}

void emit_tsv(const std::string &path,
              const std::vector<PamHit> &v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output for write: " + path);
    for (const auto &r : v) {
        f << r.position << '\t' << static_cast<int>(r.strand) << '\n';
    }
}

void emit_binary_blob(const std::string &path,
                      const std::vector<PamHit> &v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output for write: " + path);
    uint64_t n = static_cast<uint64_t>(v.size());
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &r : v) {
        f.write(reinterpret_cast<const char *>(&r.position),   4);
        f.write(reinterpret_cast<const char *>(&r.strand),     1);
        f.write(reinterpret_cast<const char *>(&r._pad),       1);
        f.write(reinterpret_cast<const char *>(&r._pad2),      2);
        f.write(reinterpret_cast<const char *>(&r._pad3),      8);
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

    uint32_t pam_mask = 0u;
    uint8_t  pam_length = 0;
    try {
        encode_pam_iupac(args.pam_iupac, pam_mask, pam_length);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }

    const int overlap_bytes = PFI_OVERLAP_BYTES;
    trace("pam=" + args.pam_iupac +
          " pam_mask=0x" + ([&]{ char b[16]; std::snprintf(b, sizeof(b), "%08x", pam_mask); return std::string(b); })() +
          " pam_length=" + std::to_string((int)pam_length) +
          " overlap_bytes=" + std::to_string(overlap_bytes) +
          " launch_chunks=" + std::to_string(args.launch_chunks) +
          " n_chunks_per_launch=" + std::to_string(args.n_chunks_per_launch));

    auto input_buf = read_packed_input(args.input_path);
    trace("input bytes=" + std::to_string(input_buf.size()));

    auto chunks = plan_chunks(input_buf.size(), overlap_bytes);
    const size_t n_chunks_host = chunks.size();
    trace("planned chunks=" + std::to_string(n_chunks_host));

    const size_t seq_in_slot_bytes =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes);

    const int n_batch = args.n_chunks_per_launch;
    const size_t per_batch_seq_in_bytes =
        static_cast<size_t>(n_batch) * seq_in_slot_bytes;
    const size_t per_dispatch_out_bytes =
        static_cast<size_t>(n_batch) *
        static_cast<size_t>(args.launch_chunks) *
        static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);

    const size_t n_batches =
        (n_chunks_host + static_cast<size_t>(n_batch) - 1u) /
        static_cast<size_t>(n_batch);
    trace("planned batches=" + std::to_string(n_batches));

    // ------------------------------------------------------------
    // XRT setup — single xclbin (v0; no multi-pass).
    // ------------------------------------------------------------
    trace("xrt device open");
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    auto xclbin = xrt::xclbin(args.xclbin_path);
    device.register_xclbin(xclbin);
    auto context = xrt::hw_context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, args.kernel);
    auto instr = read_instr_binary(args.instr_path);

    auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto *p_instr = bo_instr.map<uint32_t *>();
    std::memcpy(p_instr, instr.data(), instr.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Ring of BOs for pipelined dispatch (mirrors primer_scan).
    constexpr int RING = 4;
    xrt::bo bo_seq_in_ring[RING];
    xrt::bo bo_sparse_out_ring[RING];
    uint8_t *p_seq_in_ring[RING];
    uint8_t *p_sparse_out_ring[RING];
    for (int s = 0; s < RING; ++s) {
        bo_seq_in_ring[s] = xrt::bo(device, per_batch_seq_in_bytes,
                                    XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        bo_sparse_out_ring[s] = xrt::bo(device, per_dispatch_out_bytes,
                                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        p_seq_in_ring[s] = bo_seq_in_ring[s].map<uint8_t *>();
        p_sparse_out_ring[s] = bo_sparse_out_ring[s].map<uint8_t *>();
    }

    std::vector<PamHit> all_records;
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
        bool valid;
        std::chrono::high_resolution_clock::time_point t_dispatch;
    };

    auto accumulate = [&](const uint8_t *blob, size_t batch_idx,
                          bool keep) {
        if (!keep) return;
        const size_t per_chunk_block_bytes =
            static_cast<size_t>(args.launch_chunks) *
            static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);
        for (int slot = 0; slot < n_batch; ++slot) {
            const size_t ci = batch_idx * static_cast<size_t>(n_batch) +
                              static_cast<size_t>(slot);
            if (ci >= n_chunks_host) break;
            const ChunkPlan &c = chunks[ci];
            const uint8_t *chunk_blob =
                blob + static_cast<size_t>(slot) * per_chunk_block_bytes;
            size_t chunk_global_base_bases = c.src_offset * 4u;
            parse_chunk_tile0(chunk_blob, chunk_global_base_bases,
                              all_records, max_emit_observed);
        }
    };

    auto stage_batch = [&](int seq_slot, size_t bi) {
        const size_t batch_first_chunk = bi * static_cast<size_t>(n_batch);
        uint8_t *p_seq_in = p_seq_in_ring[seq_slot];
        std::memset(p_seq_in, 0, per_batch_seq_in_bytes);
        for (int slot = 0; slot < n_batch; ++slot) {
            const size_t ci = batch_first_chunk + static_cast<size_t>(slot);
            uint8_t *slot_base = p_seq_in +
                                 static_cast<size_t>(slot) * seq_in_slot_bytes;
            if (ci >= n_chunks_host) continue;  // tail-pad: zero
            const ChunkPlan &c = chunks[ci];

            // ----- Header layout (24 bytes) -----
            // [0..3]   pam_mask  (uint32 LE)
            std::memcpy(slot_base + 0, &pam_mask, 4);
            // [4]      pam_length (uint8)
            slot_base[4] = pam_length;
            // [5..15]  zero (already memset)
            // [16..19] actual_payload_bytes (uint32 LE)
            uint32_t actual_bytes_le = static_cast<uint32_t>(c.bytes);
            std::memcpy(slot_base + 16, &actual_bytes_le, 4);
            // [20..23] owned_start_offset_bases (int32 LE)
            int32_t owned_start_offset_bases =
                (ci == 0)
                    ? 0
                    : (overlap_bytes * 4) - (static_cast<int>(pam_length) - 1);
            std::memcpy(slot_base + 20, &owned_start_offset_bases, 4);

            // ----- Payload -----
            std::memcpy(slot_base + HEADER_BYTES,
                        input_buf.data() + c.src_offset,
                        c.bytes);
        }
        bo_seq_in_ring[seq_slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
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

        for (size_t d = 0; d < n_batches; ++d) {
            const size_t bi = d;
            const int out_slot = static_cast<int>(d % static_cast<size_t>(RING));
            const int seq_slot = out_slot;

            // Drain ring slot we're about to overwrite.
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

            if (last_seq_batch[seq_slot] != bi) {
                stage_batch(seq_slot, bi);
                last_seq_batch[seq_slot] = bi;
            }

            auto t_disp = std::chrono::high_resolution_clock::now();
            xrt::run run = kernel(
                opcode,
                bo_instr,
                static_cast<unsigned>(instr.size()),
                bo_seq_in_ring[seq_slot],
                bo_sparse_out_ring[out_slot],
                static_cast<int>(per_batch_seq_in_bytes));

            inflight[out_slot].run = std::move(run);
            inflight[out_slot].seq_slot = seq_slot;
            inflight[out_slot].out_slot = out_slot;
            inflight[out_slot].batch_idx = bi;
            inflight[out_slot].valid = true;
            inflight[out_slot].t_dispatch = t_disp;
        }

        // Drain remaining in-flight slots in dispatch-issue order.
        std::vector<int> drain_order;
        for (int s = 0; s < RING; ++s) {
            if (inflight[s].valid) drain_order.push_back(s);
        }
        std::sort(drain_order.begin(), drain_order.end(),
                  [&](int a, int b) {
                      return inflight[a].batch_idx <
                             inflight[b].batch_idx;
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

    if (max_emit_observed >= static_cast<uint32_t>(PFI_MAX_EMIT_IDX)) {
        trace("WARN: max_emit_observed=" +
              std::to_string(max_emit_observed) +
              " hit cap PFI_MAX_EMIT_IDX=" +
              std::to_string(PFI_MAX_EMIT_IDX) +
              " — kernel dropped emits; chunk has too many PAM hits");
    }
    trace("total PAM hits=" + std::to_string(all_records.size()));

    // Sort by (position asc, strand asc); de-duplicate.
    auto t_sort0 = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par_unseq,
              all_records.begin(), all_records.end(),
              [](const PamHit &a, const PamHit &b) {
                  if (a.position != b.position) return a.position < b.position;
                  return a.strand < b.strand;
              });
    auto last = std::unique(
        all_records.begin(), all_records.end(),
        [](const PamHit &a, const PamHit &b) {
            return a.position == b.position && a.strand == b.strand;
        });
    all_records.erase(last, all_records.end());
    auto t_sort1 = std::chrono::high_resolution_clock::now();
    auto sort_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t_sort1 - t_sort0).count();
    trace("unique hits=" + std::to_string(all_records.size()) +
          " sort+dedup wall=" + std::to_string(sort_us) + "us");

    if (args.top > 0 && static_cast<size_t>(args.top) < all_records.size()) {
        all_records.resize(static_cast<size_t>(args.top));
    }

    if (args.output_format == "binary") {
        emit_binary_blob(args.output_path, all_records);
        trace("wrote binary blob (" + std::to_string(all_records.size()) +
              " records) to " + args.output_path);
    } else {
        emit_tsv(args.output_path, all_records);
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
