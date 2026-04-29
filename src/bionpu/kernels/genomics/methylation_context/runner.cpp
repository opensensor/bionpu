// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// runner.cpp — Host runner for methylation_context v0.

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
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

#ifndef MC_SEQ_IN_CHUNK_BYTES_BASE
#define MC_SEQ_IN_CHUNK_BYTES_BASE 4096
#endif
constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = MC_SEQ_IN_CHUNK_BYTES_BASE;
constexpr int32_t SEQ_IN_OVERLAP = 4;
constexpr int32_t HEADER_BYTES = 8;
constexpr int32_t PARTIAL_OUT_BYTES_PADDED = 32768;
constexpr int32_t RECORD_BYTES = 8;
constexpr int32_t MC_MAX_EMIT_IDX = 4094;

struct Args {
    std::string xclbin;
    std::string instr;
    std::string kernel = "MLIR_AIE";
    std::string input_path;
    std::string output_path;
    std::string output_format = "tsv";
    int launch_chunks = 4;
    int n_chunks_per_launch = 1;
    int top = 0;
    int iters = 1;
    int warmup = 0;
    int n_bases = -1;
};

struct ChunkPlan {
    size_t src_offset;
    size_t bytes;
};

struct Record {
    uint32_t pos;
    uint8_t strand;
    uint8_t context;
};

void print_usage(std::ostream &os) {
    os <<
"Usage: methylation_context_runner [options]\n"
"\n"
"Required:\n"
"  -x, --xclbin <path>         xclbin file\n"
"  -i, --instr <path>          NPU instructions binary\n"
"  --input <packed_2bit>       packed-2-bit input\n"
"  --n-bases <N>               number of valid bases in packed input\n"
"  --output <path>             output file\n"
"\n"
"Optional:\n"
"  -k, --kernel <name>         kernel name (default MLIR_AIE)\n"
"  --launch-chunks {1,2,4,8}   tile fan-out (default 4)\n"
"  --n-chunks-per-launch {1,2,4,8}\n"
"  --top <N>                   top-N records (0=all)\n"
"  --iters <N>                 timed iterations (default 1)\n"
"  --warmup <N>                untimed warmup iters (default 0)\n"
"  --output-format {tsv,binary}\n";
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
        if (key == "-x" || key == "--xclbin") a.xclbin = next();
        else if (key == "-i" || key == "--instr") a.instr = next();
        else if (key == "-k" || key == "--kernel") a.kernel = next();
        else if (key == "--input") a.input_path = next();
        else if (key == "--n-bases") a.n_bases = std::stoi(next());
        else if (key == "--output") a.output_path = next();
        else if (key == "--output-format") a.output_format = next();
        else if (key == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (key == "--n-chunks-per-launch") a.n_chunks_per_launch = std::stoi(next());
        else if (key == "--top") a.top = std::stoi(next());
        else if (key == "--iters") a.iters = std::stoi(next());
        else if (key == "--warmup") a.warmup = std::stoi(next());
        else throw std::runtime_error("unknown arg: " + key);
    }
    if (a.xclbin.empty() || a.instr.empty() ||
        a.input_path.empty() || a.output_path.empty() || a.n_bases < 0) {
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --input <path> "
            "--n-bases <N> --output <path>");
    }
    if (a.launch_chunks != 1 && a.launch_chunks != 2 &&
        a.launch_chunks != 4 && a.launch_chunks != 8)
        throw std::runtime_error("--launch-chunks must be one of {1,2,4,8}");
    if (a.n_chunks_per_launch != 1 && a.n_chunks_per_launch != 2 &&
        a.n_chunks_per_launch != 4 && a.n_chunks_per_launch != 8)
        throw std::runtime_error(
            "--n-chunks-per-launch must be one of {1,2,4,8}");
    if (a.output_format != "tsv" && a.output_format != "binary")
        throw std::runtime_error("--output-format must be one of {tsv,binary}");
    if (a.top < 0) throw std::runtime_error("--top must be >= 0");
    if (a.iters <= 0) throw std::runtime_error("--iters must be > 0");
    if (a.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    return a;
}

std::vector<uint32_t> read_instr_binary(const std::string &path) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) throw std::runtime_error("failed to open instructions: " + path);
    std::vector<uint32_t> out;
    uint32_t word = 0;
    while (fh.read(reinterpret_cast<char *>(&word), sizeof(word)))
        out.push_back(word);
    return out;
}

std::vector<uint8_t> read_packed_input(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input: " + path);
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz < 0) throw std::runtime_error("bad seek on input: " + path);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

std::vector<ChunkPlan> plan_chunks(size_t input_bytes) {
    std::vector<ChunkPlan> plan;
    if (input_bytes == 0) return plan;
    const size_t total_chunk =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + SEQ_IN_OVERLAP);
    const size_t payload_cap = total_chunk - static_cast<size_t>(HEADER_BYTES);
    const size_t advance = payload_cap - static_cast<size_t>(SEQ_IN_OVERLAP);
    size_t off = 0;
    while (off < input_bytes) {
        size_t end = std::min(off + payload_cap, input_bytes);
        plan.push_back(ChunkPlan{off, end - off});
        if (end >= input_bytes) break;
        off += advance;
    }
    return plan;
}

void parse_chunk_tile0(const uint8_t *blob,
                       size_t chunk_global_base,
                       int n_bases,
                       std::vector<Record> &out,
                       uint32_t &max_emit_observed) {
    uint32_t emit_count = 0;
    std::memcpy(&emit_count, blob, 4);
    max_emit_observed = std::max(max_emit_observed, emit_count);
    if (emit_count > static_cast<uint32_t>(MC_MAX_EMIT_IDX)) {
        emit_count = static_cast<uint32_t>(MC_MAX_EMIT_IDX);
    }
    out.reserve(out.size() + emit_count);
    for (uint32_t e = 0; e < emit_count; ++e) {
        const uint8_t *rec = blob + 4u + (size_t)e * (size_t)RECORD_BYTES;
        uint32_t local_pos = 0;
        std::memcpy(&local_pos, rec, 4);
        uint32_t global_pos =
            static_cast<uint32_t>(chunk_global_base) + local_pos;
        if (global_pos >= static_cast<uint32_t>(n_bases)) {
            continue;
        }
        out.push_back(Record{global_pos, rec[4], rec[5]});
    }
}

bool record_less(const Record &a, const Record &b) {
    return std::tie(a.pos, a.strand, a.context) <
           std::tie(b.pos, b.strand, b.context);
}

bool record_equal(const Record &a, const Record &b) {
    return a.pos == b.pos && a.strand == b.strand && a.context == b.context;
}

void emit_tsv(const std::string &path, const std::vector<Record> &v) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    for (const Record &r : v) {
        f << r.pos << '\t' << (int)r.strand << '\t'
          << (int)r.context << '\n';
    }
}

void emit_binary(const std::string &path, const std::vector<Record> &v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    uint64_t n = static_cast<uint64_t>(v.size());
    f.write(reinterpret_cast<const char *>(&n), 8);
    for (const Record &r : v) {
        f.write(reinterpret_cast<const char *>(&r.pos), 4);
        f.write(reinterpret_cast<const char *>(&r.strand), 1);
        f.write(reinterpret_cast<const char *>(&r.context), 1);
        uint16_t pad = 0;
        f.write(reinterpret_cast<const char *>(&pad), 2);
    }
}

void trace(const std::string &msg) {
    std::cerr << "[methylation_context_runner] " << msg << std::endl;
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

    auto input_buf = read_packed_input(args.input_path);
    const size_t expected_input_bytes =
        (static_cast<size_t>(args.n_bases) + 3u) / 4u;
    if (input_buf.size() != expected_input_bytes) {
        std::cerr << "ERROR: input bytes=" << input_buf.size()
                  << " but n_bases=" << args.n_bases
                  << " requires " << expected_input_bytes << "\n";
        return 2;
    }

    auto chunks = plan_chunks(input_buf.size());
    const size_t n_chunks_host = chunks.size();
    const size_t seq_in_slot_bytes =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + SEQ_IN_OVERLAP);
    const int n_batch = args.n_chunks_per_launch;
    const size_t per_batch_seq_in_bytes =
        static_cast<size_t>(n_batch) * seq_in_slot_bytes;
    const size_t per_batch_out_bytes =
        static_cast<size_t>(n_batch) *
        static_cast<size_t>(args.launch_chunks) *
        static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);
    const size_t n_batches =
        (n_chunks_host + static_cast<size_t>(n_batch) - 1u) /
        static_cast<size_t>(n_batch);

    trace("input bytes=" + std::to_string(input_buf.size()) +
          " bases=" + std::to_string(args.n_bases) +
          " chunks=" + std::to_string(n_chunks_host) +
          " batches=" + std::to_string(n_batches));

    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    auto xclbin = xrt::xclbin(args.xclbin);
    device.register_xclbin(xclbin);
    auto context = xrt::hw_context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, args.kernel);

    auto instr = read_instr_binary(args.instr);
    auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto *p_instr = bo_instr.map<uint32_t *>();
    std::memcpy(p_instr, instr.data(), instr.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto bo_seq_in = xrt::bo(device, per_batch_seq_in_bytes,
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_sparse_out = xrt::bo(device, per_batch_out_bytes,
                                 XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto *p_seq_in = bo_seq_in.map<uint8_t *>();
    auto *p_sparse_out = bo_sparse_out.map<uint8_t *>();

    std::vector<Record> all_records;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;
    int timed_dispatches = 0;
    uint32_t max_emit_observed = 0;
    const unsigned int opcode = 3;
    const int total_iters = args.warmup + args.iters;

    for (int it = 0; it < total_iters; ++it) {
        const bool keep = it >= args.warmup;
        if (keep) {
            all_records.clear();
            max_emit_observed = 0;
        }
        for (size_t bi = 0; bi < n_batches; ++bi) {
            std::memset(p_seq_in, 0, per_batch_seq_in_bytes);
            for (int slot = 0; slot < n_batch; ++slot) {
                const size_t ci = bi * static_cast<size_t>(n_batch) +
                                  static_cast<size_t>(slot);
                uint8_t *slot_base = p_seq_in +
                    static_cast<size_t>(slot) * seq_in_slot_bytes;
                if (ci >= n_chunks_host) continue;
                const ChunkPlan &c = chunks[ci];
                uint32_t actual_bytes_le = static_cast<uint32_t>(c.bytes);
                int32_t owned_start_offset_bases =
                    (ci == 0) ? 0 : (SEQ_IN_OVERLAP * 4 - 2);
                std::memcpy(slot_base, &actual_bytes_le, 4);
                std::memcpy(slot_base + 4, &owned_start_offset_bases, 4);
                std::memcpy(slot_base + HEADER_BYTES,
                            input_buf.data() + c.src_offset,
                            c.bytes);
            }
            bo_seq_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto run = kernel(
                opcode,
                bo_instr,
                static_cast<unsigned>(instr.size()),
                bo_seq_in,
                bo_sparse_out,
                static_cast<int>(per_batch_seq_in_bytes));
            run.wait();
            bo_sparse_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto t1 = std::chrono::high_resolution_clock::now();

            if (keep) {
                auto us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count();
                float fus = static_cast<float>(us);
                total_us += fus;
                min_us = std::min(min_us, fus);
                max_us = std::max(max_us, fus);
                timed_dispatches++;

                const size_t per_chunk_block_bytes =
                    static_cast<size_t>(args.launch_chunks) *
                    static_cast<size_t>(PARTIAL_OUT_BYTES_PADDED);
                for (int slot = 0; slot < n_batch; ++slot) {
                    const size_t ci = bi * static_cast<size_t>(n_batch) +
                                      static_cast<size_t>(slot);
                    if (ci >= n_chunks_host) break;
                    const ChunkPlan &c = chunks[ci];
                    const uint8_t *chunk_blob =
                        p_sparse_out +
                        static_cast<size_t>(slot) * per_chunk_block_bytes;
                    parse_chunk_tile0(
                        chunk_blob,
                        c.src_offset * 4u,
                        args.n_bases,
                        all_records,
                        max_emit_observed);
                }
            }
        }
    }

    if (max_emit_observed >= static_cast<uint32_t>(MC_MAX_EMIT_IDX)) {
        trace("WARN: max_emit_observed=" +
              std::to_string(max_emit_observed) +
              " hit cap MC_MAX_EMIT_IDX=" +
              std::to_string(MC_MAX_EMIT_IDX));
    }

    std::sort(all_records.begin(), all_records.end(), record_less);
    all_records.erase(
        std::unique(all_records.begin(), all_records.end(), record_equal),
        all_records.end());
    if (args.top > 0 && static_cast<size_t>(args.top) < all_records.size()) {
        all_records.resize(static_cast<size_t>(args.top));
    }

    if (args.output_format == "binary") emit_binary(args.output_path, all_records);
    else emit_tsv(args.output_path, all_records);

    float avg_us = timed_dispatches ? total_us / timed_dispatches : 0.0f;
    if (timed_dispatches == 0) min_us = 0.0f;
    std::cout << "PASS!\n";
    std::cout << "Avg NPU time: " << avg_us << "us.\n";
    std::cout << "Min NPU time: " << min_us << "us.\n";
    std::cout << "Max NPU time: " << max_us << "us.\n";
    std::cout << "Records: " << all_records.size() << "\n";
    return 0;
}
