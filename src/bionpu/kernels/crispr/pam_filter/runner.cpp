// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void trace(const std::string &msg) {
    std::cerr << "[pam_filter_runner] " << msg << std::endl;
}

// Pinned shape — must match pam_filter.py / tile_a_filter.cc.
constexpr int N_GUIDES = 128;
constexpr int SPACER_BYTES = 5;
constexpr int PAM_BYTES = 1;
constexpr int WINDOW_BYTES_IN = SPACER_BYTES + PAM_BYTES;  // 6
constexpr int N_WINDOWS = 4096;
constexpr int WINDOWS_PER_CHUNK = 64;
constexpr int N_CHUNKS = N_WINDOWS / WINDOWS_PER_CHUNK;
constexpr int EMIT_RECORD_BYTES = 8;
// fix: bumped 256 -> 1024 (worst chr22 sub-chunk has 508
// host hits; 256 silently dropped 1004 records). Stays in lockstep
// with tile_a_filter.cc + pam_filter.py + __init__.py.
constexpr int EMIT_SLOT_RECORDS = 1024;
constexpr int EMIT_SLOT_BYTES = EMIT_RECORD_BYTES * EMIT_SLOT_RECORDS;

constexpr int GUIDES_VOL = N_GUIDES * SPACER_BYTES;            // 640
constexpr int WINDOWS_IN_VOL = N_WINDOWS * WINDOW_BYTES_IN;    // 24576
constexpr int SPARSE_OUT_VOL = N_CHUNKS * EMIT_SLOT_BYTES;     // 524288

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

std::vector<uint8_t> read_blob(const std::string &path, size_t expected_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << " for reading\n";
        std::exit(2);
    }
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (static_cast<size_t>(sz) != expected_bytes) {
        std::cerr << "ERROR: " << path << " is " << sz << " bytes; expected "
                  << expected_bytes << "\n";
        std::exit(2);
    }
    std::vector<uint8_t> buf(expected_bytes);
    f.read(reinterpret_cast<char *>(buf.data()), expected_bytes);
    return buf;
}

void write_blob(const std::string &path, const uint8_t *data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << " for writing\n";
        std::exit(2);
    }
    f.write(reinterpret_cast<const char *>(data), bytes);
}

struct Args {
    std::string xclbin;
    std::string instr;
    std::string kernel = "MLIR_AIE";
    std::string guides_path;
    std::string windows_path;
    std::string out_path;
    std::string windows_batch_path;
    std::string out_batch_path;
    int chunks = 1;
    int launch_chunks = 1;
    int pipeline_depth = 8;
    int max_mm = 4;
    int iters = 1;
    int warmup = 0;
};

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "-x" || k == "--xclbin") a.xclbin = next();
        else if (k == "-i" || k == "--instr") a.instr = next();
        else if (k == "-k" || k == "--kernel") a.kernel = next();
        else if (k == "--guides") a.guides_path = next();
        else if (k == "--windows") a.windows_path = next();
        else if (k == "--out") a.out_path = next();
        else if (k == "--windows-batch") a.windows_batch_path = next();
        else if (k == "--out-batch") a.out_batch_path = next();
        else if (k == "--chunks") a.chunks = std::stoi(next());
        else if (k == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (k == "--pipeline-depth") a.pipeline_depth = std::stoi(next());
        else if (k == "--max-mm") a.max_mm = std::stoi(next());
        else if (k == "--iters") a.iters = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else throw std::runtime_error("unknown arg: " + k);
    }
    bool single = !a.windows_path.empty() || !a.out_path.empty();
    bool batch = !a.windows_batch_path.empty() || !a.out_batch_path.empty();
    if (single == batch)
        throw std::runtime_error(
            "choose exactly one of single mode (--windows/--out) or "
            "batch mode (--windows-batch/--out-batch/--chunks)");
    if (a.xclbin.empty() || a.instr.empty() || a.guides_path.empty())
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --guides <path>");
    if (single && (a.windows_path.empty() || a.out_path.empty()))
        throw std::runtime_error("single mode requires --windows and --out");
    if (batch && (a.windows_batch_path.empty() || a.out_batch_path.empty() ||
                  a.chunks <= 0))
        throw std::runtime_error(
            "batch mode requires --windows-batch, --out-batch, and "
            "--chunks > 0");
    if (a.pipeline_depth <= 0)
        throw std::runtime_error("--pipeline-depth must be > 0");
    if (a.launch_chunks <= 0)
        throw std::runtime_error("--launch-chunks must be > 0");
    if (single && a.launch_chunks != 1)
        throw std::runtime_error("single mode requires --launch-chunks 1");
    return a;
}

} // namespace

int main(int argc, char **argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }

    // Load inputs.
    trace("load inputs begin");
    auto guides = read_blob(args.guides_path, GUIDES_VOL);
    const bool batch_mode = !args.windows_batch_path.empty();
    const int n_chunks_host = batch_mode ? args.chunks : 1;
    const int launch_chunks = batch_mode ? args.launch_chunks : 1;
    const int n_launches = (n_chunks_host + launch_chunks - 1) / launch_chunks;
    const int windows_in_vol = launch_chunks * WINDOWS_IN_VOL;
    const int sparse_out_vol = launch_chunks * SPARSE_OUT_VOL;
    auto windows = read_blob(
        batch_mode ? args.windows_batch_path : args.windows_path,
        static_cast<size_t>(n_chunks_host) * WINDOWS_IN_VOL);
    std::vector<uint8_t> all_out(static_cast<size_t>(n_chunks_host) * SPARSE_OUT_VOL);
    auto instr_v = read_instr_binary(args.instr);
    trace("load inputs done instr_words=" + std::to_string(instr_v.size()) +
          " guides_bytes=" + std::to_string(guides.size()) +
          " windows_bytes=" + std::to_string(windows.size()) +
          " chunks=" + std::to_string(n_chunks_host) +
          " launch_chunks=" + std::to_string(launch_chunks) +
          " launches=" + std::to_string(n_launches));

    // XRT setup.
    trace("xrt device open begin");
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    trace("xrt device open done");

    trace("xclbin load begin path=" + args.xclbin);
    auto xclbin = xrt::xclbin(args.xclbin);
    trace("xclbin load done");

    trace("xclbin register begin");
    device.register_xclbin(xclbin);
    trace("xclbin register done");

    trace("hw_context create begin");
    xrt::hw_context context(device, xclbin.get_uuid());
    trace("hw_context create done");

    trace("kernel open begin name=" + args.kernel);
    auto kernel = xrt::kernel(context, args.kernel);
    trace("kernel open done");

    // Buffer objects. Argument indices follow the IRON-generated wrapper:
    // (opcode, instr, instr_size, guides, windows, sparse_out).
    trace("bo alloc begin");
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_guides = xrt::bo(device, GUIDES_VOL, XRT_BO_FLAGS_HOST_ONLY,
                              kernel.group_id(3));
    const int pipeline_depth = batch_mode
        ? std::min(args.pipeline_depth, n_chunks_host)
        : 1;
    std::vector<xrt::bo> bo_windows;
    std::vector<xrt::bo> bo_out;
    bo_windows.reserve(pipeline_depth);
    bo_out.reserve(pipeline_depth);
    for (int d = 0; d < pipeline_depth; d++) {
        bo_windows.emplace_back(device, windows_in_vol, XRT_BO_FLAGS_HOST_ONLY,
                                kernel.group_id(4));
        bo_out.emplace_back(device, sparse_out_vol, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(5));
    }
    trace("bo alloc done");

    trace("bo map/fill begin");
    auto *p_instr = bo_instr.map<uint32_t *>();
    std::memcpy(p_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    auto *p_guides = bo_guides.map<uint8_t *>();
    std::memcpy(p_guides, guides.data(), GUIDES_VOL);

    std::vector<uint8_t *> p_windows;
    std::vector<uint8_t *> p_out;
    p_windows.reserve(pipeline_depth);
    p_out.reserve(pipeline_depth);
    for (int d = 0; d < pipeline_depth; d++) {
        p_windows.push_back(bo_windows[d].map<uint8_t *>());
        p_out.push_back(bo_out[d].map<uint8_t *>());
    }
    trace("bo map/fill done");

    trace("bo sync to device begin");
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_guides.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    trace("bo sync to device done");

    unsigned int opcode = 3;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;

    int timed_runs = 0;
    if (batch_mode) {
        for (int base = 0; base < n_launches; base += pipeline_depth) {
            int block = std::min(pipeline_depth, n_launches - base);
            for (int j = 0; j < block; j++) {
                int launch = base + j;
                int first_chunk = launch * launch_chunks;
                int chunks_this_launch =
                    std::min(launch_chunks, n_chunks_host - first_chunk);
                std::memset(p_windows[j], 0, windows_in_vol);
                for (int k = 0; k < chunks_this_launch; k++) {
                    int c = first_chunk + k;
                    std::memcpy(
                        p_windows[j] + static_cast<size_t>(k) * WINDOWS_IN_VOL,
                        windows.data() + static_cast<size_t>(c) * WINDOWS_IN_VOL,
                        WINDOWS_IN_VOL);
                }
                bo_windows[j].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }

            std::vector<xrt::run> runs;
            runs.reserve(block);
            trace("block " + std::to_string(base) + " launch begin count=" +
                  std::to_string(block));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < block; j++) {
                runs.emplace_back(kernel(opcode, bo_instr, instr_v.size(), bo_guides,
                                         bo_windows[j], bo_out[j]));
            }
            trace("block " + std::to_string(base) + " launch returned; wait begin");
            for (auto &run : runs) run.wait();
            trace("block " + std::to_string(base) + " wait done");
            auto t1 = std::chrono::high_resolution_clock::now();

            float block_us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count();
            int public_chunks_in_block = 0;
            for (int j = 0; j < block; j++) {
                int launch = base + j;
                int first_chunk = launch * launch_chunks;
                public_chunks_in_block +=
                    std::min(launch_chunks, n_chunks_host - first_chunk);
            }
            float per_run_us = block_us / public_chunks_in_block;
            total_us += block_us;
            if (per_run_us < min_us) min_us = per_run_us;
            if (per_run_us > max_us) max_us = per_run_us;
            timed_runs += public_chunks_in_block;

            for (int j = 0; j < block; j++) {
                int launch = base + j;
                int first_chunk = launch * launch_chunks;
                int chunks_this_launch =
                    std::min(launch_chunks, n_chunks_host - first_chunk);
                bo_out[j].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                for (int k = 0; k < chunks_this_launch; k++) {
                    int c = first_chunk + k;
                    std::memcpy(
                        all_out.data() + static_cast<size_t>(c) * SPARSE_OUT_VOL,
                        p_out[j] + static_cast<size_t>(k) * SPARSE_OUT_VOL,
                        SPARSE_OUT_VOL);
                }
            }
        }
    } else {
        std::memcpy(p_windows[0], windows.data(), WINDOWS_IN_VOL);
        bo_windows[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int i = 0; i < args.warmup + args.iters; i++) {
            trace("iter " + std::to_string(i) + " launch begin");
            auto t0 = std::chrono::high_resolution_clock::now();
            auto run = kernel(opcode, bo_instr, instr_v.size(), bo_guides,
                              bo_windows[0], bo_out[0]);
            trace("iter " + std::to_string(i) + " launch returned; wait begin");
            run.wait();
            trace("iter " + std::to_string(i) + " wait done");
            auto t1 = std::chrono::high_resolution_clock::now();

            if (i < args.warmup) continue;
            float us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                           .count();
            total_us += us;
            if (us < min_us) min_us = us;
            if (us > max_us) max_us = us;
            timed_runs++;
        }

        bo_out[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(all_out.data(), p_out[0], SPARSE_OUT_VOL);
    }

    std::string out_path = batch_mode ? args.out_batch_path : args.out_path;
    trace("write output begin path=" + out_path);
    write_blob(out_path, all_out.data(), all_out.size());
    trace("write output done");

    float avg_us = (timed_runs > 0) ? (total_us / timed_runs) : 0.0f;
    std::cout << "Avg NPU time: " << avg_us << "us.\n";
    std::cout << "Min NPU time: " << min_us << "us.\n";
    std::cout << "Max NPU time: " << max_us << "us.\n";
    std::cout << "\nPASS!\n\n";
    return 0;
}
