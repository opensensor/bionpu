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

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Pinned shape — must match multitile_memtile.py / match_kernel.cc.
constexpr int N_GUIDES = 128;
constexpr int SPACER_BYTES = 5;
constexpr int N_WINDOWS = 4096;

constexpr int GUIDES_VOL = N_GUIDES * SPACER_BYTES;        // 640
constexpr int WINDOWS_VOL = N_WINDOWS * SPACER_BYTES;      // 20480
constexpr int OUT_VOL = N_WINDOWS * N_GUIDES;              // 524288

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
        else if (k == "--iters") a.iters = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.xclbin.empty() || a.instr.empty() || a.guides_path.empty() ||
        a.windows_path.empty() || a.out_path.empty())
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --guides <path> "
            "--windows <path> --out <path>");
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
    auto guides = read_blob(args.guides_path, GUIDES_VOL);
    auto windows = read_blob(args.windows_path, WINDOWS_VOL);
    auto instr_v = read_instr_binary(args.instr);

    // XRT setup.
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    auto xclbin = xrt::xclbin(args.xclbin);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, args.kernel);

    // Buffer objects. Argument indices follow the IRON-generated wrapper:
    // (opcode, instr, instr_size, guides, windows, out [, ctrlpkts, trace]).
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_guides = xrt::bo(device, GUIDES_VOL, XRT_BO_FLAGS_HOST_ONLY,
                              kernel.group_id(3));
    auto bo_windows = xrt::bo(device, WINDOWS_VOL, XRT_BO_FLAGS_HOST_ONLY,
                               kernel.group_id(4));
    auto bo_out = xrt::bo(device, OUT_VOL, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(5));

    // Map host pointers.
    auto *p_instr = bo_instr.map<uint32_t *>();
    std::memcpy(p_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    auto *p_guides = bo_guides.map<uint8_t *>();
    std::memcpy(p_guides, guides.data(), GUIDES_VOL);

    auto *p_windows = bo_windows.map<uint8_t *>();
    std::memcpy(p_windows, windows.data(), WINDOWS_VOL);

    auto *p_out = bo_out.map<uint8_t *>();
    std::memset(p_out, 0, OUT_VOL);

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_guides.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_windows.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run the kernel n_warmup + n_iter times; time only the post-warmup runs.
    unsigned int opcode = 3;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;

    for (int i = 0; i < args.warmup + args.iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_guides,
                          bo_windows, bo_out);
        run.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        if (i < args.warmup) continue;
        float us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                       .count();
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    write_blob(args.out_path, p_out, OUT_VOL);

    float avg_us = (args.iters > 0) ? (total_us / args.iters) : 0.0f;
    std::cout << "Avg NPU time: " << avg_us << "us.\n";
    std::cout << "Min NPU time: " << min_us << "us.\n";
    std::cout << "Max NPU time: " << max_us << "us.\n";
    std::cout << "\nPASS!\n\n";
    return 0;
}
