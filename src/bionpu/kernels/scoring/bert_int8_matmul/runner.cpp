//===- runner.cpp ----------------------------------------------*- C++ -*-===//
//
// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// XRT host runner for `bert_int8_matmul_head` (M=47 K=768 N=2).
//
// File-backed I/O so the runner can be exercised from a Python
// driver without round-tripping through XRT bindings: the host
// writes x.bin / w.bin / s.bin to disk, calls this binary, and
// reads y.bin back. This matches the v0 approach used by
// crispr_pam_filter / dorado_fast_lstm_cell. Future iterations
// will run via `pyxrt` directly through `bionpu.dispatch.npu`.

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int M = 47;
constexpr int K = 768;
constexpr int N = 2;
constexpr size_t round4(size_t n) { return ((n + 3) / 4) * 4; }
constexpr size_t X_BYTES  = round4(static_cast<size_t>(M) * K);   // 36 096
constexpr size_t WS_BYTES = round4(
    static_cast<size_t>(N) * K +
    static_cast<size_t>(N + 1) * sizeof(float));                  //  1 548 → 1 548
constexpr size_t Y_BYTES  = round4(static_cast<size_t>(M) * N);   //     96

template <typename T>
std::vector<T> read_file(const std::string &path, size_t expected_bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open: " + path);
    }
    const auto actual_bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    // expected_bytes == 0 means "read whatever's there" (used for
    // insts.bin whose size is determined by the kernel build).
    const size_t to_read =
        (expected_bytes == 0) ? actual_bytes : expected_bytes;
    if (expected_bytes != 0 && actual_bytes != expected_bytes) {
        throw std::runtime_error(
            "size mismatch (" + std::to_string(actual_bytes) + "/" +
            std::to_string(expected_bytes) + " bytes) on " + path);
    }
    std::vector<T> out(to_read / sizeof(T));
    f.read(reinterpret_cast<char *>(out.data()), to_read);
    return out;
}

template <typename T>
void write_file(const std::string &path, const std::vector<T> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()),
            v.size() * sizeof(T));
}

void usage(const char *argv0) {
    std::cerr <<
        "Usage: " << argv0 <<
        " --xclbin FILE --insts FILE"
        " --x FILE --ws FILE --out FILE\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char **argv) {
    std::string xclbin_path, insts_path, x_path, ws_path, out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--xclbin" && i + 1 < argc) xclbin_path = argv[++i];
        else if (a == "--insts" && i + 1 < argc) insts_path = argv[++i];
        else if (a == "--x"  && i + 1 < argc) x_path  = argv[++i];
        else if (a == "--ws" && i + 1 < argc) ws_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else usage(argv[0]);
    }
    if (xclbin_path.empty() || insts_path.empty() || x_path.empty() ||
        ws_path.empty() || out_path.empty()) {
        usage(argv[0]);
    }

    // Load runtime artifacts.
    auto x_data  = read_file<int8_t>(x_path,  X_BYTES);
    auto ws_data = read_file<int8_t>(ws_path, WS_BYTES);
    auto insts   = read_file<uint32_t>(insts_path, 0);  // size-flexible

    // XRT device + xclbin load.
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    device.register_xclbin(xclbin);

    // Locate the entrypoint kernel (IRON places one default kernel
    // per program; aiecc names it `MLIR_AIE` by convention).
    xrt::hw_context ctx(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(ctx, "MLIR_AIE");

    // Allocate device-shared buffers. Group ids are per the IRON
    // Runtime sequence ordering: (insts, opcode, x, ws, y).
    auto bo_insts = xrt::bo(device, insts.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE,
                            kernel.group_id(1));
    auto bo_x  = xrt::bo(device, X_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_ws = xrt::bo(device, WS_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_y  = xrt::bo(device, Y_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    std::memcpy(bo_insts.map<uint32_t *>(), insts.data(),
                insts.size() * sizeof(uint32_t));
    std::memcpy(bo_x.map<int8_t *>(),  x_data.data(),  X_BYTES);
    std::memcpy(bo_ws.map<int8_t *>(), ws_data.data(), WS_BYTES);

    bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ws.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(/*opcode*/ 3,
                      bo_insts, insts.size(),
                      bo_x, bo_ws, bo_y);
    run.wait();

    bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<int8_t> y_out(Y_BYTES);
    std::memcpy(y_out.data(), bo_y.map<int8_t *>(), Y_BYTES);
    write_file(out_path, y_out);

    std::cerr << "bert_int8_matmul_head: wrote "
              << Y_BYTES << " bytes -> " << out_path << "\n";
    return 0;
}
