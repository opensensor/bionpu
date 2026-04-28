//===- runner.cpp ----------------------------------------------*- C++ -*-===//
//
// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// XRT host runner for bert_int8_matmul (head + qkvo variants).
//
// File-backed I/O so the runner can be exercised from a Python
// driver without round-tripping through XRT bindings: the host
// writes input bins to disk, calls this binary, and reads the output
// bin back. This matches the v0 approach used by crispr_pam_filter /
// dorado_fast_lstm_cell. Future iterations will run via `pyxrt`
// directly through `bionpu.dispatch.npu`.
//
// Two invocation forms:
//
//   head variant (M=47 K=768 N=2):
//     ./bert_int8_matmul --variant head \
//         --xclbin FILE --insts FILE \
//         --x FILE --ws FILE --out FILE
//
//   qkvo variant (M=47 K=768 N=768):
//     ./bert_int8_matmul --variant qkvo \
//         --xclbin FILE --insts FILE \
//         --xs FILE --w FILE --out FILE
//
// The qkvo `xs` file is the chunk-major packed (x + scales) blob that
// the IRON topology streams directly into the xs ObjectFifo; see
// bert_int8_matmul.py emit_mlir_qkvo for the layout.

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
constexpr size_t round4(size_t n) { return ((n + 3) / 4) * 4; }

// ─── head variant sizes (N=2) ───
constexpr int    HEAD_N         = 2;
constexpr size_t HEAD_X_BYTES   = round4(static_cast<size_t>(M) * K);                                            // 36 096
constexpr size_t HEAD_WS_BYTES  = round4(static_cast<size_t>(HEAD_N) * K + static_cast<size_t>(HEAD_N + 1) * 4); //  1 548
constexpr size_t HEAD_Y_BYTES   = round4(static_cast<size_t>(M) * HEAD_N);                                       //     96

// ─── qkvo variant sizes (N=768, K_CHUNK=64, K_CHUNKS=12) ───
constexpr int    QKVO_N         = 768;
constexpr int    QKVO_K_CHUNK   = 64;
constexpr int    QKVO_K_CHUNKS  = K / QKVO_K_CHUNK;                                                              // 12
// xs_chunk = M*K_CHUNK + (N+1)*4 = 47*64 + 769*4 = 3008+3076 = 6084 bytes
constexpr size_t QKVO_XS_CHUNK  = round4(static_cast<size_t>(M) * QKVO_K_CHUNK + static_cast<size_t>(QKVO_N + 1) * 4);
constexpr size_t QKVO_XS_BYTES  = round4(static_cast<size_t>(QKVO_K_CHUNKS) * QKVO_XS_CHUNK);                    // 73,008
constexpr size_t QKVO_W_BYTES   = round4(static_cast<size_t>(QKVO_K_CHUNKS) * QKVO_N * QKVO_K_CHUNK);            // 589,824 (= N*K)
constexpr size_t QKVO_Y_BYTES   = round4(static_cast<size_t>(M) * QKVO_N);                                       // 36,096

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
        "Usage:\n"
        "  " << argv0 << " --variant head"
        " --xclbin FILE --insts FILE --x FILE --ws FILE --out FILE\n"
        "  " << argv0 << " --variant qkvo"
        " --xclbin FILE --insts FILE --xs FILE --w FILE --out FILE\n";
    std::exit(1);
}

int run_head(const std::string &xclbin_path,
             const std::string &insts_path,
             const std::string &x_path,
             const std::string &ws_path,
             const std::string &out_path) {
    auto x_data  = read_file<int8_t>(x_path,  HEAD_X_BYTES);
    auto ws_data = read_file<int8_t>(ws_path, HEAD_WS_BYTES);
    auto insts   = read_file<uint32_t>(insts_path, 0);

    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    device.register_xclbin(xclbin);

    xrt::hw_context ctx(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(ctx, "MLIR_AIE");

    auto bo_insts = xrt::bo(device, insts.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE,
                            kernel.group_id(1));
    auto bo_x  = xrt::bo(device, HEAD_X_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_ws = xrt::bo(device, HEAD_WS_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_y  = xrt::bo(device, HEAD_Y_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    std::memcpy(bo_insts.map<uint32_t *>(), insts.data(),
                insts.size() * sizeof(uint32_t));
    std::memcpy(bo_x.map<int8_t *>(),  x_data.data(),  HEAD_X_BYTES);
    std::memcpy(bo_ws.map<int8_t *>(), ws_data.data(), HEAD_WS_BYTES);

    bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ws.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(/*opcode*/ 3,
                      bo_insts, insts.size(),
                      bo_x, bo_ws, bo_y);
    run.wait();

    bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<int8_t> y_out(HEAD_Y_BYTES);
    std::memcpy(y_out.data(), bo_y.map<int8_t *>(), HEAD_Y_BYTES);
    write_file(out_path, y_out);

    std::cerr << "bert_int8_matmul_head: wrote "
              << HEAD_Y_BYTES << " bytes -> " << out_path << "\n";
    return 0;
}

int run_qkvo(const std::string &xclbin_path,
             const std::string &insts_path,
             const std::string &xs_path,
             const std::string &w_path,
             const std::string &out_path) {
    auto xs_data = read_file<int8_t>(xs_path, QKVO_XS_BYTES);
    auto w_data  = read_file<int8_t>(w_path,  QKVO_W_BYTES);
    auto insts   = read_file<uint32_t>(insts_path, 0);

    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    device.register_xclbin(xclbin);

    xrt::hw_context ctx(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(ctx, "MLIR_AIE");

    // qkvo runtime sequence: (insts, opcode, xs_total, w_total, y_total)
    // Group ids: 1=insts, 3=xs, 4=w, 5=y.
    auto bo_insts = xrt::bo(device, insts.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE,
                            kernel.group_id(1));
    auto bo_xs = xrt::bo(device, QKVO_XS_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_w  = xrt::bo(device, QKVO_W_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_y  = xrt::bo(device, QKVO_Y_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    std::memcpy(bo_insts.map<uint32_t *>(), insts.data(),
                insts.size() * sizeof(uint32_t));
    std::memcpy(bo_xs.map<int8_t *>(), xs_data.data(), QKVO_XS_BYTES);
    std::memcpy(bo_w.map<int8_t *>(),  w_data.data(),  QKVO_W_BYTES);

    bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_xs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(/*opcode*/ 3,
                      bo_insts, insts.size(),
                      bo_xs, bo_w, bo_y);
    run.wait();

    bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<int8_t> y_out(QKVO_Y_BYTES);
    std::memcpy(y_out.data(), bo_y.map<int8_t *>(), QKVO_Y_BYTES);
    write_file(out_path, y_out);

    std::cerr << "bert_int8_matmul_qkvo: wrote "
              << QKVO_Y_BYTES << " bytes -> " << out_path << "\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    std::string variant = "head";
    std::string xclbin_path, insts_path, x_path, ws_path, xs_path, w_path, out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--variant" && i + 1 < argc) variant     = argv[++i];
        else if (a == "--xclbin"  && i + 1 < argc) xclbin_path = argv[++i];
        else if (a == "--insts"   && i + 1 < argc) insts_path  = argv[++i];
        else if (a == "--x"       && i + 1 < argc) x_path      = argv[++i];
        else if (a == "--ws"      && i + 1 < argc) ws_path     = argv[++i];
        else if (a == "--xs"      && i + 1 < argc) xs_path     = argv[++i];
        else if (a == "--w"       && i + 1 < argc) w_path      = argv[++i];
        else if (a == "--out"     && i + 1 < argc) out_path    = argv[++i];
        else usage(argv[0]);
    }
    if (xclbin_path.empty() || insts_path.empty() || out_path.empty()) {
        usage(argv[0]);
    }

    if (variant == "head") {
        if (x_path.empty() || ws_path.empty()) usage(argv[0]);
        return run_head(xclbin_path, insts_path, x_path, ws_path, out_path);
    }
    if (variant == "qkvo") {
        if (xs_path.empty() || w_path.empty()) usage(argv[0]);
        return run_qkvo(xclbin_path, insts_path, xs_path, w_path, out_path);
    }
    std::cerr << "unknown variant: " << variant << "\n";
    usage(argv[0]);
    return 1;
}
