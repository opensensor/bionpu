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

constexpr int HIDDEN = 96;
constexpr int INPUT_DIM = 96;
constexpr int HALF_IN = INPUT_DIM / 2; // 48
constexpr int N_GATES = 4;
constexpr int WIH_GATE_LEN = HIDDEN * INPUT_DIM; // 9216
constexpr int WHH_GATE_LEN = HIDDEN * HIDDEN;    // 9216
constexpr int BIH_GATE_LEN = HIDDEN;             // 96
constexpr int BHH_GATE_LEN = HIDDEN;             // 96
constexpr int WEIGHT_HALF_LEN = HIDDEN * HALF_IN; // 4608
constexpr int BIAS_LEN = N_GATES * 2 * HIDDEN;   // 768

// Compact host-side WB buffer is per __init__.py's _pack_wb:
//   [W_ih_i (9216), W_hh_i (9216), W_ih_f, W_hh_f,
//    W_ih_g, W_hh_g, W_ih_o, W_hh_o,
//    b_ih_i (96), b_hh_i (96), b_ih_f, b_hh_f,
//    b_ih_g, b_hh_g, b_ih_o, b_hh_o]
// Total = 4 * (9216 + 9216) + 4 * (96 + 96) = 74496 FP32.
constexpr int COMPACT_WB_LEN =
    N_GATES * (WIH_GATE_LEN + WHH_GATE_LEN) +
    N_GATES * (BIH_GATE_LEN + BHH_GATE_LEN);

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

std::vector<float> read_floats(const std::string &path, size_t expected_len) {
  std::ifstream fh(path, std::ios::binary);
  if (!fh)
    throw std::runtime_error("failed to open binary file: " + path);
  fh.seekg(0, std::ios::end);
  size_t bytes = fh.tellg();
  fh.seekg(0, std::ios::beg);
  if (bytes != expected_len * sizeof(float))
    throw std::runtime_error("file " + path + " has " + std::to_string(bytes) +
                             " bytes; expected " +
                             std::to_string(expected_len * sizeof(float)));
  std::vector<float> out(expected_len);
  fh.read(reinterpret_cast<char *>(out.data()), bytes);
  return out;
}

void write_floats(const std::string &path, const float *data, size_t len) {
  std::ofstream fh(path, std::ios::binary);
  if (!fh)
    throw std::runtime_error("failed to open output file: " + path);
  fh.write(reinterpret_cast<const char *>(data), len * sizeof(float));
}

struct Args {
  std::string xclbin;
  std::string instr;
  std::string kernel = "MLIR_AIE";
  std::string input_path;
  std::string wb_path;
  std::string output_path;
  int seq = 334;
  int hidden = 96;
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
    else if (k == "--input") a.input_path = next();
    else if (k == "--wb") a.wb_path = next();
    else if (k == "--output") a.output_path = next();
    else if (k == "--seq") a.seq = std::stoi(next());
    else if (k == "--hidden") a.hidden = std::stoi(next());
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else throw std::runtime_error("unknown arg: " + k);
  }
  if (a.xclbin.empty() || a.instr.empty() || a.input_path.empty() ||
      a.wb_path.empty() || a.output_path.empty())
    throw std::runtime_error(
        "required: -x <xclbin> -i <instr> --input <path> --wb <path> "
        "--output <path>");
  if (a.hidden != HIDDEN)
    throw std::runtime_error("hidden must be 96 (pinned)");
  return a;
}

// Expand the compact host-side WB into the IRON per-timestep cycle.
// Each chunk has shape (BIAS_LEN + WEIGHT_HALF_LEN) FP32 = 5376 floats:
// the first BIAS_LEN floats hold the full bias slab (kernel reads on
// t==0,g==0 only); the remaining WEIGHT_HALF_LEN floats hold the
// half-gate weight slab.
struct ExpandedWB {
  std::vector<float> weight; // L * N_GATES * 4 chunks, each
                             // (BIAS_LEN + WEIGHT_HALF_LEN) floats.
};

constexpr int CHUNK_LEN = BIAS_LEN + WEIGHT_HALF_LEN; // 768 + 4608 = 5376

ExpandedWB expand_wb(const std::vector<float> &compact, int L) {
  // Compact layout (gate-major: g=0..3 then biases gate-major):
  //   [WIH_g0 (9216), WHH_g0 (9216),
  //    WIH_g1 (9216), WHH_g1 (9216),
  //    WIH_g2 (9216), WHH_g2 (9216),
  //    WIH_g3 (9216), WHH_g3 (9216),
  //    BIH_g0 (96), BHH_g0 (96),
  //    BIH_g1, BHH_g1, BIH_g2, BHH_g2, BIH_g3, BHH_g3]
  const float *wp = compact.data();
  const float *Wih[N_GATES];
  const float *Whh[N_GATES];
  size_t off = 0;
  for (int g = 0; g < N_GATES; ++g) {
    Wih[g] = wp + off; off += WIH_GATE_LEN;
    Whh[g] = wp + off; off += WHH_GATE_LEN;
  }
  // Biases: copy gate-major (b_ih_g, b_hh_g) into a single 768-float
  // slab to be replicated as a prefix on every chunk. The kernel only
  // needs them on the (t==0, g==0) call but we must include them on
  // every chunk to keep the IRON DMA pattern simple.
  std::vector<float> bias_slab(BIAS_LEN);
  for (int g = 0; g < N_GATES; ++g) {
    const float *b_ih = wp + off; off += BIH_GATE_LEN;
    const float *b_hh = wp + off; off += BHH_GATE_LEN;
    std::memcpy(bias_slab.data() + (g * 2) * HIDDEN, b_ih,
                HIDDEN * sizeof(float));
    std::memcpy(bias_slab.data() + (g * 2 + 1) * HIDDEN, b_hh,
                HIDDEN * sizeof(float));
  }

  ExpandedWB out;
  out.weight.resize(
      static_cast<size_t>(L) * N_GATES * 4 * CHUNK_LEN);

  auto emit_chunk = [&](const float *src_full, // (HIDDEN, INPUT_DIM)
                        float *dst,            // (CHUNK_LEN,)
                        int half_idx) {
    // Bias prefix
    std::memcpy(dst, bias_slab.data(), BIAS_LEN * sizeof(float));
    // Half-gate weight slab
    float *wdst = dst + BIAS_LEN;
    for (int oc = 0; oc < HIDDEN; ++oc) {
      const float *src_row = src_full + oc * INPUT_DIM + half_idx * HALF_IN;
      float *dst_row = wdst + oc * HALF_IN;
      std::memcpy(dst_row, src_row, HALF_IN * sizeof(float));
    }
  };

  size_t per_ts_chunks = N_GATES * 4;
  for (int t = 0; t < L; ++t) {
    for (int g = 0; g < N_GATES; ++g) {
      size_t base =
          (static_cast<size_t>(t) * per_ts_chunks + g * 4) * CHUNK_LEN;
      // Order matches IRON acquire order: ih_h0, ih_h1, hh_h0, hh_h1.
      emit_chunk(Wih[g], out.weight.data() + base + 0 * CHUNK_LEN, 0);
      emit_chunk(Wih[g], out.weight.data() + base + 1 * CHUNK_LEN, 1);
      emit_chunk(Whh[g], out.weight.data() + base + 2 * CHUNK_LEN, 0);
      emit_chunk(Whh[g], out.weight.data() + base + 3 * CHUNK_LEN, 1);
    }
  }

  return out;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse(argc, argv);

  const size_t IN_TOTAL = static_cast<size_t>(args.seq) * HIDDEN;
  const size_t OUT_TOTAL = IN_TOTAL;

  auto input = read_floats(args.input_path, IN_TOTAL);
  auto wb_compact = read_floats(args.wb_path, COMPACT_WB_LEN);
  auto instr_v = read_instr_binary(args.instr);

  // Expand compact wb into per-timestep cycle (with bias prefixed onto
  // every chunk). One large bo_weight buffer; no separate bias BO.
  ExpandedWB wb = expand_wb(wb_compact, args.seq);

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(args.xclbin);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, args.kernel);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  // Kernel arg order (matches lstm_cell.py runtime sequence):
  //   group_id 3 = input  (X)
  //   group_id 4 = weight (W) — expanded with bias prefixes
  //   group_id 5 = output (Y)
  auto bo_input = xrt::bo(device, IN_TOTAL * sizeof(float),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_weight = xrt::bo(device, wb.weight.size() * sizeof(float),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_output = xrt::bo(device, OUT_TOTAL * sizeof(float),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  // Trace BO (group_id 6 in the IRON layout): tiny placeholder.
  auto bo_trace =
      xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  float *bufInput = bo_input.map<float *>();
  std::memcpy(bufInput, input.data(), IN_TOTAL * sizeof(float));

  float *bufWeight = bo_weight.map<float *>();
  std::memcpy(bufWeight, wb.weight.data(), wb.weight.size() * sizeof(float));

  float *bufOutput = bo_output.map<float *>();
  std::memset(bufOutput, 0, OUT_TOTAL * sizeof(float));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  const unsigned int opcode = 3;
  unsigned num_iter = args.iters + args.warmup;
  float npu_time_total = 0;
  float npu_time_min = 9e9f;
  float npu_time_max = 0;

  for (unsigned it = 0; it < num_iter; ++it) {
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_weight,
                      bo_output, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (it < (unsigned)args.warmup)
      continue;

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  write_floats(args.output_path, bufOutput, OUT_TOTAL);

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / args.iters << "us."
            << std::endl;
  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  std::cout << "\nPASS!\n\n";
  return 0;
}
