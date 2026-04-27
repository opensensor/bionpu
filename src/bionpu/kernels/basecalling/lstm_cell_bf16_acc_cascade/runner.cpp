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
constexpr int N_LAYERS = 5;
constexpr int WIH_GATE_LEN = HIDDEN * INPUT_DIM; // 9216
constexpr int WHH_GATE_LEN = HIDDEN * HIDDEN;    // 9216
constexpr int BIH_GATE_LEN = HIDDEN;             // 96
constexpr int BHH_GATE_LEN = HIDDEN;             // 96
constexpr int WEIGHT_HALF_LEN = HIDDEN * HALF_IN; // 4608
constexpr int BIAS_LEN = N_GATES * 2 * HIDDEN;   // 768

using bf16 = uint16_t;
constexpr int BYTES_PER_BF16 = sizeof(bf16);

constexpr int COMPACT_WB_LEN =
    N_GATES * (WIH_GATE_LEN + WHH_GATE_LEN) +
    N_GATES * (BIH_GATE_LEN + BHH_GATE_LEN);

constexpr int CHUNK_LEN = BIAS_LEN + WEIGHT_HALF_LEN; // 768 + 4608 = 5376

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

std::vector<bf16> read_bf16(const std::string &path, size_t expected_len) {
  std::ifstream fh(path, std::ios::binary);
  if (!fh)
    throw std::runtime_error("failed to open binary file: " + path);
  fh.seekg(0, std::ios::end);
  size_t bytes = fh.tellg();
  fh.seekg(0, std::ios::beg);
  if (bytes != expected_len * BYTES_PER_BF16)
    throw std::runtime_error("file " + path + " has " + std::to_string(bytes) +
                             " bytes; expected " +
                             std::to_string(expected_len * BYTES_PER_BF16));
  std::vector<bf16> out(expected_len);
  fh.read(reinterpret_cast<char *>(out.data()), bytes);
  return out;
}

void write_bf16(const std::string &path, const bf16 *data, size_t len) {
  std::ofstream fh(path, std::ios::binary);
  if (!fh)
    throw std::runtime_error("failed to open output file: " + path);
  fh.write(reinterpret_cast<const char *>(data), len * BYTES_PER_BF16);
}

struct Args {
  std::string xclbin;
  std::string instr;
  std::string kernel = "MLIR_AIE";
  std::string input_path;
  // closure: single consolidated weights buffer, packed
  // host-side into the layer-minor / chunk-major on-wire layout
  // documented at the top of this file.
  std::string weights_path;
  std::string output_path;
  int seq = 334;
  int hidden = 96;
  int iters = 1;
  int warmup = 0;
  bool hidden_state_only = false;
  bool hidden_state_weight = false;
  bool direct_layer_weights = false;
  bool direct_group_weights = false;
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
    else if (k == "--weights") a.weights_path = next();
    else if (k == "--output") a.output_path = next();
    else if (k == "--seq") a.seq = std::stoi(next());
    else if (k == "--hidden") a.hidden = std::stoi(next());
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else if (k == "--hidden-state-only") a.hidden_state_only = true;
    else if (k == "--hidden-state-weight") a.hidden_state_weight = true;
    else if (k == "--direct-layer-weights") a.direct_layer_weights = true;
    else if (k == "--direct-group-weights") a.direct_group_weights = true;
    else throw std::runtime_error("unknown arg: " + k);
  }
  if (a.xclbin.empty() || a.instr.empty() || a.input_path.empty() ||
      a.output_path.empty())
    throw std::runtime_error(
        "required: -x <xclbin> -i <instr> --input <path> --output <path> "
        "[--weights <path> | --hidden-state-only]");
  if (a.hidden_state_only && a.hidden_state_weight)
    throw std::runtime_error(
        "--hidden-state-only and --hidden-state-weight are mutually exclusive");
  if (!a.hidden_state_only && a.weights_path.empty())
    throw std::runtime_error(
        "production/hidden-state-weight mode requires --weights <path>");
  if (a.hidden_state_only && !a.weights_path.empty())
    throw std::runtime_error("--hidden-state-only must not pass --weights");
  if (a.direct_layer_weights && a.hidden_state_only)
    throw std::runtime_error(
        "--direct-layer-weights requires production weights");
  if (a.direct_group_weights && a.hidden_state_only)
    throw std::runtime_error(
        "--direct-group-weights requires production weights");
  if (a.direct_layer_weights && a.direct_group_weights)
    throw std::runtime_error(
        "--direct-layer-weights and --direct-group-weights are mutually exclusive");
  if (a.hidden != HIDDEN)
    throw std::runtime_error("hidden must be 96 (pinned)");
  return a;
}

// Drop unused per-layer-wb constants (read but never referenced after the
// host-side consolidation). Suppress a -Wunused-const-variable
// warning by referencing them in a static_assert that pins the consolidated
// buffer's expected size.
static_assert(COMPACT_WB_LEN ==
                  N_GATES * (WIH_GATE_LEN + WHH_GATE_LEN) +
                      N_GATES * (BIH_GATE_LEN + BHH_GATE_LEN),
              "COMPACT_WB_LEN out of sync with per-gate constants");

// closure: read the consolidated weights file from disk.
// The file MUST already be in the on-wire layer-minor / chunk-major
// layout (host-side ``interleave_wbs`` produces this). We do NOT
// expand or interleave on this side — that's the dispatch wrapper's
// job (see __init__.py::_pack_consolidated_wb_per_layer).
std::vector<bf16> read_consolidated_wb(const std::string &path, int L) {
  const size_t n_chunk_frames =
      static_cast<size_t>(L) * N_GATES * 4; // per layer
  const size_t expected_elts =
      n_chunk_frames * static_cast<size_t>(N_LAYERS) * CHUNK_LEN;
  return read_bf16(path, expected_elts);
}

std::vector<std::vector<bf16>>
split_consolidated_wb_per_layer(const std::vector<bf16> &weights, int L) {
  const size_t n_chunk_frames = static_cast<size_t>(L) * N_GATES * 4;
  std::vector<std::vector<bf16>> out(
      N_LAYERS, std::vector<bf16>(n_chunk_frames * CHUNK_LEN));
  for (size_t f = 0; f < n_chunk_frames; ++f) {
    for (int layer = 0; layer < N_LAYERS; ++layer) {
      const size_t src =
          (f * N_LAYERS + static_cast<size_t>(layer)) * CHUNK_LEN;
      const size_t dst = f * CHUNK_LEN;
      std::memcpy(out[layer].data() + dst, weights.data() + src,
                  CHUNK_LEN * BYTES_PER_BF16);
    }
  }
  return out;
}

std::vector<std::vector<bf16>>
split_consolidated_wb_groups(const std::vector<bf16> &weights, int L) {
  const size_t n_chunk_frames = static_cast<size_t>(L) * N_GATES * 4;
  std::vector<std::vector<bf16>> out;
  out.emplace_back(n_chunk_frames * 2 * CHUNK_LEN);
  out.emplace_back(n_chunk_frames * 2 * CHUNK_LEN);
  out.emplace_back(n_chunk_frames * CHUNK_LEN);

  for (size_t f = 0; f < n_chunk_frames; ++f) {
    const size_t frame_src = f * N_LAYERS * CHUNK_LEN;
    std::memcpy(out[0].data() + f * 2 * CHUNK_LEN,
                weights.data() + frame_src,
                CHUNK_LEN * BYTES_PER_BF16);
    std::memcpy(out[0].data() + f * 2 * CHUNK_LEN + CHUNK_LEN,
                weights.data() + frame_src + CHUNK_LEN,
                CHUNK_LEN * BYTES_PER_BF16);
    std::memcpy(out[1].data() + f * 2 * CHUNK_LEN,
                weights.data() + frame_src + 2 * CHUNK_LEN,
                CHUNK_LEN * BYTES_PER_BF16);
    std::memcpy(out[1].data() + f * 2 * CHUNK_LEN + CHUNK_LEN,
                weights.data() + frame_src + 3 * CHUNK_LEN,
                CHUNK_LEN * BYTES_PER_BF16);
    std::memcpy(out[2].data() + f * CHUNK_LEN,
                weights.data() + frame_src + 4 * CHUNK_LEN,
                CHUNK_LEN * BYTES_PER_BF16);
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse(argc, argv);

  const size_t IN_TOTAL = static_cast<size_t>(args.seq) * HIDDEN;
  const size_t OUT_TOTAL = IN_TOTAL;

  auto input = read_bf16(args.input_path, IN_TOTAL);
  auto instr_v = read_instr_binary(args.instr);

  // Production mode uses one consolidated weights buffer, pre-interleaved
  // by the dispatch wrapper. Hidden-state-only bisection deliberately
  // removes the weight BO and memtile split surface.
  std::vector<bf16> weights;
  if (args.hidden_state_weight)
    weights = read_bf16(args.weights_path, IN_TOTAL);
  else if (!args.hidden_state_only)
    weights = read_consolidated_wb(args.weights_path, args.seq);
  std::vector<std::vector<bf16>> layer_weights;
  if (args.direct_layer_weights)
    layer_weights = split_consolidated_wb_per_layer(weights, args.seq);
  std::vector<std::vector<bf16>> group_weights;
  if (args.direct_group_weights)
    group_weights = split_consolidated_wb_groups(weights, args.seq);

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(args.xclbin);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, args.kernel);

  // Buffer-object slot map:
  //   production runtime_sequence: input, weights_consolidated, output
  //     group ids: 1=instr, 3=input, 4=weights, 5=output, 6=trace
  //   hidden-state-only runtime_sequence: input, output
  //     group ids: 1=instr, 3=input, 4=output, 5=trace
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input = xrt::bo(device, IN_TOTAL * BYTES_PER_BF16,
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_weights = args.hidden_state_only
                        ? xrt::bo()
                        : xrt::bo(device, weights.size() * BYTES_PER_BF16,
                                  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  std::vector<xrt::bo> bo_layer_weights;
  if (args.direct_layer_weights) {
    bo_weights = xrt::bo();
    for (int layer = 0; layer < N_LAYERS; ++layer) {
      bo_layer_weights.emplace_back(
          device, layer_weights[layer].size() * BYTES_PER_BF16,
          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4 + layer));
    }
  }
  std::vector<xrt::bo> bo_group_weights;
  if (args.direct_group_weights) {
    bo_weights = xrt::bo();
    for (int group = 0; group < 3; ++group) {
      bo_group_weights.emplace_back(
          device, group_weights[group].size() * BYTES_PER_BF16,
          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4 + group));
    }
  }
  auto bo_output = xrt::bo(device, OUT_TOTAL * BYTES_PER_BF16,
                           XRT_BO_FLAGS_HOST_ONLY,
                           kernel.group_id(args.hidden_state_only ? 4 :
                                           args.direct_layer_weights ? 9 :
                                           args.direct_group_weights ? 7 : 5));
  auto bo_trace =
      args.direct_group_weights
          ? xrt::bo()
          : xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(args.hidden_state_only ? 5 :
                                    args.direct_layer_weights ? 10 : 6));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bf16 *bufInput = bo_input.map<bf16 *>();
  std::memcpy(bufInput, input.data(), IN_TOTAL * BYTES_PER_BF16);

  if (args.direct_layer_weights) {
    for (int layer = 0; layer < N_LAYERS; ++layer) {
      bf16 *bufLayerWeights = bo_layer_weights[layer].map<bf16 *>();
      std::memcpy(bufLayerWeights, layer_weights[layer].data(),
                  layer_weights[layer].size() * BYTES_PER_BF16);
    }
  } else if (args.direct_group_weights) {
    for (int group = 0; group < 3; ++group) {
      bf16 *bufGroupWeights = bo_group_weights[group].map<bf16 *>();
      std::memcpy(bufGroupWeights, group_weights[group].data(),
                  group_weights[group].size() * BYTES_PER_BF16);
    }
  } else if (!args.hidden_state_only) {
    bf16 *bufWeights = bo_weights.map<bf16 *>();
    std::memcpy(bufWeights, weights.data(), weights.size() * BYTES_PER_BF16);
  }

  bf16 *bufOutput = bo_output.map<bf16 *>();
  std::memset(bufOutput, 0, OUT_TOTAL * BYTES_PER_BF16);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (args.direct_layer_weights) {
    for (auto &bo : bo_layer_weights)
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  } else if (args.direct_group_weights) {
    for (auto &bo : bo_group_weights)
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  } else if (!args.hidden_state_only) {
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  const unsigned int opcode = 3;
  unsigned num_iter = args.iters + args.warmup;
  float npu_time_total = 0;
  float npu_time_min = 9e9f;
  float npu_time_max = 0;

  for (unsigned it = 0; it < num_iter; ++it) {
    auto start = std::chrono::high_resolution_clock::now();
    // Production has 4 user BOs (input, weights, output, trace).
    // Hidden-state-only has 3 user BOs (input, output, trace).
    auto run = args.hidden_state_only
                   ? kernel(opcode, bo_instr, instr_v.size(),
                            bo_input, bo_output, bo_trace)
                   : args.direct_layer_weights
                         ? kernel(opcode, bo_instr, instr_v.size(),
                                  bo_input,
                                  bo_layer_weights[0], bo_layer_weights[1],
                                  bo_layer_weights[2], bo_layer_weights[3],
                                  bo_layer_weights[4], bo_output, bo_trace)
                         : args.direct_group_weights
                               ? kernel(opcode, bo_instr, instr_v.size(),
                                        bo_input,
                                        bo_group_weights[0],
                                        bo_group_weights[1],
                                        bo_group_weights[2],
                                        bo_output)
                         : kernel(opcode, bo_instr, instr_v.size(),
                                  bo_input, bo_weights, bo_output, bo_trace);
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

  write_bf16(args.output_path, bufOutput, OUT_TOTAL);

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
