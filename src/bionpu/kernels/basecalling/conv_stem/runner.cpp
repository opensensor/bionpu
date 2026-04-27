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

constexpr int CONV_K = 5;
constexpr int CONV_OUT_CH = 16;
constexpr int PAD = 2;
constexpr int WB_LEN = CONV_OUT_CH * CONV_K + CONV_OUT_CH; // 96 floats

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
  std::string signal_path;
  std::string wb_path;
  std::string output_path;
  int time = 2000;
  int chunk = 200;
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
    if (k == "-x" || k == "--xclbin")
      a.xclbin = next();
    else if (k == "-i" || k == "--instr")
      a.instr = next();
    else if (k == "-k" || k == "--kernel")
      a.kernel = next();
    else if (k == "--signal")
      a.signal_path = next();
    else if (k == "--wb")
      a.wb_path = next();
    else if (k == "--output")
      a.output_path = next();
    else if (k == "--time")
      a.time = std::stoi(next());
    else if (k == "--chunk")
      a.chunk = std::stoi(next());
    else if (k == "--iters")
      a.iters = std::stoi(next());
    else if (k == "--warmup")
      a.warmup = std::stoi(next());
    else
      throw std::runtime_error("unknown arg: " + k);
  }
  if (a.xclbin.empty() || a.instr.empty() || a.signal_path.empty() ||
      a.wb_path.empty() || a.output_path.empty())
    throw std::runtime_error(
        "required: -x <xclbin> -i <instr> --signal <path> "
        "--wb <path> --output <path>");
  if (a.time % a.chunk != 0)
    throw std::runtime_error("time must be a multiple of chunk");
  return a;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse(argc, argv);

  const int N = args.time / args.chunk;
  const int chunk_in_len = args.chunk + 2 * PAD;
  const size_t SIG_TOTAL = static_cast<size_t>(N) * chunk_in_len;
  const size_t OUT_TOTAL =
      static_cast<size_t>(N) * CONV_OUT_CH * args.chunk;

  // Load inputs
  auto signal = read_floats(args.signal_path, SIG_TOTAL);
  auto wb = read_floats(args.wb_path, WB_LEN);
  auto instr_v = read_instr_binary(args.instr);

  // XRT setup
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(args.xclbin);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, args.kernel);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_signal = xrt::bo(device, SIG_TOTAL * sizeof(float),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_wb = xrt::bo(device, WB_LEN * sizeof(float),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_output = xrt::bo(device, OUT_TOTAL * sizeof(float),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Dummy ctrlpkt + trace buffers (kernel signature has 7 args after opcode).
  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace =
      xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Map host pointers
  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  float *bufSignal = bo_signal.map<float *>();
  std::memcpy(bufSignal, signal.data(), SIG_TOTAL * sizeof(float));

  float *bufWb = bo_wb.map<float *>();
  std::memcpy(bufWb, wb.data(), WB_LEN * sizeof(float));

  float *bufOutput = bo_output.map<float *>();
  std::memset(bufOutput, 0, OUT_TOTAL * sizeof(float));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_signal.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_wb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  const unsigned int opcode = 3;
  unsigned num_iter = args.iters + args.warmup;
  float npu_time_total = 0;
  float npu_time_min = 9e9f;
  float npu_time_max = 0;

  for (unsigned it = 0; it < num_iter; ++it) {
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_signal, bo_wb,
                      bo_output, bo_ctrlpkts, bo_trace);
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

  // Persist output
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
