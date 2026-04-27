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
constexpr int OUT_DIM = 256;
constexpr int OC_GROUP_SIZE = 64;
constexpr int N_OC_GROUPS = OUT_DIM / OC_GROUP_SIZE;
constexpr int WB_PER_GROUP = OC_GROUP_SIZE * HIDDEN;

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
  std::string xclbin, instr, kernel = "MLIR_AIE";
  std::string input_path, wb_path, output_path;
  int seq = 334;
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
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else throw std::runtime_error("unknown arg: " + k);
  }
  if (a.xclbin.empty() || a.instr.empty() || a.input_path.empty() ||
      a.wb_path.empty() || a.output_path.empty())
    throw std::runtime_error(
        "required: -x <xclbin> -i <instr> --input <path> --wb <path> "
        "--output <path>");
  return a;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse(argc, argv);

  const size_t IN_TOTAL = static_cast<size_t>(args.seq) * HIDDEN;
  const size_t WB_TOTAL = static_cast<size_t>(args.seq) * N_OC_GROUPS * WB_PER_GROUP;
  const size_t OUT_TOTAL = static_cast<size_t>(args.seq) * OUT_DIM;

  auto input = read_floats(args.input_path, IN_TOTAL);
  auto wb = read_floats(args.wb_path, WB_TOTAL);
  auto instr_v = read_instr_binary(args.instr);

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(args.xclbin);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, args.kernel);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input = xrt::bo(device, IN_TOTAL * sizeof(float),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_weight = xrt::bo(device, WB_TOTAL * sizeof(float),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_output = xrt::bo(device, OUT_TOTAL * sizeof(float),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_trace = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(6));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  float *bufInput = bo_input.map<float *>();
  std::memcpy(bufInput, input.data(), IN_TOTAL * sizeof(float));

  float *bufWeight = bo_weight.map<float *>();
  std::memcpy(bufWeight, wb.data(), WB_TOTAL * sizeof(float));

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
