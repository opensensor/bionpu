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
//
// Per state/kmer_count_interface_contract.md (T1) — symbols, ObjectFifo
// names, constants, the streaming chunk-with-overlap dispatch protocol,
// the emit-on-evict overflow record layout, and the host-side dedup-by-
// canonical re-aggregation pass are pinned there. This runner.cpp is
// the T7 deliverable: chunked dispatch with per-k overlap, host-side
// re-aggregation, and Jellyfish-FASTA output.
//
// Mirrors:
//  - bionpu/kernels/crispr/pam_filter/runner.cpp (chunked dispatch,
//    sparse-emit drain, PASS! marker, Avg/Min/Max NPU time lines)
//  - bionpu/kernels/basecalling/linear_projection/runner.cpp (simpler
//    single-launch CLI shape; parser regex format).

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
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Pinned constants (mirror kmer_count_constants.h byte-equal). T7's
// runner cannot include the AIE-side header directly because it pulls
// AIE-specific sysinclude paths via Peano's libcxx; we duplicate the
// numerics here and static_assert the math against the contract.
// ---------------------------------------------------------------------

constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;

// Per-k overlap = ceil((k-1)/4) bytes. Pinned per T1 contract.
constexpr int32_t SEQ_IN_OVERLAP_K15 = 4;   // ceil((15-1)/4) = 4
constexpr int32_t SEQ_IN_OVERLAP_K21 = 5;   // ceil((21-1)/4) = 5
constexpr int32_t SEQ_IN_OVERLAP_K31 = 8;   // ceil((31-1)/4) = 8

// Catch silent contract drift at compile time. The integer rounding
// rule is ((k-1) + 3) / 4 == ceil((k-1)/4) for k >= 1.
static_assert(((15 - 1) + 3) / 4 == SEQ_IN_OVERLAP_K15,
              "k=15 overlap must be 4 bytes per T1 contract");
static_assert(((21 - 1) + 3) / 4 == SEQ_IN_OVERLAP_K21,
              "k=21 overlap must be 5 bytes per T1 contract");
static_assert(((31 - 1) + 3) / 4 == SEQ_IN_OVERLAP_K31,
              "k=31 overlap must be 8 bytes per T1 contract");

// EmitRecord layout (16 B). Mirrors kmer_count_constants.h.
struct EmitRecord {
    uint64_t canonical;
    uint32_t count;
    uint32_t flags;
};

static_assert(sizeof(EmitRecord) == 16, "EmitRecord must be 16 bytes packed");

constexpr uint32_t EVICT_FLAG = 1u << 0;
constexpr int32_t EMIT_RECORD_BYTES = 16;
constexpr int32_t EMIT_SLOT_RECORDS = 1024;
constexpr int32_t EMIT_SLOT_BYTES = EMIT_SLOT_RECORDS * EMIT_RECORD_BYTES;

static_assert(EMIT_SLOT_BYTES == 16384,
              "EMIT_SLOT_BYTES must be 16 KiB per T1 contract");

// Per-k canonical-bit-width mask (apply when decoding back to ACGT).
constexpr uint64_t KMER_MASK_K15 = (1ULL << 30) - 1ULL;
constexpr uint64_t KMER_MASK_K21 = (1ULL << 42) - 1ULL;
constexpr uint64_t KMER_MASK_K31 = (1ULL << 62) - 1ULL;

constexpr int32_t MAX_TILES = 8;

// ---------------------------------------------------------------------
// Per-k overlap selector.
// ---------------------------------------------------------------------
int overlap_bytes_for_k(int k) {
    switch (k) {
        case 15: return SEQ_IN_OVERLAP_K15;
        case 21: return SEQ_IN_OVERLAP_K21;
        case 31: return SEQ_IN_OVERLAP_K31;
        default:
            throw std::runtime_error(
                "unsupported k=" + std::to_string(k) +
                " (supported: 15, 21, 31 per T1 contract)");
    }
}

uint64_t kmer_mask_for_k(int k) {
    switch (k) {
        case 15: return KMER_MASK_K15;
        case 21: return KMER_MASK_K21;
        case 31: return KMER_MASK_K31;
        default:
            throw std::runtime_error(
                "unsupported k=" + std::to_string(k) +
                " (supported: 15, 21, 31 per T1 contract)");
    }
}

// ---------------------------------------------------------------------
// I/O helpers.
// ---------------------------------------------------------------------
void trace(const std::string &msg) {
    std::cerr << "[kmer_count_runner] " << msg << std::endl;
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

// ---------------------------------------------------------------------
// 2-bit canonical -> ACGT decoder. Inverse of T2's pack_dna_2bit /
// matches T1's MSB-first wire format. Decodes the lowest 2*k bits of
// `canonical` to a length-k ASCII string. Bit pair (k-1) is the MSB
// (most significant 2 bits within the masked region) which corresponds
// to the FIRST base of the k-mer (consistent with the rolling kernel
// shift order: forward = (forward << 2) | new_base, so the most
// recently shifted-in base is the LSB and the oldest base is at the
// top).
// ---------------------------------------------------------------------
char base_of_2bit(uint64_t code) {
    switch (static_cast<unsigned>(code & 0x3ULL)) {
        case 0: return 'A';
        case 1: return 'C';
        case 2: return 'G';
        case 3: return 'T';
    }
    return 'N';  // unreachable
}

std::string decode_canonical_to_acgt(uint64_t canonical, int k) {
    // Mask off any high garbage above the canonical bit-width before
    // decoding, defensive against corrupt records.
    uint64_t masked = canonical & kmer_mask_for_k(k);
    std::string out;
    out.resize(static_cast<size_t>(k));
    // Position 0 = oldest (top 2 bits of the masked value);
    // Position k-1 = newest (lowest 2 bits).
    for (int pos = 0; pos < k; ++pos) {
        int shift = 2 * (k - 1 - pos);
        uint64_t code = (masked >> shift) & 0x3ULL;
        out[static_cast<size_t>(pos)] = base_of_2bit(code);
    }
    return out;
}

// ---------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------
struct Args {
    std::string xclbin;
    std::string instr;
    std::string kernel = "MLIR_AIE";
    std::string input_path;
    std::string output_path;
    int k = 21;
    int top = 1000;
    int threshold = 1;
    int launch_chunks = 4;
    int iters = 1;
    int warmup = 0;
};

void print_usage(std::ostream &os) {
    os <<
"Usage: kmer_count_runner [options]\n"
"\n"
"Required:\n"
"  -x, --xclbin <path>         xclbin file (final.xclbin)\n"
"  -i, --instr <path>          NPU instructions binary (insts.bin)\n"
"  --input <packed_2bit>       packed-2-bit input (.2bit.bin)\n"
"  --output <jf_fasta>         Jellyfish-FASTA output ('>count\\nkmer\\n')\n"
"  --k {15,21,31}              k-mer length\n"
"\n"
"Optional:\n"
"  -k, --kernel <name>         kernel name (default MLIR_AIE)\n"
"  --top <N>                   top-N records (default 1000; 0 = all)\n"
"  --launch-chunks {1,2,4,8}   tile fan-out (default 4)\n"
"  --iters <N>                 timed iterations (default 1)\n"
"  --warmup <N>                untimed warmup iters (default 0)\n"
"  --threshold <int>           min count to emit (default 1)\n"
"\n"
"All 9 args (--xclbin/--instr/--kernel/--input/--output/--k/--top/\n"
"--launch-chunks/--iters/--warmup) parsed per T1 contract.\n";
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
        else if (key == "--output") a.output_path = next();
        else if (key == "--k") a.k = std::stoi(next());
        else if (key == "--top") a.top = std::stoi(next());
        else if (key == "--threshold") a.threshold = std::stoi(next());
        else if (key == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (key == "--iters") a.iters = std::stoi(next());
        else if (key == "--warmup") a.warmup = std::stoi(next());
        else throw std::runtime_error("unknown arg: " + key);
    }
    if (a.xclbin.empty() || a.instr.empty() || a.input_path.empty() ||
        a.output_path.empty())
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --input <path> "
            "--output <path> --k {15,21,31}");
    if (a.k != 15 && a.k != 21 && a.k != 31)
        throw std::runtime_error(
            "--k must be one of {15, 21, 31} per T1 contract");
    if (a.launch_chunks != 1 && a.launch_chunks != 2 &&
        a.launch_chunks != 4 && a.launch_chunks != 8)
        throw std::runtime_error(
            "--launch-chunks must be one of {1, 2, 4, 8} per T1 contract");
    if (a.iters <= 0)
        throw std::runtime_error("--iters must be > 0");
    if (a.warmup < 0)
        throw std::runtime_error("--warmup must be >= 0");
    if (a.top < 0)
        throw std::runtime_error("--top must be >= 0 (0 = all-above-threshold)");
    if (a.threshold < 0)
        throw std::runtime_error("--threshold must be >= 0");
    return a;
}

// ---------------------------------------------------------------------
// Streaming chunk-with-overlap dispatch.
//
// Per T1 contract: input is dispatched in SEQ_IN_CHUNK_BYTES_BASE
// (=4096) byte chunks with overlap_bytes (per-k) of overlap between
// consecutive chunks. The seq_in ObjectFifo carries
// (chunk_bytes + overlap_bytes) per element. Without the overlap,
// k-mers spanning chunk boundaries are lost (correctness fail vs
// Jellyfish at T14).
//
// chunk i covers source bytes [i*4096, i*4096 + 4096 + overlap),
// clamped to the input length on the last chunk. The kernel uses
// n_input_bytes to bound its rolling loop so the last (possibly
// short) chunk is correct.
// ---------------------------------------------------------------------
struct ChunkPlan {
    size_t src_offset;     // first byte of this chunk in the source buffer
    size_t bytes;          // bytes copied into the BO for this chunk
                            // (<= 4096 + overlap; smaller on the final chunk)
};

std::vector<ChunkPlan> plan_chunks(size_t input_bytes, int overlap_bytes) {
    std::vector<ChunkPlan> plan;
    if (input_bytes == 0) return plan;
    const size_t base = static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE);
    const size_t ov = static_cast<size_t>(overlap_bytes);
    size_t off = 0;
    while (off < input_bytes) {
        size_t end = std::min(off + base + ov, input_bytes);
        ChunkPlan p;
        p.src_offset = off;
        p.bytes = end - off;
        plan.push_back(p);
        if (end >= input_bytes) break;
        off += base;  // advance by chunk_base; the overlap region is
                       // re-read by the next chunk to preserve k-mers
                       // straddling the boundary.
    }
    return plan;
}

// ---------------------------------------------------------------------
// Host-side overflow re-aggregation.
//
// Per T1 contract, the output BO carries:
//   (a) regular sparse-emit records (flags == 0, length-prefixed
//       within each EMIT_SLOT)
//   (b) evicted-overflow records (flags & EVICT_FLAG)
// Both are 16 bytes (canonical_u64, count_u32, flags_u32). The host
// deduplicates by canonical_u64 across the entire output BO,
// summing count_u32 regardless of EVICT_FLAG. This is REQUIRED for
// T14 byte-equal-vs-Jellyfish on chr22.
//
// `out_blob` is the entire flattened post-sync output BO (size =
// n_chunks * launch_chunks * EMIT_SLOT_BYTES, or whatever the
// runner allocated). We scan EMIT_RECORD_BYTES at a stride and
// accept records whose `count > 0` (this matches the kernel's
// convention: zero-count slots are skipped on emit). Records that
// happen to land in the all-zero post-end-of-stream tail are
// filtered out by `count == 0`.
// ---------------------------------------------------------------------
std::unordered_map<uint64_t, uint64_t>
reaggregate_records(const std::vector<uint8_t> &out_blob) {
    std::unordered_map<uint64_t, uint64_t> merged;
    if (out_blob.size() < sizeof(EmitRecord)) return merged;
    const size_t n_records = out_blob.size() / sizeof(EmitRecord);
    const EmitRecord *recs =
        reinterpret_cast<const EmitRecord *>(out_blob.data());
    for (size_t i = 0; i < n_records; ++i) {
        const EmitRecord &r = recs[i];
        if (r.count == 0) continue;  // empty/uninitialized slot
        // Sum counts across duplicates regardless of EVICT_FLAG.
        merged[r.canonical] += static_cast<uint64_t>(r.count);
    }
    return merged;
}

// ---------------------------------------------------------------------
// Sort + filter + emit Jellyfish-FASTA.
//
// Output format per T1 / PRD §3.2 / Q5: '>count\nkmer\n' per record,
// with kmer in ACGT decoded MSB-first from the 2-bit canonical uint64.
// Sort key: (count desc, canonical asc).
// ---------------------------------------------------------------------
struct FinalRecord {
    uint64_t canonical;
    uint64_t count;
};

void emit_jellyfish_fasta(const std::string &path,
                          const std::unordered_map<uint64_t, uint64_t> &merged,
                          int k, int top, int threshold) {
    std::vector<FinalRecord> v;
    v.reserve(merged.size());
    for (const auto &kv : merged) {
        if (kv.second < static_cast<uint64_t>(threshold)) continue;
        v.push_back({kv.first, kv.second});
    }
    std::sort(v.begin(), v.end(), [](const FinalRecord &a, const FinalRecord &b) {
        if (a.count != b.count) return a.count > b.count;
        return a.canonical < b.canonical;
    });
    if (top > 0 && static_cast<size_t>(top) < v.size()) {
        v.resize(static_cast<size_t>(top));
    }
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("cannot open output for write: " + path);
    for (const auto &r : v) {
        f << '>' << r.count << '\n'
          << decode_canonical_to_acgt(r.canonical, k) << '\n';
    }
}

}  // namespace

// ---------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------
int main(int argc, char **argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return 2;
    }

    const int overlap_bytes = overlap_bytes_for_k(args.k);
    trace("k=" + std::to_string(args.k) +
          " overlap_bytes=" + std::to_string(overlap_bytes) +
          " launch_chunks=" + std::to_string(args.launch_chunks));

    // Load packed-2-bit input from disk.
    auto input_buf = read_packed_input(args.input_path);
    trace("input bytes=" + std::to_string(input_buf.size()));

    // Plan streaming chunks (with per-k overlap).
    auto chunks = plan_chunks(input_buf.size(), overlap_bytes);
    const size_t n_chunks_host = chunks.size();
    trace("planned chunks=" + std::to_string(n_chunks_host));

    // Load instr binary.
    auto instr_v = read_instr_binary(args.instr);
    trace("instr_words=" + std::to_string(instr_v.size()));

    // Per-chunk seq_in BO size = chunk_bytes_base + overlap_bytes
    // (last chunk may carry fewer bytes; we still allocate full slot
    // and zero-pad).
    const size_t seq_in_slot_bytes =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes);

    // Per-chunk sparse_out BO size = launch_chunks * EMIT_SLOT_BYTES.
    // (One EMIT_SLOT per active match-tile; T1 contract pins.)
    const size_t sparse_out_slot_bytes =
        static_cast<size_t>(args.launch_chunks) *
        static_cast<size_t>(EMIT_SLOT_BYTES);

    // ------------------------------------------------------------
    // XRT setup.
    // ------------------------------------------------------------
    trace("xrt device open");
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    trace("xclbin load path=" + args.xclbin);
    auto xclbin = xrt::xclbin(args.xclbin);
    device.register_xclbin(xclbin);

    xrt::hw_context context(device, xclbin.get_uuid());
    trace("kernel open name=" + args.kernel);
    auto kernel = xrt::kernel(context, args.kernel);

    // Allocate 3 BOs: instr, seq_in (one slot, reused per chunk),
    // sparse_out (one slot per launch, accumulated host-side).
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_seq_in = xrt::bo(device, seq_in_slot_bytes,
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_sparse_out = xrt::bo(device, sparse_out_slot_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    auto *p_instr = bo_instr.map<uint32_t *>();
    std::memcpy(p_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));
    auto *p_seq_in = bo_seq_in.map<uint8_t *>();
    auto *p_sparse_out = bo_sparse_out.map<uint8_t *>();

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Host-side accumulator for ALL emitted records across ALL chunks
    // and ALL timed iters of the LAST iteration only (warmup discards
    // host data; iter k>warmup keeps host data).
    std::vector<uint8_t> all_out_concat;
    all_out_concat.reserve(n_chunks_host * sparse_out_slot_bytes);

    const unsigned int opcode = 3;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;
    int timed_chunks = 0;

    const int total_iters = args.warmup + args.iters;
    for (int it = 0; it < total_iters; ++it) {
        const bool keep = (it >= args.warmup);
        if (keep) all_out_concat.clear();

        for (size_t ci = 0; ci < n_chunks_host; ++ci) {
            const ChunkPlan &c = chunks[ci];

            // Zero-pad the seq_in slot so the tail of the LAST
            // (possibly short) chunk doesn't alias stale data from
            // a previous chunk.
            std::memset(p_seq_in, 0, seq_in_slot_bytes);
            std::memcpy(p_seq_in,
                        input_buf.data() + c.src_offset,
                        c.bytes);
            bo_seq_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Zero the output slot so leftover records from the
            // previous chunk don't double-count after re-aggregation.
            std::memset(p_sparse_out, 0, sparse_out_slot_bytes);
            bo_sparse_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            auto t0 = std::chrono::high_resolution_clock::now();
            // Kernel signature: (opcode, instr, instr_size, seq_in,
            // sparse_out, n_input_bytes_for_this_chunk).
            // The last arg lets the kernel bound its rolling loop on
            // the final (short) chunk.
            auto run = kernel(opcode, bo_instr,
                              static_cast<unsigned>(instr_v.size()),
                              bo_seq_in, bo_sparse_out,
                              static_cast<int>(c.bytes));
            run.wait();
            auto t1 = std::chrono::high_resolution_clock::now();

            bo_sparse_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            if (keep) {
                size_t pos = all_out_concat.size();
                all_out_concat.resize(pos + sparse_out_slot_bytes);
                std::memcpy(all_out_concat.data() + pos,
                            p_sparse_out, sparse_out_slot_bytes);
            }

            float us = std::chrono::duration_cast<std::chrono::microseconds>(
                           t1 - t0).count();
            if (keep) {
                total_us += us;
                if (us < min_us) min_us = us;
                if (us > max_us) max_us = us;
                timed_chunks++;
            }
        }
    }

    // ------------------------------------------------------------
    // Host-side re-aggregation + Jellyfish-FASTA emit.
    // ------------------------------------------------------------
    trace("re-aggregating " +
          std::to_string(all_out_concat.size() / sizeof(EmitRecord)) +
          " emit slots");
    auto merged = reaggregate_records(all_out_concat);
    trace("unique canonicals after merge=" + std::to_string(merged.size()));

    emit_jellyfish_fasta(args.output_path, merged, args.k,
                         args.top, args.threshold);
    trace("wrote Jellyfish-FASTA to " + args.output_path);

    float avg_us = (timed_chunks > 0) ? (total_us / timed_chunks) : 0.0f;
    if (timed_chunks == 0) {
        // Defensive: keep the regex-parseable lines well-formed even
        // if no chunks were timed (e.g. zero-length input). A zero
        // value is intentional and parseable by `_RE_AVG/MIN/MAX`.
        min_us = 0.0f;
        max_us = 0.0f;
    }
    std::cout << "Avg NPU time: " << avg_us << "us." << std::endl;
    std::cout << "Min NPU time: " << min_us << "us." << std::endl;
    std::cout << "Max NPU time: " << max_us << "us." << std::endl;
    std::cout << "\nPASS!\n\n";
    return 0;
}
