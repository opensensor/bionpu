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
// Per state/kmer_count_interface_contract.md (T1) v0.5 — symbols, ObjectFifo
// names, constants, the streaming chunk-with-overlap protocol, and the
// host-side hash-slice multi-pass loop are pinned by the contract's
// "v0.5 REDESIGN" section.
//
// v0.5 host-side dataflow:
//   1. Load N_PASSES xclbins (one per pass_idx; xclbin/insts paths are
//      derived from --xclbin / --instr templates by appending _p{idx}
//      OR by passing --xclbins-dir <dir> with final_p{i}.xclbin per pass).
//   2. Plan input chunks (4096 bytes + per-k overlap, chunk-aligned).
//   3. For each chunk, for each pass_idx: zero the output BO,
//      kernel.run(opcode, instr, instr_size, bo_seq_in, bo_sparse_out,
//      n_input_bytes), wait, sync from device, parse the joined output
//      blob (N_TILES × [uint32 emit_idx][emit_idx × uint64 canonical]),
//      counts[canonical]++ for each emitted canonical.
//   4. After all chunks × all passes, sort by (count desc, canonical asc),
//      apply --top + --threshold, emit Jellyfish-FASTA.

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
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Pinned constants (mirror kmer_count_constants.h byte-equal).
// ---------------------------------------------------------------------

constexpr int32_t SEQ_IN_CHUNK_BYTES_BASE = 4096;

// Per-k overlap = max(ceil((k-1)/4), 4-byte alignment requirement).
constexpr int32_t SEQ_IN_OVERLAP_K15 = 4;
constexpr int32_t SEQ_IN_OVERLAP_K21 = 8;
constexpr int32_t SEQ_IN_OVERLAP_K31 = 8;

static_assert(SEQ_IN_OVERLAP_K15 >= ((15 - 1) + 3) / 4 &&
              (SEQ_IN_CHUNK_BYTES_BASE + SEQ_IN_OVERLAP_K15) % 4 == 0,
              "k=15 overlap must cover (k-1) bases AND keep total 4-byte aligned");
static_assert(SEQ_IN_OVERLAP_K21 >= ((21 - 1) + 3) / 4 &&
              (SEQ_IN_CHUNK_BYTES_BASE + SEQ_IN_OVERLAP_K21) % 4 == 0,
              "k=21 overlap must cover (k-1) bases AND keep total 4-byte aligned");
static_assert(SEQ_IN_OVERLAP_K31 >= ((31 - 1) + 3) / 4 &&
              (SEQ_IN_CHUNK_BYTES_BASE + SEQ_IN_OVERLAP_K31) % 4 == 0,
              "k=31 overlap must cover (k-1) bases AND keep total 4-byte aligned");

// v0.5 partial-out element size: 32 KiB per pass-slot. Layout per slot:
//   [uint32 emit_idx LE prefix][emit_idx × uint64 canonical][zero pad]
constexpr int32_t PARTIAL_OUT_BYTES_V05_PADDED = 32768;
constexpr int32_t MAX_EMIT_IDX_V05 = 4095;

constexpr uint64_t KMER_MASK_K15 = (1ULL << 30) - 1ULL;
constexpr uint64_t KMER_MASK_K21 = (1ULL << 42) - 1ULL;
constexpr uint64_t KMER_MASK_K31 = (1ULL << 62) - 1ULL;

// SLICE_HASH_SHIFT is 0 (low bits of canonical = slice index).
// (No need to use it host-side; the kernel applies it during emit.)

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
// 2-bit canonical -> ACGT decoder. Inverse of T2's pack_dna_2bit; matches
// T1's MSB-first wire format. Decodes the lowest 2*k bits of `canonical`
// to a length-k ASCII string. Bit pair (k-1) is the MSB (most significant
// 2 bits within the masked region) → first base of the k-mer.
// ---------------------------------------------------------------------
char base_of_2bit(uint64_t code) {
    switch (static_cast<unsigned>(code & 0x3ULL)) {
        case 0: return 'A';
        case 1: return 'C';
        case 2: return 'G';
        case 3: return 'T';
    }
    return 'N';
}

std::string decode_canonical_to_acgt(uint64_t canonical, int k) {
    uint64_t masked = canonical & kmer_mask_for_k(k);
    std::string out;
    out.resize(static_cast<size_t>(k));
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
    // Comma-separated list of xclbin paths, one per pass. Length must equal n_passes.
    // Alternatively, `--xclbin <single>` sets pass 0 (n_passes=1) or
    // `--xclbins-dir <dir>` finds final_p{i}.xclbin / insts_p{i}.bin in <dir>.
    std::string xclbins_dir;
    std::string xclbin_template;        // single xclbin path; for n_passes=1 OR template-mode
    std::string instr_template;         // matching insts.bin path
    std::string artifact_suffix;        // optional, used when constructing per-pass paths from a base dir
    std::string kernel = "MLIR_AIE";
    std::string input_path;
    std::string output_path;
    std::string output_format = "fasta";  // {fasta, binary} — v1.1 (b) gap close
    int k = 21;
    int top = 1000;
    int threshold = 1;
    int launch_chunks = 4;
    int n_passes = 4;
    int iters = 1;
    int warmup = 0;
    // v1.2 (b): how many chunks one silicon dispatch processes. Must
    // match the IRON-side BIONPU_KMER_COUNT_N_CHUNKS_PER_LAUNCH baked
    // into the xclbin. Default 1 = legacy per-chunk dispatch.
    int n_chunks_per_launch = 1;
};

void print_usage(std::ostream &os) {
    os <<
"Usage: kmer_count_runner [options]\n"
"\n"
"Required:\n"
"  -x, --xclbin <path>         xclbin file (final_p0.xclbin) — pass-0 xclbin\n"
"                              when --n-passes>1, the runner derives pass-i\n"
"                              xclbin by replacing the final '_p0' (or 'p0')\n"
"                              token in the filename with '_p<i>' (or 'p<i>').\n"
"  -i, --instr <path>          NPU instructions binary (insts_p0.bin) — pass-0\n"
"                              instr; same _p<i>-substitution as --xclbin.\n"
"  --input <packed_2bit>       packed-2-bit input (.2bit.bin)\n"
"  --output <jf_fasta>         Jellyfish-FASTA output ('>count\\nkmer\\n')\n"
"  --k {15,21,31}              k-mer length\n"
"\n"
"Optional:\n"
"  -k, --kernel <name>         kernel name (default MLIR_AIE)\n"
"  --top <N>                   top-N records (default 1000; 0 = all)\n"
"  --launch-chunks {1,2,4,8}   tile fan-out (default 4)\n"
"  --n-passes {1,4,16}         hash-slice partition count (default 4)\n"
"  --n-chunks-per-launch {1,2,4,8}\n"
"                              chunks processed per silicon dispatch\n"
"                              (default 1; v1.2 (b) batched-dispatch knob.\n"
"                              MUST match the value baked into the xclbin\n"
"                              by IRON Python at build time).\n"
"  --iters <N>                 timed iterations (default 1)\n"
"  --warmup <N>                untimed warmup iters (default 0)\n"
"  --threshold <int>           min count to emit (default 1)\n"
"  --output-format {fasta,binary} output wire format (default fasta).\n"
"                              binary: little-endian uint64 n_records,\n"
"                              followed by n_records × {uint64 canonical,\n"
"                              uint64 count}. Used by the v1.1 (b) op-class\n"
"                              path to skip the Python Jellyfish-FASTA parse.\n"
"\n"
"Per state/kmer_count_interface_contract.md v0.5: the runner dispatches\n"
"N_PASSES xclbins per chunk-batch (one per hash-slice partition),\n"
"accumulates canonical k-mers via flat-vector + sort-RLE merge\n"
"(v1.2 (b)), then sorts + topN + emits Jellyfish-FASTA / binary blob.\n";
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
        if (key == "-x" || key == "--xclbin") a.xclbin_template = next();
        else if (key == "-i" || key == "--instr") a.instr_template = next();
        else if (key == "-k" || key == "--kernel") a.kernel = next();
        else if (key == "--input") a.input_path = next();
        else if (key == "--output") a.output_path = next();
        else if (key == "--k") a.k = std::stoi(next());
        else if (key == "--top") a.top = std::stoi(next());
        else if (key == "--threshold") a.threshold = std::stoi(next());
        else if (key == "--launch-chunks") a.launch_chunks = std::stoi(next());
        else if (key == "--n-passes") a.n_passes = std::stoi(next());
        else if (key == "--n-chunks-per-launch") a.n_chunks_per_launch = std::stoi(next());
        else if (key == "--iters") a.iters = std::stoi(next());
        else if (key == "--warmup") a.warmup = std::stoi(next());
        else if (key == "--output-format") a.output_format = next();
        else throw std::runtime_error("unknown arg: " + key);
    }
    if (a.xclbin_template.empty() || a.instr_template.empty() ||
        a.input_path.empty() || a.output_path.empty())
        throw std::runtime_error(
            "required: -x <xclbin> -i <instr> --input <path> "
            "--output <path> --k {15,21,31}");
    if (a.k != 15 && a.k != 21 && a.k != 31)
        throw std::runtime_error("--k must be one of {15, 21, 31}");
    if (a.launch_chunks != 1 && a.launch_chunks != 2 &&
        a.launch_chunks != 4 && a.launch_chunks != 8)
        throw std::runtime_error("--launch-chunks must be one of {1, 2, 4, 8}");
    if (a.n_passes != 1 && a.n_passes != 4 && a.n_passes != 16)
        throw std::runtime_error("--n-passes must be one of {1, 4, 16}");
    if (a.n_chunks_per_launch != 1 && a.n_chunks_per_launch != 2 &&
        a.n_chunks_per_launch != 4 && a.n_chunks_per_launch != 8)
        throw std::runtime_error(
            "--n-chunks-per-launch must be one of {1, 2, 4, 8}");
    if (a.iters <= 0)
        throw std::runtime_error("--iters must be > 0");
    if (a.warmup < 0)
        throw std::runtime_error("--warmup must be >= 0");
    if (a.top < 0)
        throw std::runtime_error("--top must be >= 0");
    if (a.threshold < 0)
        throw std::runtime_error("--threshold must be >= 0");
    if (a.output_format != "fasta" && a.output_format != "binary")
        throw std::runtime_error(
            "--output-format must be one of {fasta, binary}; got '" +
            a.output_format + "'");
    return a;
}

// ---------------------------------------------------------------------
// Per-pass artifact path resolver.
//
// Convention: --xclbin and --instr point to the pass-0 artifacts
// (containing '_p0' or 'p0' in filename). Per-pass paths derived by
// replacing the substring with '_p<i>' / 'p<i>'.
//
// For n_passes=1, pass_idx=0 — path used as-is. The caller may pass a
// path without '_p0' substring; that's fine for n_passes=1 (we pass it
// through unchanged).
// ---------------------------------------------------------------------
std::string resolve_pass_path(const std::string &template_path,
                              int pass_idx, int n_passes) {
    if (n_passes == 1) {
        return template_path;  // single pass; use template path as-is
    }
    // Try '_p<n>' substring substitution (standard Makefile naming).
    std::string out = template_path;
    std::string from = "_p0";
    std::string to = "_p" + std::to_string(pass_idx);
    auto pos = out.rfind(from);
    if (pos != std::string::npos) {
        out.replace(pos, from.size(), to);
        return out;
    }
    // Fallback: append _p<i> before the final extension.
    auto dot = template_path.rfind('.');
    if (dot != std::string::npos) {
        out = template_path.substr(0, dot) + "_p" + std::to_string(pass_idx) +
              template_path.substr(dot);
        return out;
    }
    return template_path + "_p" + std::to_string(pass_idx);
}

// ---------------------------------------------------------------------
// Streaming chunk-with-overlap dispatch planner.
//
// v1.2 (a): in-band header expanded from 4 bytes to 8 bytes. Bytes
// [0..3] carry uint32 LE actual_bytes (existing); bytes [4..7] carry
// int32 LE owned_start_offset_bases (per-chunk; chunk 0 = 0; chunk i
// (i>0) = overlap_bases - (k-1)). Closes kmer-chunk-overlap-double-emit.
//
// Each chunk carries up to (SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes - 8)
// bytes of payload (the leading 8 bytes are the in-band header consumed
// by the kernel). Without the -8 the kernel's header would clobber the
// first 8 payload bytes.
// ---------------------------------------------------------------------
constexpr size_t HEADER_BYTES = 8;  // v1.2 (a): was 4; now 4 (length) + 4 (owned_offset)

struct ChunkPlan {
    size_t src_offset;     // first byte of this chunk in the source buffer
    size_t bytes;          // bytes copied into the BO for this chunk
                           // (<= chunk_payload_capacity)
};

std::vector<ChunkPlan> plan_chunks(size_t input_bytes, int overlap_bytes) {
    std::vector<ChunkPlan> plan;
    if (input_bytes == 0) return plan;
    const size_t total_chunk = static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE)
                             + static_cast<size_t>(overlap_bytes);
    const size_t payload_cap = total_chunk - HEADER_BYTES;
    // Advance per chunk by base = payload_cap - overlap_bytes (so the
    // overlap region is re-read by the next chunk to preserve k-mers
    // straddling boundaries).
    const size_t advance = payload_cap - static_cast<size_t>(overlap_bytes);
    size_t off = 0;
    while (off < input_bytes) {
        size_t end = std::min(off + payload_cap, input_bytes);
        ChunkPlan p;
        p.src_offset = off;
        p.bytes = end - off;
        plan.push_back(p);
        if (end >= input_bytes) break;
        off += advance;
    }
    return plan;
}

// ---------------------------------------------------------------------
// Per-pass output blob parser.
//
// Output layout per pass: N_TILES × PARTIAL_OUT_BYTES_V05_PADDED bytes.
// Per-tile slot:
//   [uint32 emit_idx LE prefix]
//   [emit_idx × uint64 canonical]
//   [middle zero pad]
//   [uint32 LE all_a_counter at offset PARTIAL_OUT_BYTES_V05_PADDED-4]
//
// For each canonical, counts[canonical]++. Each emit IS one observation
// (NOT a count summary) — the kernel emits one canonical per k-mer
// occurrence whose canonical falls into the active hash slice.
//
// v1.2 (a): the trailing all_a_counter is added to counts[0] (only ever
// non-zero on the pass=0 xclbin; canonical=0 lands in pass=0 by the
// slice mask). Closes kmer-chr22-canonical0-cap-fire.
// ---------------------------------------------------------------------
// v1.2 (b): accumulate emits from one (chunk_or_batch × pass) blob into
// a flat std::vector<uint64_t> sink. all_a counters are tracked in a
// separate uint64 (avoids appending millions of zeros to all_emits and
// keeps Phase 2 sort cheaper). After all dispatches, the host sorts
// all_emits, run-length encodes adjacent equal values to (canonical,
// count), and inserts (0, all_a_total) into the records list. Closes
// kmer-runner-host-merge-unordered_map.
//
// Returns the count of canonicals appended (excluding all_a). Out
// param max_emit_observed_io tracks the kernel-side per-slot emit_idx
// for the cap-fire warning.
size_t accumulate_pass_blob(const uint8_t *blob, size_t blob_size,
                            int n_tiles, int n_chunks_per_launch,
                            std::vector<uint64_t> &all_emits,
                            uint64_t &all_a_total,
                            size_t &out_emit_total,
                            uint32_t &max_emit_observed_io) {
    size_t emits = 0;
    // The IRON-side memtile.join produces one joined element per
    // tile-chunk iteration; the runtime sequence drains
    // n_chunks_per_launch joined elements per dispatch, laid out
    // contiguously in the BO. Per-iteration block layout:
    //   N_TILES × PARTIAL_OUT_BYTES_V05_PADDED bytes
    const size_t per_iter_bytes =
        static_cast<size_t>(n_tiles) *
        static_cast<size_t>(PARTIAL_OUT_BYTES_V05_PADDED);

    for (int it = 0; it < n_chunks_per_launch; ++it) {
        const size_t iter_base = static_cast<size_t>(it) * per_iter_bytes;
        for (int t = 0; t < n_tiles; ++t) {
            size_t off = iter_base +
                         static_cast<size_t>(t) *
                             static_cast<size_t>(PARTIAL_OUT_BYTES_V05_PADDED);
            if (off + 4 > blob_size) break;
            uint32_t emit_idx = 0;
            std::memcpy(&emit_idx, blob + off, 4);
            if (emit_idx > max_emit_observed_io) {
                max_emit_observed_io = emit_idx;
            }
            if (emit_idx > static_cast<uint32_t>(MAX_EMIT_IDX_V05)) {
                // Defensive: kernel must cap at MAX_EMIT_IDX_V05; if blob
                // returned a higher value, treat it as corruption and clamp.
                emit_idx = static_cast<uint32_t>(MAX_EMIT_IDX_V05);
            }
            size_t payload_off = off + 4;
            size_t payload_max = static_cast<size_t>(emit_idx) * 8u;
            if (payload_off + payload_max > blob_size) {
                payload_max = (blob_size > payload_off) ? (blob_size - payload_off) : 0;
                payload_max = (payload_max / 8u) * 8u;
            }
            // Bulk append: 8-byte canonicals are little-endian uint64
            // already; one memcpy into a pre-resized vector slice is
            // cheaper than a per-record memcpy + push_back.
            const size_t old_size = all_emits.size();
            const size_t add = static_cast<size_t>(emit_idx);
            all_emits.resize(old_size + add);
            if (add > 0) {
                std::memcpy(all_emits.data() + old_size,
                            blob + payload_off, add * 8u);
            }
            emits += add;

            // v1.2 (a): read trailing all_a_counter. Slot tail offset =
            // off + PARTIAL_OUT_BYTES_V05_PADDED - 4.
            size_t all_a_off = off +
                               static_cast<size_t>(PARTIAL_OUT_BYTES_V05_PADDED) - 4u;
            if (all_a_off + 4 <= blob_size) {
                uint32_t all_a_counter = 0;
                std::memcpy(&all_a_counter, blob + all_a_off, 4);
                if (all_a_counter > 0u) {
                    all_a_total += static_cast<uint64_t>(all_a_counter);
                    emits += static_cast<size_t>(all_a_counter);
                }
            }
        }
    }
    out_emit_total += emits;
    return emits;
}

// ---------------------------------------------------------------------
// Sort + filter + emit Jellyfish-FASTA.
// Output format per T1 / PRD §3.2 / Q5: '>count\nkmer\n' per record.
// Sort key: (count desc, canonical asc).
// ---------------------------------------------------------------------
struct FinalRecord {
    uint64_t canonical;
    uint64_t count;
};

// v1.2 (b): sort-then-RLE merge. Replaces the previous unordered_map
// accumulate (5-10s on chr22's 42M emits → 30.7M unique canonicals).
//
// Phase 1 (collection): caller appended every emit to all_emits +
// tracked all_a_total separately.
// Phase 2 (this fn):
//   1. std::sort the flat vector (O(n log n); ~1-2s for 42M uint64 on
//      a single core; the implementation is much more cache-friendly
//      than an unordered_map of node-allocated buckets).
//   2. Linear sweep adjacent-equal to produce (canonical, count) pairs.
//   3. Insert (0, all_a_total) at the canonical=0 position (front,
//      since 0 sorts first).
//   4. Re-sort by (count desc, canonical asc); apply --top + --threshold.
//
// Closes kmer-runner-host-merge-unordered_map.
std::vector<FinalRecord> sort_and_filter(
        std::vector<uint64_t> &all_emits,
        uint64_t all_a_total,
        int top, int threshold) {
    // Phase 2.1: sort all_emits. v1.3: parallel sort via libstdc++
    // <execution> + TBB (C++17 parallel std::sort). Drops 4.6s sort
    // wall (chr22 single-thread) to <1s on 16-core HX laptop.
    std::sort(std::execution::par_unseq,
              all_emits.begin(), all_emits.end());

    // Phase 2.2: RLE sweep. Pre-reserve a generous upper bound (unique
    // count ≤ all_emits.size() + 1; reserving exact size avoids
    // reallocations during the sweep).
    std::vector<FinalRecord> v;
    v.reserve(all_emits.size() / 2u + 16u);
    size_t i = 0;
    const size_t n = all_emits.size();
    bool inserted_canonical0 = false;
    while (i < n) {
        const uint64_t c = all_emits[i];
        size_t j = i;
        while (j < n && all_emits[j] == c) ++j;
        uint64_t cnt = static_cast<uint64_t>(j - i);
        if (c == 0ULL) {
            // Merge any kernel-emitted canonical=0 (should never happen
            // post-v1.2 (a) since the kernel diverts to all_a_counter,
            // but guard for safety) with the summary counter.
            cnt += all_a_total;
            inserted_canonical0 = true;
        }
        if (cnt >= static_cast<uint64_t>(threshold)) {
            v.push_back({c, cnt});
        }
        i = j;
    }
    if (!inserted_canonical0 && all_a_total > 0ULL &&
        all_a_total >= static_cast<uint64_t>(threshold)) {
        v.push_back({0ULL, all_a_total});
    }

    // Phase 2.3: re-sort by (count desc, canonical asc). v1.3: when
    // top is small (typical 1000), use partial_sort to avoid sorting
    // the full unique-canonical list (~30M records on chr22).
    if (top > 0 && static_cast<size_t>(top) < v.size()) {
        std::partial_sort(
            v.begin(), v.begin() + static_cast<std::ptrdiff_t>(top), v.end(),
            [](const FinalRecord &a, const FinalRecord &b) {
                if (a.count != b.count) return a.count > b.count;
                return a.canonical < b.canonical;
            });
        v.resize(static_cast<size_t>(top));
    } else {
        std::sort(std::execution::par_unseq,
                  v.begin(), v.end(),
                  [](const FinalRecord &a, const FinalRecord &b) {
                      if (a.count != b.count) return a.count > b.count;
                      return a.canonical < b.canonical;
                  });
    }
    return v;
}

void emit_jellyfish_fasta(const std::string &path,
                          const std::vector<FinalRecord> &v, int k) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("cannot open output for write: " + path);
    for (const auto &r : v) {
        f << '>' << r.count << '\n'
          << decode_canonical_to_acgt(r.canonical, k) << '\n';
    }
}

// Binary blob output (v1.1 (b) — closes kmer-host-postpass-python-bottleneck).
// Layout (all little-endian):
//   [uint64 n_records][n_records × {uint64 canonical, uint64 count}]
// Total size = 8 + n_records*16 bytes. Op-class on the Python side
// `np.fromfile`s this into an ndarray and does NO ASCII parse.
void emit_binary_blob(const std::string &path,
                      const std::vector<FinalRecord> &v) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open output for write: " + path);
    uint64_t n = static_cast<uint64_t>(v.size());
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &r : v) {
        f.write(reinterpret_cast<const char *>(&r.canonical), sizeof(r.canonical));
        f.write(reinterpret_cast<const char *>(&r.count),     sizeof(r.count));
    }
}

// ---------------------------------------------------------------------
// Per-pass loaded artifacts. We load all N_PASSES xclbins up-front so
// dispatching per-pass is O(1) per kernel call.
// ---------------------------------------------------------------------
struct PassArtifacts {
    xrt::xclbin xclbin;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo bo_instr;
    std::vector<uint32_t> instr;
};

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
          " launch_chunks=" + std::to_string(args.launch_chunks) +
          " n_passes=" + std::to_string(args.n_passes) +
          " n_chunks_per_launch=" + std::to_string(args.n_chunks_per_launch));

    // Load packed-2-bit input from disk.
    auto input_buf = read_packed_input(args.input_path);
    trace("input bytes=" + std::to_string(input_buf.size()));

    // Plan streaming chunks (with per-k overlap).
    auto chunks = plan_chunks(input_buf.size(), overlap_bytes);
    const size_t n_chunks_host = chunks.size();
    trace("planned chunks=" + std::to_string(n_chunks_host));

    // v1.2 (b): per-chunk seq_in slot size = chunk_bytes_base + overlap_bytes.
    const size_t seq_in_slot_bytes =
        static_cast<size_t>(SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes);

    // Per-batch BO sizes: each silicon dispatch processes
    // n_chunks_per_launch chunks. seq_in BO = N_BATCH × slot bytes;
    // partial_out BO = N_BATCH × (N_TILES × 32 KiB).
    const int n_batch = args.n_chunks_per_launch;
    const size_t per_batch_seq_in_bytes =
        static_cast<size_t>(n_batch) * seq_in_slot_bytes;
    const size_t per_pass_out_bytes =
        static_cast<size_t>(n_batch) *
        static_cast<size_t>(args.launch_chunks) *
        static_cast<size_t>(PARTIAL_OUT_BYTES_V05_PADDED);

    // Number of batches the host issues (round up; tail batch padded
    // with zero-actual-bytes chunks so the kernel sees exactly N_BATCH
    // chunk acquires).
    const size_t n_batches =
        (n_chunks_host + static_cast<size_t>(n_batch) - 1u) /
        static_cast<size_t>(n_batch);
    trace("planned batches=" + std::to_string(n_batches) +
          " (n_chunks_per_launch=" + std::to_string(n_batch) + ")");

    // ------------------------------------------------------------
    // XRT setup — load N_PASSES xclbins.
    // ------------------------------------------------------------
    trace("xrt device open");
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    std::vector<PassArtifacts> per_pass(static_cast<size_t>(args.n_passes));
    for (int p = 0; p < args.n_passes; ++p) {
        std::string xclbin_path =
            resolve_pass_path(args.xclbin_template, p, args.n_passes);
        std::string instr_path =
            resolve_pass_path(args.instr_template, p, args.n_passes);
        trace("pass=" + std::to_string(p) +
              " xclbin=" + xclbin_path +
              " instr=" + instr_path);

        per_pass[p].xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(per_pass[p].xclbin);
        per_pass[p].context = xrt::hw_context(
            device, per_pass[p].xclbin.get_uuid());
        per_pass[p].kernel = xrt::kernel(per_pass[p].context, args.kernel);
        per_pass[p].instr = read_instr_binary(instr_path);

        per_pass[p].bo_instr = xrt::bo(
            device, per_pass[p].instr.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, per_pass[p].kernel.group_id(1));
        auto *p_instr = per_pass[p].bo_instr.map<uint32_t *>();
        std::memcpy(p_instr, per_pass[p].instr.data(),
                    per_pass[p].instr.size() * sizeof(uint32_t));
        per_pass[p].bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Allocate seq_in / sparse_out BOs against pass-0's kernel
    // (group_ids should match across passes since the IRON Python
    // graph topology is identical across n_passes_log2/pass_idx values).
    // v1.2 (b): BOs sized for one BATCH of n_chunks_per_launch chunks.
    //
    // v1.3: depth-2 ring of (seq_in, sparse_out) BOs to enable pipelined
    // dispatch. While silicon is computing dispatch i, host accumulates
    // dispatch i-1's output and stages dispatch i+1's seq_in.
    //
    //   seq_in_ring[2]: ping-pongs PER BATCH (each batch reuses the
    //     same seq_in across n_passes dispatches; both passes within
    //     a batch read the same seq_in BO, so the ping-pong is at the
    //     batch granularity, not per-dispatch).
    //   sparse_out_ring[2]: ping-pongs PER DISPATCH (every dispatch
    //     writes a fresh output blob; we double-buffer so dispatch i+1
    //     can write to the alternate BO while host parses dispatch i's
    //     output).
    //
    // Ring depth = 2 is sufficient for the access pattern (1 host
    // consumer + 1 silicon producer with no cross-dependency between
    // adjacent dispatches' outputs).
    const auto &k0 = per_pass[0].kernel;
    // v1.3 ring depth. Each slot holds (xrt::run, sparse_out BO, seq_in
    // tag). RING=4 gives XRT 3-deep submit-ahead while host accumulates,
    // hiding more cross-dispatch queue latency than depth=2. seq_in is
    // ping-ponged at batch granularity; with RING=4 and n_passes=4, the
    // last dispatch of batch bi shares seq_in[bi%2] with all 4 of its
    // in-flight passes — no aliasing because batch bi+1 always uses
    // seq_in[(bi+1)%2].
    constexpr int RING = 4;
    xrt::bo bo_seq_in_ring[RING];
    xrt::bo bo_sparse_out_ring[RING];
    uint8_t *p_seq_in_ring[RING];
    uint8_t *p_sparse_out_ring[RING];
    for (int s = 0; s < RING; ++s) {
        bo_seq_in_ring[s] = xrt::bo(device, per_batch_seq_in_bytes,
                                    XRT_BO_FLAGS_HOST_ONLY, k0.group_id(3));
        bo_sparse_out_ring[s] = xrt::bo(device, per_pass_out_bytes,
                                        XRT_BO_FLAGS_HOST_ONLY, k0.group_id(4));
        p_seq_in_ring[s] = bo_seq_in_ring[s].map<uint8_t *>();
        p_sparse_out_ring[s] = bo_sparse_out_ring[s].map<uint8_t *>();
    }

    // v1.2 (b): host-side accumulator is now a flat std::vector<uint64_t>
    // of canonicals (sort-merge replaces unordered_map; closes
    // kmer-runner-host-merge-unordered_map). all_a counter tracked
    // separately to avoid 12M zero-appends on chr22 centromere.
    std::vector<uint64_t> all_emits;
    // Pre-reserve a generous upper bound for chr22 (42M emits at
    // n_tiles=4, n_passes=16). Smaller inputs simply use less.
    all_emits.reserve(64 * 1024 * 1024);
    uint64_t all_a_total = 0ULL;

    const unsigned int opcode = 3;
    float total_us = 0.0f;
    float min_us = 1e30f;
    float max_us = 0.0f;
    int timed_dispatches = 0;
    size_t total_emits = 0;
    uint32_t max_emit_observed = 0;

    // v1.3 diagnostic profiling: when BIONPU_KMER_PROFILE=1, record
    // per-phase µs (stage_seq_in, zero_out, sync_to_device, dispatch+wait,
    // sync_from_device, accumulate) for the first 10 dispatches of the
    // first timed iteration. Helps confirm where the budget goes (silicon
    // compute vs DMA sync vs host accumulate).
    const bool profile_enabled =
        (std::getenv("BIONPU_KMER_PROFILE") != nullptr) &&
        std::string(std::getenv("BIONPU_KMER_PROFILE")) == "1";

    // v1.3 fast path: skip the per-dispatch memset+sync_to_device of
    // sparse_out. The kernel writes emit_idx (offset 0) and
    // all_a_counter (offset PADDED-4) unconditionally; the parser
    // only reads exactly emit_idx canonicals starting at offset 4.
    // Leftover bytes from a prior dispatch beyond emit_idx are
    // harmless. This saves ~50 µs per dispatch (memset ~25µs +
    // sync_to_device ~25µs); at 1556 dispatches that's ~78 ms.
    // Set BIONPU_KMER_OUTPUT_ZERO=1 to restore the v1.2 (b) safe
    // behavior if a regression is suspected.
    const bool output_zero_enabled =
        (std::getenv("BIONPU_KMER_OUTPUT_ZERO") != nullptr) &&
        std::string(std::getenv("BIONPU_KMER_OUTPUT_ZERO")) == "1";
    constexpr int PROFILE_MAX_DISPATCHES = 10;
    struct ProfileRow {
        int dispatch_idx;
        size_t batch_idx;
        int pass_idx;
        float stage_seq_in_us;     // memcpy seq_in slots (per-batch only)
        float sync_seq_in_us;      // sync_to_device seq_in (per-batch only)
        float zero_out_us;         // memset of sparse_out
        float sync_out_to_dev_us;  // sync_to_device sparse_out (zero seed)
        float dispatch_us;         // kernel(...) + run.wait() + sync_from_device
        float accumulate_us;       // accumulate_pass_blob
    };
    std::vector<ProfileRow> profile_rows;
    if (profile_enabled) {
        profile_rows.reserve(PROFILE_MAX_DISPATCHES);
        trace("profile: BIONPU_KMER_PROFILE=1 — capturing first " +
              std::to_string(PROFILE_MAX_DISPATCHES) +
              " dispatches");
    }

    // v1.3 inflight ring slot: (xrt::run, slot index, dispatch index)
    struct InflightSlot {
        xrt::run run;
        int seq_slot;     // 0..RING-1
        int out_slot;     // 0..RING-1
        size_t batch_idx;
        int pass_idx;
        int dispatch_idx;
        bool valid;
        std::chrono::high_resolution_clock::time_point t_dispatch;
    };

    const int total_iters = args.warmup + args.iters;
    for (int it = 0; it < total_iters; ++it) {
        const bool keep = (it >= args.warmup);
        if (keep) {
            all_emits.clear();
            all_a_total = 0ULL;
            total_emits = 0;
        }

        InflightSlot inflight[RING];
        for (int s = 0; s < RING; ++s) inflight[s].valid = false;

        // Track which seq_in slot the most recent dispatch of each
        // batch wrote to (for the per-batch staging predicate).
        // Track whether each seq_in ring slot has been staged for the
        // currently-pending batch index; we re-stage when we advance
        // to a new batch using the ping-pong slot.
        size_t last_seq_batch[RING];
        bool seq_slot_has_data[RING];
        for (int s = 0; s < RING; ++s) {
            last_seq_batch[s] = static_cast<size_t>(-1);
            seq_slot_has_data[s] = false;
        }

        // v1.3 pipelined dispatch loop. We linearise (batch, pass) into
        // a single dispatch index d ∈ [0, n_batches × n_passes). For
        // each d, we issue an async kernel(...) (which start()s on
        // construction) into out_slot = d % RING. Before issuing, we
        // wait for any prior dispatch occupying out_slot, sync_from_
        // device on that slot, and accumulate its blob. After the loop,
        // we drain the remaining RING-1 inflight slots.
        //
        // seq_in is staged once per BATCH (seq_slot = batch_idx % RING).
        // The ring depth on sparse_out (= RING) implicitly bounds how
        // far ahead we can run; since RING=2 < n_passes (typically 4),
        // we drain pass 0 of batch bi BEFORE issuing pass 0 of batch
        // bi+1, which guarantees seq_in[bi % RING] is no longer in use
        // by the time we re-stage it for batch bi+RING.
        const size_t total_dispatches =
            n_batches * static_cast<size_t>(args.n_passes);

        for (size_t d = 0; d < total_dispatches; ++d) {
            const size_t bi = d / static_cast<size_t>(args.n_passes);
            const int p = static_cast<int>(d % static_cast<size_t>(args.n_passes));
            const int out_slot = static_cast<int>(d % static_cast<size_t>(RING));
            const int seq_slot = static_cast<int>(bi % static_cast<size_t>(RING));

            // (1) Drain the ring slot we're about to overwrite.
            float drain_us = 0.0f;
            float accumulate_us = 0.0f;
            if (inflight[out_slot].valid) {
                auto t_drain0 = std::chrono::high_resolution_clock::now();
                inflight[out_slot].run.wait();
                bo_sparse_out_ring[inflight[out_slot].out_slot].sync(
                    XCL_BO_SYNC_BO_FROM_DEVICE);
                auto t_drain1 = std::chrono::high_resolution_clock::now();
                drain_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t_drain1 - t_drain0).count();

                auto t_acc0 = std::chrono::high_resolution_clock::now();
                if (keep) {
                    accumulate_pass_blob(
                        p_sparse_out_ring[inflight[out_slot].out_slot],
                        per_pass_out_bytes,
                        args.launch_chunks,
                        n_batch,
                        all_emits,
                        all_a_total,
                        total_emits,
                        max_emit_observed);
                }
                auto t_acc1 = std::chrono::high_resolution_clock::now();
                accumulate_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t_acc1 - t_acc0).count();

                if (keep) {
                    auto td = std::chrono::duration_cast<
                        std::chrono::microseconds>(
                            t_drain1 - inflight[out_slot].t_dispatch).count();
                    float us = static_cast<float>(td);
                    total_us += us;
                    if (us < min_us) min_us = us;
                    if (us > max_us) max_us = us;
                    timed_dispatches++;
                }
                inflight[out_slot].valid = false;

                if (profile_enabled && keep &&
                    inflight[out_slot].dispatch_idx <
                        PROFILE_MAX_DISPATCHES &&
                    static_cast<int>(profile_rows.size()) >
                        inflight[out_slot].dispatch_idx) {
                    // Back-fill the drain + accumulate µs onto the row
                    // we created when we issued this dispatch.
                    profile_rows[inflight[out_slot].dispatch_idx]
                        .dispatch_us += drain_us;
                    profile_rows[inflight[out_slot].dispatch_idx]
                        .accumulate_us = accumulate_us;
                }
            }

            // (2) Stage seq_in for batch bi, if not already staged.
            //     Within a batch, all n_passes dispatches share the
            //     same seq_in BO; we stage exactly once (on pass 0 of
            //     the batch).
            float stage_seq_in_us = 0.0f;
            float sync_seq_in_us = 0.0f;
            if (last_seq_batch[seq_slot] != bi) {
                // Drain any other in-flight dispatch that may still be
                // reading from this seq_slot before we overwrite it.
                // (With RING=2 on sparse_out and n_passes>=2, this is
                // already guaranteed by step (1) above for typical
                // configurations, but we double-check to be safe for
                // n_passes=1 + RING=2.)
                for (int s = 0; s < RING; ++s) {
                    if (inflight[s].valid && inflight[s].seq_slot == seq_slot) {
                        inflight[s].run.wait();
                        bo_sparse_out_ring[inflight[s].out_slot].sync(
                            XCL_BO_SYNC_BO_FROM_DEVICE);
                        if (keep) {
                            accumulate_pass_blob(
                                p_sparse_out_ring[inflight[s].out_slot],
                                per_pass_out_bytes,
                                args.launch_chunks,
                                n_batch,
                                all_emits,
                                all_a_total,
                                total_emits,
                                max_emit_observed);
                            auto td = std::chrono::duration_cast<
                                std::chrono::microseconds>(
                                    std::chrono::high_resolution_clock::now() -
                                    inflight[s].t_dispatch).count();
                            float us = static_cast<float>(td);
                            total_us += us;
                            if (us < min_us) min_us = us;
                            if (us > max_us) max_us = us;
                            timed_dispatches++;
                        }
                        inflight[s].valid = false;
                    }
                }

                auto t_stage0 = std::chrono::high_resolution_clock::now();
                const size_t batch_first_chunk =
                    bi * static_cast<size_t>(n_batch);
                uint8_t *p_seq_in = p_seq_in_ring[seq_slot];
                std::memset(p_seq_in, 0, per_batch_seq_in_bytes);
                for (int slot = 0; slot < n_batch; ++slot) {
                    const size_t ci = batch_first_chunk +
                                      static_cast<size_t>(slot);
                    uint8_t *slot_base = p_seq_in +
                                         static_cast<size_t>(slot) *
                                             seq_in_slot_bytes;
                    if (ci >= n_chunks_host) continue;
                    const ChunkPlan &c = chunks[ci];
                    uint32_t actual_bytes_le = static_cast<uint32_t>(c.bytes);
                    std::memcpy(slot_base, &actual_bytes_le, 4);
                    int32_t owned_start_offset_bases =
                        (ci == 0)
                            ? 0
                            : (overlap_bytes * 4) - (args.k - 1);
                    std::memcpy(slot_base + 4, &owned_start_offset_bases, 4);
                    std::memcpy(slot_base + HEADER_BYTES,
                                input_buf.data() + c.src_offset,
                                c.bytes);
                }
                auto t_stage1 = std::chrono::high_resolution_clock::now();
                bo_seq_in_ring[seq_slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto t_stage2 = std::chrono::high_resolution_clock::now();
                stage_seq_in_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t_stage1 - t_stage0).count();
                sync_seq_in_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t_stage2 - t_stage1).count();
                last_seq_batch[seq_slot] = bi;
                seq_slot_has_data[seq_slot] = true;
            }

            // (3) Zero the output BO + sync_to_device. v1.3 default:
            //     SKIPPED — the kernel writes emit_idx and all_a_counter
            //     at fixed offsets unconditionally, and the parser only
            //     reads exactly emit_idx canonicals; leftover bytes are
            //     harmless. Restore via BIONPU_KMER_OUTPUT_ZERO=1 if a
            //     regression is suspected.
            float zero_out_us = 0.0f;
            float sync_out_to_dev_us = 0.0f;
            if (output_zero_enabled) {
                auto t_zero0 = std::chrono::high_resolution_clock::now();
                std::memset(p_sparse_out_ring[out_slot], 0,
                            per_pass_out_bytes);
                auto t_zero1 = std::chrono::high_resolution_clock::now();
                bo_sparse_out_ring[out_slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto t_zero2 = std::chrono::high_resolution_clock::now();
                zero_out_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t_zero1 - t_zero0).count();
                sync_out_to_dev_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t_zero2 - t_zero1).count();
            }

            // (4) Issue the kernel — non-blocking. kernel(...) returns
            //     a started xrt::run; we won't wait on it until step
            //     (1) of a future iteration drains it.
            auto t_disp = std::chrono::high_resolution_clock::now();
            xrt::run run = per_pass[p].kernel(
                opcode,
                per_pass[p].bo_instr,
                static_cast<unsigned>(per_pass[p].instr.size()),
                bo_seq_in_ring[seq_slot],
                bo_sparse_out_ring[out_slot],
                static_cast<int>(per_batch_seq_in_bytes));

            inflight[out_slot].run = std::move(run);
            inflight[out_slot].seq_slot = seq_slot;
            inflight[out_slot].out_slot = out_slot;
            inflight[out_slot].batch_idx = bi;
            inflight[out_slot].pass_idx = p;
            inflight[out_slot].dispatch_idx = static_cast<int>(d);
            inflight[out_slot].valid = true;
            inflight[out_slot].t_dispatch = t_disp;

            if (profile_enabled && keep &&
                static_cast<int>(d) < PROFILE_MAX_DISPATCHES) {
                ProfileRow row{};
                row.dispatch_idx = static_cast<int>(d);
                row.batch_idx = bi;
                row.pass_idx = p;
                row.stage_seq_in_us = stage_seq_in_us;
                row.sync_out_to_dev_us = sync_out_to_dev_us +
                                          sync_seq_in_us;
                row.zero_out_us = zero_out_us;
                row.dispatch_us = 0.0f;       // back-filled at drain
                row.accumulate_us = 0.0f;     // back-filled at drain
                profile_rows.push_back(row);
            }
        }

        // (5) Drain remaining RING-1 inflight slots in dispatch-issue
        //     order. We sort by dispatch_idx ascending so we accumulate
        //     in the same order as the v1.2 (b) sequential loop (the
        //     accumulator is order-independent — it's a flat append +
        //     sort-merge — so this is purely cosmetic for trace output).
        {
            std::vector<int> drain_order;
            for (int s = 0; s < RING; ++s) {
                if (inflight[s].valid) drain_order.push_back(s);
            }
            std::sort(drain_order.begin(), drain_order.end(),
                      [&](int a, int b) {
                          return inflight[a].dispatch_idx <
                                 inflight[b].dispatch_idx;
                      });
            for (int s : drain_order) {
                inflight[s].run.wait();
                bo_sparse_out_ring[inflight[s].out_slot].sync(
                    XCL_BO_SYNC_BO_FROM_DEVICE);
                auto t_drain1 = std::chrono::high_resolution_clock::now();
                if (keep) {
                    accumulate_pass_blob(
                        p_sparse_out_ring[inflight[s].out_slot],
                        per_pass_out_bytes,
                        args.launch_chunks,
                        n_batch,
                        all_emits,
                        all_a_total,
                        total_emits,
                        max_emit_observed);
                    auto td = std::chrono::duration_cast<
                        std::chrono::microseconds>(
                            t_drain1 - inflight[s].t_dispatch).count();
                    float us = static_cast<float>(td);
                    total_us += us;
                    if (us < min_us) min_us = us;
                    if (us > max_us) max_us = us;
                    timed_dispatches++;
                }
                inflight[s].valid = false;
            }
        }
    }

    // Emit profile diagnostic rows (if enabled).
    if (profile_enabled) {
        std::cerr << "\n[kmer_count_runner] profile per-dispatch (us):\n";
        std::cerr <<
            "  d batch pass | stage_seq_in sync_setup zero_out | "
            "dispatch+wait+sync_from accumulate\n";
        for (const auto &r : profile_rows) {
            std::cerr << "  " << r.dispatch_idx << "  "
                      << r.batch_idx << "    " << r.pass_idx << "  | "
                      << r.stage_seq_in_us << "  "
                      << r.sync_out_to_dev_us << "  "
                      << r.zero_out_us << "  | "
                      << r.dispatch_us << "  "
                      << r.accumulate_us << "\n";
        }
        std::cerr << std::flush;
    }

    if (max_emit_observed >= static_cast<uint32_t>(MAX_EMIT_IDX_V05)) {
        trace("WARN: max_emit_idx_observed=" +
              std::to_string(max_emit_observed) +
              " hit cap MAX_EMIT_IDX_V05=" +
              std::to_string(MAX_EMIT_IDX_V05) +
              " — kernel dropped k-mers; increase --n-passes");
    }
    trace("total emits across all passes=" + std::to_string(total_emits) +
          " all_a_total=" + std::to_string(all_a_total) +
          " all_emits.size=" + std::to_string(all_emits.size()));

    auto t_sort0 = std::chrono::high_resolution_clock::now();
    auto sorted_records = sort_and_filter(all_emits, all_a_total,
                                          args.top, args.threshold);
    auto t_sort1 = std::chrono::high_resolution_clock::now();
    auto sort_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t_sort1 - t_sort0).count();
    trace("sort_and_filter wall=" + std::to_string(sort_us) + "us");
    if (args.output_format == "binary") {
        emit_binary_blob(args.output_path, sorted_records);
        trace("wrote binary blob (" + std::to_string(sorted_records.size()) +
              " records) to " + args.output_path);
    } else {
        emit_jellyfish_fasta(args.output_path, sorted_records, args.k);
        trace("wrote Jellyfish-FASTA (" + std::to_string(sorted_records.size()) +
              " records) to " + args.output_path);
    }

    float avg_us = (timed_dispatches > 0) ?
                   (total_us / timed_dispatches) : 0.0f;
    if (timed_dispatches == 0) {
        min_us = 0.0f;
        max_us = 0.0f;
    }
    std::cout << "Avg NPU time: " << avg_us << "us." << std::endl;
    std::cout << "Min NPU time: " << min_us << "us." << std::endl;
    std::cout << "Max NPU time: " << max_us << "us." << std::endl;
    std::cout << "\nPASS!\n\n";
    return 0;
}
