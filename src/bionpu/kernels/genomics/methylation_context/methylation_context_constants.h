// bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
// Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
//
// methylation_context_constants.h — pinned ABI for methylation_context v0.

#pragma once

#include <stdint.h>

static constexpr int32_t MC_SEQ_IN_CHUNK_BYTES_BASE = 4096;
static constexpr int32_t MC_SEQ_IN_OVERLAP = 4;
static constexpr int32_t MC_HEADER_BYTES = 8;
static constexpr int32_t MC_PARTIAL_OUT_BYTES_PADDED = 32768;
static constexpr int32_t MC_RECORD_BYTES = 8;
static constexpr int32_t MC_MAX_EMIT_IDX = 4094;

static constexpr uint8_t MC_BASE_A = 0;
static constexpr uint8_t MC_BASE_C = 1;
static constexpr uint8_t MC_BASE_G = 2;
static constexpr uint8_t MC_BASE_T = 3;

static constexpr uint8_t MC_STRAND_PLUS = 0;
static constexpr uint8_t MC_STRAND_MINUS = 1;

static constexpr uint8_t MC_CONTEXT_CG = 0;
static constexpr uint8_t MC_CONTEXT_CHG = 1;
static constexpr uint8_t MC_CONTEXT_CHH = 2;
