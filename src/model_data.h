#pragma once
#include <cstddef>

// --------------------- model hyper-params -----------------------------------
constexpr size_t BATCH      = 1;
constexpr size_t IN_H       = 28;
constexpr size_t IN_W       = 28;
constexpr size_t IN_C       = 1;

constexpr size_t CONV_K     = 3;
constexpr size_t CONV_STR   = 1;
constexpr size_t CONV_OUT_C = 2;               // number of filters

constexpr size_t FC_OUT     = 3;               // number of classes

// --------------------- tiny hard-coded weights ------------------------------
// 2 filters × 1 input-ch × 3 × 3  = 18 floats
static const float CONV_W[CONV_OUT_C * IN_C * CONV_K * CONV_K] = {
  // filter 0
   0.2f,  0.0f, -0.1f,
   0.0f,  0.3f,  0.0f,
  -0.1f,  0.0f,  0.2f,
  // filter 1
  -0.2f,  0.1f,  0.0f,
   0.1f, -0.3f,  0.1f,
   0.0f,  0.2f, -0.2f
};
// 2 biases
static const float CONV_B[CONV_OUT_C] = { 0.05f, -0.05f };

// After 28×28 avg-pool we have 1×1×2 → flatten to 2 features.
// FC: 3 outputs × 2 inputs = 6 floats
static const float FC_W[FC_OUT * CONV_OUT_C] = {
  0.5f, -0.4f,
  0.3f,  0.2f,
 -0.1f,  0.6f
};
// 3 biases
static const float FC_B[FC_OUT] = { 0.1f, 0.0f, -0.1f };