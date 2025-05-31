#include <xnnpack.h>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include "model_data.h"

int main() {
  // ------------------ init runtime ------------------------------------------
  if (xnn_initialize(nullptr) != xnn_status_success) {
    std::cerr << "XNNPACK init failed\n";
    return 1;
  }

  // ------------------ allocate tensors --------------------------------------
  constexpr size_t IN_SIZE   = IN_H * IN_W * IN_C;
  constexpr size_t CONV_OUT  = IN_H * IN_W * CONV_OUT_C;      // same padding
  constexpr size_t POOL_OUT  = 1 * 1 * CONV_OUT_C;            // global avg-pool
  constexpr size_t SOFT_IN   = FC_OUT;

  std::vector<float> input(IN_SIZE);
  std::vector<float> conv_out(CONV_OUT);
  std::vector<float> pool_out(POOL_OUT);
  std::vector<float> fc_out(SOFT_IN);
  std::vector<float> probs(SOFT_IN);

  // fill input with random values 0-1
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> unif(0.f, 1.f);
  for (auto& v : input) v = unif(rng);

  // ------------------ op 1: Conv + ReLU -------------------------------------
  xnn_operator_t conv_op = nullptr;
  if (xnn_create_convolution2d_nhwc_f32(
        /*padding_t*/1, /*padding_b*/1,
        /*padding_l*/1, /*padding_r*/1,
        /*kernel_h*/CONV_K, /*kernel_w*/CONV_K,
        /*sub_h*/CONV_STR, /*sub_w*/CONV_STR,
        /*dilation_h*/1, /*dilation_w*/1,
        /*groups*/1,
        /*in_ch*/IN_C, /*in_stride*/IN_C,
        /*out_ch*/CONV_OUT_C, /*out_stride*/CONV_OUT_C,
        CONV_W, CONV_B,
        /*output min*/0.0f,                      // -> ReLU
        /*output max*/std::numeric_limits<float>::infinity(),
        /*flags*/0, &conv_op) != xnn_status_success) {
    std::cerr << "Conv create failed\n"; return 1;
  }
  xnn_setup_convolution2d_nhwc_f32(
      conv_op, BATCH, IN_H, IN_W, input.data(), conv_out.data(), nullptr);

  // ------------------ op 2: 28×28 AvgPool (global) --------------------------
  xnn_operator_t pool_op = nullptr;
  if (xnn_create_average_pooling2d_nhwc_f32(
        /*pad_t,b,l,r*/0, 0, 0, 0,
        /*kernel_h*/IN_H, /*kernel_w*/IN_W,
        /*sub_h*/IN_H, /*sub_w*/IN_W,
        /*in_ch=*/CONV_OUT_C, /*in_stride=*/CONV_OUT_C,
        /*out_stride=*/CONV_OUT_C,
        /*output min*/-std::numeric_limits<float>::infinity(),
        /*output max*/ std::numeric_limits<float>::infinity(),
        0, &pool_op) != xnn_status_success) {
    std::cerr << "Pool create failed\n"; return 1;
  }
  xnn_setup_average_pooling2d_nhwc_f32(
      pool_op, BATCH, IN_H, IN_W, conv_out.data(), pool_out.data(), nullptr);

  // ------------------ op 3: Fully-connected ---------------------------------
  xnn_operator_t fc_op = nullptr;
  if (xnn_create_fully_connected_nc_f32(
        /*in_ch*/CONV_OUT_C, /*out_ch*/FC_OUT,
        /*in_stride*/CONV_OUT_C, /*out_stride*/FC_OUT,
        FC_W, FC_B,
        -std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
        0, &fc_op) != xnn_status_success) {
    std::cerr << "FC create failed\n"; return 1;
  }
  xnn_setup_fully_connected_nc_f32(
      fc_op, BATCH, pool_out.data(), fc_out.data(), nullptr);

  // ------------------ op 4: Softmax -----------------------------------------
  xnn_operator_t sm_op = nullptr;
  if (xnn_create_softmax_nc_f32(
        /*ch=*/FC_OUT, /*ch_stride=*/FC_OUT,
        /*flags=*/0, &sm_op) != xnn_status_success) {
    std::cerr << "Softmax create failed\n"; return 1;
  }
  xnn_setup_softmax_nc_f32(
      sm_op, BATCH, fc_out.data(), probs.data(), nullptr);

  // ------------------ run pipeline ------------------------------------------
  for (xnn_operator_t op : {conv_op, pool_op, fc_op, sm_op})
    if (xnn_run_operator(op, nullptr) != xnn_status_success) {
      std::cerr << "Run failed\n"; return 1;
    }

  // ------------------ show result -------------------------------------------
  std::cout << "Probabilities: ";
  for (float p : probs) std::cout << p << ' ';
  std::cout << '\n';

  // ------------------ cleanup -----------------------------------------------
  xnn_delete_operator(conv_op);
  xnn_delete_operator(pool_op);
  xnn_delete_operator(fc_op);
  xnn_delete_operator(sm_op);
  return 0;
}
