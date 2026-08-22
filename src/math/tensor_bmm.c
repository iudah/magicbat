#include "matmul.h"
#include "matmul_batched_tile.h"
#include "matmul_tile.h"
#include "tensor.h"
#include "tensor_prot.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline void contiguous_lt_bs_unbatched(bool broadcast_a,
                                              bool broadcast_b, u32 batch,
                                              u32 row, u32 col, u32 com,
                                              Tensor tensor_a, Tensor tensor_b,
                                              Tensor product);
static inline void contiguous_gt_bs_unbatched(bool broadcast_a,
                                              bool broadcast_b, u32 batch,
                                              u32 row, u32 col, u32 com,
                                              Tensor tensor_a, Tensor tensor_b,
                                              Tensor product);
static inline void bmm_non_contiguous_bc_b_2(Tensor tensor_a, Tensor tensor_b,
                                             Tensor product, u32 batch, u32 row,
                                             u32 col, u32 com,
                                             bool UNUSED_ARG broadcast_a,
                                             bool broadcast_b);
static inline void bmm_non_contiguous_bc_a_2(Tensor tensor_a, Tensor tensor_b,
                                             Tensor product, u32 batch, u32 row,
                                             u32 col, u32 com, bool broadcast_a,
                                             bool UNUSED_ARG broadcast_b);
static inline void bmm_non_contiguous_bc_3(Tensor tensor_a, Tensor tensor_b,
                                           Tensor product, u32 batch, u32 row,
                                           u32 col, u32 com, bool broadcast_a,
                                           bool broadcast_b);
static inline void bmm_non_contiguous_no_bc(Tensor tensor_a, Tensor tensor_b,
                                            Tensor product, u32 batch, u32 row,
                                            u32 col, u32 com,
                                            bool UNUSED_ARG broadcast_a,
                                            bool UNUSED_ARG broadcast_b);
Tensor tensor_bmm(const Tensor tensor_a, const Tensor tensor_b) {

  if (bmm_tensors_invalid(tensor_a, tensor_b, false, false))
    return nullptr;

  auto broadcast_a =
      tensor_a->ndims == 2 || (tensor_a->shape[0] == 1 &&
                               tensor_b->shape[0] != 1 && tensor_b->ndims == 3);
  auto broadcast_b =
      tensor_b->ndims == 2 || (tensor_b->shape[0] == 1 &&
                               tensor_a->shape[0] != 1 && tensor_a->ndims == 3);

  u32 batch = broadcast_a && broadcast_b
                  ? 1
                  : (!broadcast_a ? tensor_a : tensor_b)->shape[0];
  u32 row = tensor_a->shape[tensor_a->ndims - 2];
  u32 col = tensor_b->shape[tensor_b->ndims - 1];
  u32 com = tensor_b->shape[tensor_b->ndims - 2];

  // guarantees zeros
  Tensor product = tensor_new(3, (u32[]){batch, row, col});
  if (product == nullptr)
    return nullptr;

  auto switch_flag = SWITCH_FLAG;

  switch (switch_flag) {
  case 1ULL << FULL_DIM_A | 1ULL << FULL_DIM_B | 1ULL << NO_BROADCAST:
    bmm_non_contiguous_no_bc(tensor_a, tensor_b, product, batch, row, col, com,
                             broadcast_a, broadcast_b);
    break;
  case 1ULL << FULL_DIM_A | 1ULL << FULL_DIM_B:
    bmm_non_contiguous_bc_3(tensor_a, tensor_b, product, batch, row, col, com,
                            broadcast_a, broadcast_b);
    break;
  case 1ULL << FULL_DIM_B:
    bmm_non_contiguous_bc_a_2(tensor_a, tensor_b, product, batch, row, col, com,
                              broadcast_a, broadcast_b);
    break;
  case 1ULL << FULL_DIM_A:
    bmm_non_contiguous_bc_b_2(tensor_a, tensor_b, product, batch, row, col, com,
                              broadcast_a, broadcast_b);
    break;
  case 1ULL << CONTIGUOUS_DUO | 1ULL << NO_BROADCAST | 1ULL << FULL_DIM_A |
      1ULL << FULL_DIM_B:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << FULL_DIM_A | 1ULL << FULL_DIM_B:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << FULL_DIM_B:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << FULL_DIM_A:
  case 1ULL << CONTIGUOUS_DUO:
    contiguous_lt_bs_unbatched(broadcast_a, broadcast_b, batch, row, col, com,
                               tensor_a, tensor_b, product);
    break;
  case 1ULL << CONTIGUOUS_DUO | 1ULL << NO_BROADCAST | 1ULL << FULL_DIM_A |
      1ULL << FULL_DIM_B | 1ULL << ALL_CAN_TILE:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << FULL_DIM_A | 1ULL << FULL_DIM_B |
      1ULL << ALL_CAN_TILE:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << FULL_DIM_B | 1ULL << ALL_CAN_TILE:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << FULL_DIM_A | 1ULL << ALL_CAN_TILE:
  case 1ULL << CONTIGUOUS_DUO | 1ULL << ALL_CAN_TILE:
    contiguous_gt_bs_unbatched(broadcast_a, broadcast_b, batch, row, col, com,
                               tensor_a, tensor_b, product);
    break;
  case 1ULL << CONTIGUOUS_DUO | 1ULL << NO_BROADCAST | 1ULL << FULL_DIM_A |
      1ULL << FULL_DIM_B | 1ULL << ALL_CAN_TILE | 1ULL << BATCH_CAN_TILE:
    bmatmul_gt_block_size(batch, row, col, com, tensor_a->data->data,
                          tensor_b->data->data, product->data->data);
    break;
  default:
    fprintf(stderr, "Unknown BMM combination.\n");
    abort();
  }

  return product;
}

void contiguous_lt_bs_unbatched(bool broadcast_a, bool broadcast_b, u32 batch,
                                u32 row, u32 col, u32 com, Tensor tensor_a,
                                Tensor tensor_b, Tensor product) {
  if (!broadcast_a && !broadcast_b) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_lt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (broadcast_a && !broadcast_b) {
    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_lt_block_size(
          row, col, com, tensor_a->data->data,
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (!broadcast_a && broadcast_b) {
    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_lt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          tensor_b->data->data,
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (broadcast_a && broadcast_b) {
    matmul_lt_block_size(row, col, com, tensor_a->data->data,
                         tensor_b->data->data, product->data->data);
  }
}

void contiguous_gt_bs_unbatched(bool broadcast_a, bool broadcast_b, u32 batch,
                                u32 row, u32 col, u32 com, Tensor tensor_a,
                                Tensor tensor_b, Tensor product) {
  if (!broadcast_a && !broadcast_b) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_gt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (broadcast_a && !broadcast_b) {
    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_gt_block_size(
          row, col, com, tensor_a->data->data,
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (!broadcast_a && broadcast_b) {
    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_gt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          tensor_b->data->data,
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (broadcast_a && broadcast_b) {
    matmul_gt_block_size(row, col, com, tensor_a->data->data,
                         tensor_b->data->data, product->data->data);
  }
}

static inline void bmm_non_contiguous_no_bc(Tensor tensor_a, Tensor tensor_b,
                                            Tensor product, u32 batch, u32 row,
                                            u32 col, u32 com,
                                            bool UNUSED_ARG broadcast_a,
                                            bool UNUSED_ARG broadcast_b) {
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[(batch_index * row + i) * col];
      for (u32 j = 0; j < com; ++j) {
        for (u32 k = 0; k < col; ++k) {
          auto a_val = tensor_get(tensor_a, (u32[MAX_DIMS]){batch_index, i, j});
          auto b_val = tensor_get(tensor_b, (u32[MAX_DIMS]){batch_index, j, k});
          p_data[k] += a_val * b_val;
        }
      }
    }
  }
}
static inline void bmm_non_contiguous_bc_3(Tensor tensor_a, Tensor tensor_b,
                                           Tensor product, u32 batch, u32 row,
                                           u32 col, u32 com, bool broadcast_a,
                                           bool broadcast_b) {
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[(batch_index * row + i) * col];
      for (u32 j = 0; j < com; ++j) {
        for (u32 k = 0; k < col; ++k) {
          auto a_val = tensor_get(
              tensor_a, (u32[MAX_DIMS]){broadcast_a ? 0 : batch_index, i, j});
          auto b_val = tensor_get(
              tensor_b, (u32[MAX_DIMS]){broadcast_b ? 0 : batch_index, j, k});
          p_data[k] += a_val * b_val;
        }
      }
    }
  }
}
static inline void bmm_non_contiguous_bc_a_2(Tensor tensor_a, Tensor tensor_b,
                                             Tensor product, u32 batch, u32 row,
                                             u32 col, u32 com, bool broadcast_a,
                                             bool UNUSED_ARG broadcast_b) {
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[(batch_index * row + i) * col];
      for (u32 j = 0; j < com; ++j) {
        for (u32 k = 0; k < col; ++k) {
          auto idx_a_0 = broadcast_a ? i : batch_index;
          auto idx_a_1 = broadcast_a ? j : i;
          auto idx_b_0 = batch_index;
          auto idx_b_1 = j;
          auto a_val =
              tensor_get(tensor_a, (u32[MAX_DIMS]){idx_a_0, idx_a_1, j});
          auto b_val =
              tensor_get(tensor_b, (u32[MAX_DIMS]){idx_b_0, idx_b_1, k});
          p_data[k] += a_val * b_val;
        }
      }
    }
  }
}
static inline void bmm_non_contiguous_bc_b_2(Tensor tensor_a, Tensor tensor_b,
                                             Tensor product, u32 batch, u32 row,
                                             u32 col, u32 com,
                                             bool UNUSED_ARG broadcast_a,
                                             bool broadcast_b) {
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[(batch_index * row + i) * col];
      for (u32 j = 0; j < com; ++j) {
        for (u32 k = 0; k < col; ++k) {
          auto idx_a_0 = batch_index;
          auto idx_a_1 = i;
          auto idx_b_0 = broadcast_b ? j : batch_index;
          auto idx_b_1 = broadcast_b ? k : j;
          auto a_val =
              tensor_get(tensor_a, (u32[MAX_DIMS]){idx_a_0, idx_a_1, j});
          auto b_val =
              tensor_get(tensor_b, (u32[MAX_DIMS]){idx_b_0, idx_b_1, k});
          p_data[k] += a_val * b_val;
        }
      }
    }
  }
}

Tensor tensor_bmm_wrt_a(const Tensor tensor_grad, const Tensor tensor_b) {
  return tensor_bmatmul_transpose_b(tensor_grad, tensor_b);
}

Tensor tensor_bmm_wrt_b(const Tensor tensor_a, const Tensor tensor_grad) {
  return tensor_bmatmul_transpose_a(tensor_a, tensor_grad);
}
