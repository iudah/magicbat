#ifndef MATMUL_BATCHED_TILE_H
#define MATMUL_BATCHED_TILE_H

#include "matmul.h"
#include "matmul_tile.h"
#include "tensor.h"
#include "type_alias.h"

#define CONTIGUOUS_DUO 0
#define NO_BROADCAST 1
#define ALL_CAN_TILE 2
#define FULL_DIM_A 3
#define FULL_DIM_B 4
#define BATCH_CAN_TILE 5
#define SWITCH_FLAG                                                            \
  (((u32)(tensor_a->is_contiguous && tensor_b->is_contiguous))                 \
   << CONTIGUOUS_DUO) |                                                        \
      ((u32)(!broadcast_a && !broadcast_b)) << NO_BROADCAST |                  \
      ((u32)(row >= BLOCK_SIZE && col >= BLOCK_SIZE && com >= BLOCK_SIZE))     \
          << ALL_CAN_TILE |                                                    \
      ((u32)tensor_a->ndims == 3) << FULL_DIM_A |                              \
      ((u32)tensor_b->ndims == 3) << FULL_DIM_B |                              \
      ((u32)batch >= BLOCK_SIZE) << BATCH_CAN_TILE

#define BMATMUL_FAST_PIPELINE(fn)                                              \
  static inline void fn##_fast(                                                \
      u32 b_outer, u32 m_outer, u32 n_outer, u32 k_outer, u32 m_len,           \
      u32 n_len, u32 k_len, const f32 *restrict m_data,                        \
      const f32 *restrict n_data, f32 *restrict r_data)

#define BMATMUL_FRINGE_PIPELINE(fn)                                            \
  static inline void fn##_fringe(                                              \
      u32 b_main, u32 m_main, u32 n_main, u32 k_main, u32 b_len, u32 m_len,    \
      u32 n_len, u32 k_len, const f32 *restrict m_data,                        \
      const f32 *restrict n_data, f32 *restrict r_data)

#define BMATMUL_M_VAL(type)                                                    \
  auto m_val = m_data[((b_##type * m_len + m_##type) * k_len) + k_##type];
#define BMATMUL_M_T_VAL(type)                                                  \
  auto m_val = m_data[((b_##type * k_len + k_##type) * m_len) + m_##type];
#define BMATMUL_N_VAL(type)                                                    \
  n_data[((b_##type * k_len + k_##type) * n_len) + n_##type]
#define BMATMUL_N_T_VAL(type)                                                  \
  n_data[((b_##type * n_len + n_##type) * k_len) + k_##type]

BMATMUL_FAST_PIPELINE(bmatmul_gt_block_size) {

  MATMUL_INNER_LOOP(b) {
    MATMUL_INNER_LOOP(m) {
      MATMUL_INNER_LOOP(k) {
        auto m_val = m_data[((b_inner * m_len + m_inner) * k_len) + k_inner];
        MATMUL_INNER_LOOP(n) {
          r_data[((b_inner * m_len + m_inner) * n_len) + n_inner] +=
              m_val * n_data[((b_inner * k_len + k_inner) * n_len) + n_inner];
        }
      }
    }
  }
}

BMATMUL_FRINGE_PIPELINE(bmatmul_gt_block_size) {
  // k
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_MAIN(m)
  MATMUL_INDEX_FRINGE(k) {
    BMATMUL_M_VAL(indx);

    MATMUL_INDEX_MAIN(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_MAIN(m)
  MATMUL_LOOP(k) {
    BMATMUL_M_VAL(indx);

    MATMUL_INDEX_FRINGE(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_FRINGE(m)
  MATMUL_LOOP(k) {
    BMATMUL_M_VAL(indx);

    MATMUL_LOOP(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
  // b
  MATMUL_INDEX_FRINGE(b)
  MATMUL_LOOP(m)
  MATMUL_LOOP(k) {
    BMATMUL_M_VAL(indx);

    MATMUL_LOOP(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
}

static inline f32 *bmatmul_gt_block_size(u32 b_len, u32 m_len, u32 n_len,
                                         u32 k_len, const f32 *m_data,
                                         const f32 *n_data, f32 *r_data) {

  MAIN_LEN(b);
  MAIN_LEN(m);
  MAIN_LEN(n);
  MAIN_LEN(k);

  MATMUL_OUTER_LOOP(b) {
    MATMUL_OUTER_LOOP(m) {
      MATMUL_OUTER_LOOP(k) {
        MATMUL_OUTER_LOOP(n) {

          bmatmul_gt_block_size_fast(b_outer, m_outer, n_outer, k_outer, m_len,
                                     n_len, k_len, m_data, n_data, r_data);
        }
      }
    }
  }

  bmatmul_gt_block_size_fringe(b_main, m_main, n_main, k_main, b_len, m_len,
                               n_len, k_len, m_data, n_data, r_data);

  return r_data;
}

BMATMUL_FAST_PIPELINE(bmatmul_transpose_a_gt_block_size) {

  MATMUL_INNER_LOOP(b) {
    MATMUL_INNER_LOOP(k) {
      MATMUL_INNER_LOOP(m) {
        auto m_val = m_data[((b_inner * k_len + k_inner) * m_len) + m_inner];
        MATMUL_INNER_LOOP(n) {
          r_data[((b_inner * m_len + m_inner) * n_len) + n_inner] +=
              m_val * n_data[((b_inner * k_len + k_inner) * n_len) + n_inner];
        }
      }
    }
  }
}

BMATMUL_FRINGE_PIPELINE(bmatmul_transpose_a_gt_block_size) {
  // k
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_FRINGE(k)
  MATMUL_INDEX_MAIN(m) {
    BMATMUL_M_T_VAL(indx);

    MATMUL_INDEX_MAIN(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(b)
  MATMUL_LOOP(k)
  MATMUL_INDEX_MAIN(m) {
    BMATMUL_M_T_VAL(indx);

    MATMUL_INDEX_FRINGE(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(b)
  MATMUL_LOOP(k)
  MATMUL_INDEX_FRINGE(m) {
    BMATMUL_M_T_VAL(indx);

    MATMUL_LOOP(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
  // b
  MATMUL_INDEX_FRINGE(b)
  MATMUL_LOOP(k)
  MATMUL_LOOP(m) {
    BMATMUL_M_T_VAL(indx);

    MATMUL_LOOP(n) {
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_VAL(indx);
    }
  }
}

static inline f32 *bmatmul_transpose_a_gt_block_size(u32 b_len, u32 m_len,
                                                     u32 n_len, u32 k_len,
                                                     const f32 *m_data,
                                                     const f32 *n_data,
                                                     f32 *r_data) {

  MAIN_LEN(b);
  MAIN_LEN(m);
  MAIN_LEN(n);
  MAIN_LEN(k);

  MATMUL_OUTER_LOOP(b) {
    MATMUL_OUTER_LOOP(k) {
      MATMUL_OUTER_LOOP(m) {
        MATMUL_OUTER_LOOP(n) {

          bmatmul_transpose_a_gt_block_size_fast(b_outer, m_outer, n_outer,
                                                 k_outer, m_len, n_len, k_len,
                                                 m_data, n_data, r_data);
        }
      }
    }
  }

  bmatmul_transpose_a_gt_block_size_fringe(b_main, m_main, n_main, k_main,
                                           b_len, m_len, n_len, k_len, m_data,
                                           n_data, r_data);

  return r_data;
}

BMATMUL_FAST_PIPELINE(bmatmul_transpose_b_gt_block_size) {

  MATMUL_INNER_LOOP(b) {
    MATMUL_INNER_LOOP(m) {
      MATMUL_INNER_LOOP(n) {
        MATMUL_INNER_LOOP(k) {
          auto m_val = m_data[((b_inner * m_len + m_inner) * k_len) + k_inner];
          r_data[((b_inner * m_len + m_inner) * n_len) + n_inner] +=
              m_val * n_data[((b_inner * n_len + n_inner) * k_len) + k_inner];
        }
      }
    }
  }
}

BMATMUL_FRINGE_PIPELINE(bmatmul_transpose_b_gt_block_size) {
  // k
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_MAIN(m)

  MATMUL_INDEX_MAIN(n) {
    MATMUL_INDEX_FRINGE(k) {
      BMATMUL_M_VAL(indx);
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_T_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_MAIN(m)

  MATMUL_INDEX_FRINGE(n) {
    MATMUL_LOOP(k) {
      BMATMUL_M_VAL(indx);
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_T_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(b)
  MATMUL_INDEX_FRINGE(m)

  MATMUL_LOOP(n) {
    MATMUL_LOOP(k) {
      BMATMUL_M_VAL(indx);
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_T_VAL(indx);
    }
  }
  // b
  MATMUL_INDEX_FRINGE(b)
  MATMUL_LOOP(m)

  MATMUL_LOOP(n) {
    MATMUL_LOOP(k) {
      BMATMUL_M_VAL(indx);
      r_data[((b_indx * m_len + m_indx) * n_len) + n_indx] +=
          m_val * BMATMUL_N_T_VAL(indx);
    }
  }
}

static inline f32 *bmatmul_transpose_b_gt_block_size(u32 b_len, u32 m_len,
                                                     u32 n_len, u32 k_len,
                                                     const f32 *m_data,
                                                     const f32 *n_data,
                                                     f32 *r_data) {

  MAIN_LEN(b);
  MAIN_LEN(m);
  MAIN_LEN(n);
  MAIN_LEN(k);

  MATMUL_OUTER_LOOP(b) {
    MATMUL_OUTER_LOOP(m) {
      MATMUL_OUTER_LOOP(n) {
        MATMUL_OUTER_LOOP(k) {

          bmatmul_transpose_b_gt_block_size_fast(b_outer, m_outer, n_outer,
                                                 k_outer, m_len, n_len, k_len,
                                                 m_data, n_data, r_data);
        }
      }
    }
  }

  bmatmul_transpose_b_gt_block_size_fringe(b_main, m_main, n_main, k_main,
                                           b_len, m_len, n_len, k_len, m_data,
                                           n_data, r_data);

  return r_data;
}

static inline bool bmm_tensors_invalid(const Tensor tensor_a,
                                       const Tensor tensor_b, bool transpose_a,
                                       bool transpose_b) {
  if (!tensor_a || !tensor_b)
    return true;

  auto com_a = tensor_a->shape[tensor_a->ndims - (transpose_a ? 2 : 1)];
  auto com_b = tensor_b->shape[tensor_b->ndims - (transpose_b ? 1 : 2)];
  auto valid_dim_a = tensor_a->ndims == 2 || tensor_a->ndims == 3;
  auto valid_dim_b = tensor_b->ndims == 2 || tensor_b->ndims == 3;
  auto valid_pair = tensor_a->ndims == 3 && tensor_b->ndims == 3
                        ? tensor_a->shape[0] == tensor_b->shape[0] ||
                              tensor_a->shape[0] == 1 || tensor_b->shape[0] == 1
                        : true;
  return !(com_a == com_b && valid_dim_a && valid_dim_b && valid_pair);
}

#endif
