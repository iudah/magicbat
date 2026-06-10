#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "matmul.h"
#include "type_alias.h"

#define BLOCK_SIZE 16

#define MATMUL_OUTER_LOOP(var_name)                                            \
  for (u32 var_name##_outer = 0; var_name##_outer < var_name##_main;           \
       var_name##_outer += BLOCK_SIZE)

#define MATMUL_INNER_LOOP(var_name)                                            \
  for (u32 var_name##_inner = var_name##_outer;                                \
       var_name##_inner < var_name##_outer + BLOCK_SIZE; ++var_name##_inner)

#define MATMUL_INDEX_MAIN(var_name)                                            \
  for (u32 var_name##_indx = 0; var_name##_indx < var_name##_main;             \
       ++var_name##_indx)

#define MATMUL_INDEX_FRINGE(var_name)                                          \
  for (u32 var_name##_indx = var_name##_main;                                  \
       var_name##_indx < var_name##_len; ++var_name##_indx)

#define MAIN_LEN(var) u32 var##_main = var##_len & ~(BLOCK_SIZE - 1)

#define MATMUL_FAST_PIPELINE(fn)                                               \
  static inline void fn##_fast(                                                \
      u32 m_main, u32 n_main, u32 k_main, u32 m_len, u32 n_len, u32 k_len,     \
      const f32 *restrict m_data, const f32 *restrict n_data,                  \
      f32 *restrict r_data)

#define MATMUL_FRINGE_PIPELINE(fn)                                             \
  static inline void fn##_fringe(                                              \
      u32 m_main, u32 n_main, u32 k_main, u32 m_len, u32 n_len, u32 k_len,     \
      const f32 *restrict m_data, const f32 *restrict n_data,                  \
      f32 *restrict r_data)

#define MATMUL_M_VAL(type) auto m_val = m_data[(m_##type * k_len) + k_##type];
#define MATMUL_M_T_VAL(type) auto m_val = m_data[(k_##type * m_len) + m_##type];
#define MATMUL_N_VAL(type) n_data[(k_##type * n_len) + n_##type]
#define MATMUL_N_T_VAL(type) n_data[(n_##type * k_len) + k_##type]

MATMUL_FAST_PIPELINE(matmul_gt_block_size) {
  (void)m_len;

  MATMUL_OUTER_LOOP(m) {
    MATMUL_OUTER_LOOP(k) {
      MATMUL_OUTER_LOOP(n) {

        MATMUL_INNER_LOOP(m) {
          MATMUL_INNER_LOOP(k) {
            MATMUL_M_VAL(inner);

            MATMUL_INNER_LOOP(n) {
              r_data[(m_inner * n_len) + n_inner] +=
                  m_val * MATMUL_N_VAL(inner);
            }
          }
        }
      }
    }
  }
}

MATMUL_FRINGE_PIPELINE(matmul_gt_block_size) {
  // k
  MATMUL_INDEX_MAIN(m)
  MATMUL_INDEX_FRINGE(k) {
    MATMUL_M_VAL(indx);

    MATMUL_INDEX_MAIN(n) {
      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(m)
  MATMUL_LOOP(k) {
    MATMUL_M_VAL(indx);

    MATMUL_INDEX_FRINGE(n) {
      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_FRINGE(m)
  MATMUL_LOOP(k) {
    MATMUL_M_VAL(indx);

    MATMUL_LOOP(n) {
      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_VAL(indx);
    }
  }
}

static inline f32 *matmul_gt_block_size(u32 m_len, u32 n_len, u32 k_len,
                                        const f32 *m_data, const f32 *n_data,
                                        f32 *r_data) {

  MAIN_LEN(m);
  MAIN_LEN(n);
  MAIN_LEN(k);

  matmul_gt_block_size_fast(m_main, n_main, k_main, m_len, n_len, k_len, m_data,
                            n_data, r_data);
  matmul_gt_block_size_fringe(m_main, n_main, k_main, m_len, n_len, k_len,
                              m_data, n_data, r_data);

  return r_data;
}

MATMUL_FAST_PIPELINE(matmul_transpose_a_gt_block_size) {
  (void)k_len;

  MATMUL_OUTER_LOOP(k) {
    MATMUL_OUTER_LOOP(m) {
      MATMUL_OUTER_LOOP(n) {

        MATMUL_INNER_LOOP(k) {
          MATMUL_INNER_LOOP(m) {
            MATMUL_M_T_VAL(inner);

            MATMUL_INNER_LOOP(n) {
              r_data[(m_inner * n_len) + n_inner] +=
                  m_val * MATMUL_N_VAL(inner);
            }
          }
        }
      }
    }
  }
}

MATMUL_FRINGE_PIPELINE(matmul_transpose_a_gt_block_size) {
  // k
  MATMUL_INDEX_FRINGE(k)
  MATMUL_INDEX_MAIN(m) {
    MATMUL_M_T_VAL(indx);

    MATMUL_INDEX_MAIN(n) {
      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(m)
  MATMUL_LOOP(k) {
    MATMUL_M_T_VAL(indx);

    MATMUL_INDEX_FRINGE(n) {
      r_data[(m_indx * n_len) + n_indx] +=
          m_val * n_data[(k_indx * n_len) + n_indx];
    }
  }
  // n
  MATMUL_INDEX_FRINGE(m)
  MATMUL_LOOP(k) {
    MATMUL_M_T_VAL(indx);

    MATMUL_LOOP(n) {
      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_VAL(indx);
    }
  }
}

static inline f32 *matmul_transpose_a_gt_block_size(u32 m_len, u32 n_len,
                                                    u32 k_len,
                                                    const f32 *m_data,
                                                    const f32 *n_data,
                                                    f32 *r_data) {

  MAIN_LEN(m);
  MAIN_LEN(n);
  MAIN_LEN(k);

  matmul_transpose_a_gt_block_size_fast(m_main, n_main, k_main, m_len, n_len,
                                        k_len, m_data, n_data, r_data);
  matmul_transpose_a_gt_block_size_fringe(m_main, n_main, k_main, m_len, n_len,
                                          k_len, m_data, n_data, r_data);

  return r_data;
}

MATMUL_FAST_PIPELINE(matmul_transpose_b_gt_block_size) {
  (void)m_len;

  MATMUL_OUTER_LOOP(m) {
    MATMUL_OUTER_LOOP(n) {
      MATMUL_OUTER_LOOP(k) {

        MATMUL_INNER_LOOP(m) {
          MATMUL_INNER_LOOP(n) {
            MATMUL_INNER_LOOP(k) {

              MATMUL_M_VAL(inner);
              auto n_val = MATMUL_N_T_VAL(inner);
              r_data[(m_inner * n_len) + n_inner] += m_val * n_val;
            }
          }
        }
      }
    }
  }
}

MATMUL_FRINGE_PIPELINE(matmul_transpose_b_gt_block_size) {
  // k
  MATMUL_INDEX_MAIN(m)
  MATMUL_INDEX_MAIN(n) {
    MATMUL_INDEX_FRINGE(k) {
      MATMUL_M_VAL(indx);

      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_T_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_MAIN(m)
  MATMUL_INDEX_FRINGE(n) {
    MATMUL_LOOP(k) {
      MATMUL_M_VAL(indx);

      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_T_VAL(indx);
    }
  }
  // n
  MATMUL_INDEX_FRINGE(m)
  MATMUL_LOOP(n) {
    MATMUL_LOOP(k) {
      MATMUL_M_VAL(indx);
      r_data[(m_indx * n_len) + n_indx] += m_val * MATMUL_N_T_VAL(indx);
    }
  }
}

static inline f32 *matmul_transpose_b_gt_block_size(u32 m_len, u32 n_len,
                                                    u32 k_len,
                                                    const f32 *m_data,
                                                    const f32 *n_data,
                                                    f32 *r_data) {

  MAIN_LEN(m);
  MAIN_LEN(n);
  MAIN_LEN(k);

  matmul_transpose_b_gt_block_size_fast(m_main, n_main, k_main, m_len, n_len,
                                        k_len, m_data, n_data, r_data);
  matmul_transpose_b_gt_block_size_fringe(m_main, n_main, k_main, m_len, n_len,
                                          k_len, m_data, n_data, r_data);

  return r_data;
}

#endif
