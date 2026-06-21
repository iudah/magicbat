#ifndef MATMUL_H
#define MATMUL_H

#include "type_alias.h"

#define MATMUL_LOOP(var_name)                                                  \
  for (u32 var_name##_indx = 0; var_name##_indx < var_name##_len;              \
       ++var_name##_indx)

static inline f32 *matmul_lt_block_size(u32 m_len, u32 n_len, u32 k_len,
                                        const f32 *m_data, const f32 *n_data,
                                        f32 *r_data) {
  MATMUL_LOOP(m) {
    MATMUL_LOOP(k) {
      auto m_val = m_data[(m_indx * k_len) + k_indx];
      MATMUL_LOOP(n) {
        r_data[(m_indx * n_len) + n_indx] +=
            m_val * n_data[(k_indx * n_len) + n_indx];
      }
    }
  }

  return r_data;
}

static inline f32 *matmul_transpose_a_lt_block_size(u32 m_len, u32 n_len,
                                                    u32 k_len,
                                                    const f32 *m_data,
                                                    const f32 *n_data,
                                                    f32 *r_data) {
  MATMUL_LOOP(m) {
    MATMUL_LOOP(k) {
      auto m_val = m_data[(k_indx * m_len) + m_indx];
      MATMUL_LOOP(n) {
        r_data[(m_indx * n_len) + n_indx] +=
            m_val * n_data[(k_indx * n_len) + n_indx];
      }
    }
  }

  return r_data;
}

static inline f32 *matmul_transpose_b_lt_block_size(u32 m_len, u32 n_len,
                                                    u32 k_len,
                                                    const f32 *m_data,
                                                    const f32 *n_data,
                                                    f32 *r_data) {
  MATMUL_LOOP(k) {
    MATMUL_LOOP(n) {
      auto n_val = n_data[(n_indx * k_len) + k_indx];
      MATMUL_LOOP(m) {
        auto m_val = m_data[(m_indx * k_len) + k_indx];
        r_data[(m_indx * n_len) + n_indx] += m_val * n_val;
      }
    }
  }

  return r_data;
}

#endif
