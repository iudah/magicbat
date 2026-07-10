#ifndef FUSED_ATTENTION_CAUSAL_BACKWARD_H
#define FUSED_ATTENTION_CAUSAL_BACKWARD_H

#include "matmul.h"
#include "matmul_tile.h"
#include <math.h>

static inline void fused_attention_causal_backward_gt_block_size_fast(
    u32 b_outer, u32 m_outer, u32 n_outer, u32 m_len, u32 n_len, u32 k_len,
    const f32 *restrict q_data, const f32 *restrict k_data,
    const f32 *restrict v_data, f32 *restrict dq_data, f32 *restrict dk_data,
    f32 *restrict dv_data, const f32 *restrict dr_data, f32 inv_sqrt_d,
    const f32 *restrict max, const f32 *restrict sum,
    const f32 *restrict row_sum) {

  MATMUL_INNER_LOOP(b) {
    MATMUL_INNER_LOOP(n) {
      MATMUL_INNER_LOOP(m) {
        const f32 *restrict max_data = &max[(b_inner * m_len) + m_inner];
        const f32 *restrict sum_data = &sum[(b_inner * m_len) + m_inner];
        f32 score = 0;
        bool mask = n_inner > m_inner;
        MATMUL_CAUSAL_LOOP(k, mask) {
          auto q_val = q_data[((b_inner * m_len + m_inner) * k_len) + k_indx];
          score +=
              q_val * k_data[((b_inner * n_len + n_inner) * k_len) + k_indx];
        }
        // normalize QK_T
        score = mask ? -INFINITY : score * inv_sqrt_d; // S_{m,n}
        auto e_score = expf(score - *max_data) / *sum_data;
        f32 de_score = 0;
        MATMUL_LOOP(k) {
          dv_data[((b_inner * n_len + n_inner) * k_len) + k_indx] +=
              e_score * dr_data[((b_inner * m_len + m_inner) * k_len) + k_indx];

          de_score += dr_data[((b_inner * m_len + m_inner) * k_len) + k_indx] *
                      v_data[((b_inner * n_len + n_inner) * k_len) + k_indx];
        }

        f32 dscore =
            e_score * (de_score - row_sum[(b_inner * m_len) + m_inner]);
        MATMUL_LOOP(k) {
          dq_data[((b_inner * m_len + m_inner) * k_len) + k_indx] +=
              dscore * k_data[((b_inner * n_len + n_inner) * k_len) + k_indx] *
              inv_sqrt_d;

          dk_data[((b_inner * n_len + n_inner) * k_len) + k_indx] +=
              dscore * q_data[((b_inner * m_len + m_inner) * k_len) + k_indx] *
              inv_sqrt_d;
        }
      }
    }
  }
}
static inline void fused_attention_causal_backward_gt_block_size(
    u32 b_len, u32 m_len, u32 n_len, u32 k_len, const f32 *q_data,
    const f32 *restrict k_data, const f32 *restrict v_data,
    f32 *restrict dq_data, f32 *restrict dk_data, f32 *restrict dv_data,
    const f32 *restrict dr_data, f32 sqrt_d, const f32 *restrict max_data,
    const f32 *restrict sum_data, const f32 *restrict row_sum) {

  f32 inv_sqrt_d = 1 / sqrt_d;

  MAIN_LEN(b);
  MAIN_LEN(m);
  MAIN_LEN(n);

  MATMUL_OUTER_LOOP(b) {
    MATMUL_OUTER_LOOP(m) {
      MATMUL_OUTER_LOOP(n) {

        fused_attention_causal_backward_gt_block_size_fast(
            b_outer, m_outer, n_outer, m_len, n_len, k_len, q_data, k_data,
            v_data, dq_data, dk_data, dv_data, dr_data, inv_sqrt_d, max_data,
            sum_data, row_sum);
      }
    }
  }
}

static inline void fused_attention_causal_backward_gt_block_size_unbatched_fast(
    u32 m_outer, u32 n_outer, u32 k_len, const f32 *restrict q_data,
    const f32 *restrict k_data, const f32 *restrict v_data,
    f32 *restrict dq_data, f32 *restrict dk_data, f32 *restrict dv_data,
    const f32 *restrict dr_data, f32 inv_sqrt_d, const f32 *restrict max,
    const f32 *restrict sum, const f32 *restrict row_sum) {

  MATMUL_INNER_LOOP(n) {
    MATMUL_INNER_LOOP(m) {
      const f32 *restrict max_data = &max[m_inner];
      const f32 *restrict sum_data = &sum[m_inner];
      f32 score = 0;
      bool mask = n_inner > m_inner;
      MATMUL_CAUSAL_LOOP(k, mask) {
        auto q_val = q_data[(m_inner * k_len) + k_indx];
        score += q_val * k_data[(n_inner * k_len) + k_indx];
      }
      // normalize QK_T
      score = mask ? -INFINITY : score * inv_sqrt_d; // S_{m,n}
      auto e_score = expf(score - *max_data) / *sum_data;
      f32 de_score = 0;
      MATMUL_LOOP(k) {
        dv_data[(n_inner * k_len) + k_indx] +=
            e_score * dr_data[(m_inner * k_len) + k_indx];

        de_score += dr_data[(m_inner * k_len) + k_indx] *
                    v_data[(n_inner * k_len) + k_indx];
      }

      f32 dscore = e_score * (de_score - row_sum[m_inner]);
      MATMUL_LOOP(k) {
        dq_data[(m_inner * k_len) + k_indx] +=
            dscore * k_data[(n_inner * k_len) + k_indx] * inv_sqrt_d;

        dk_data[(n_inner * k_len) + k_indx] +=
            dscore * q_data[(m_inner * k_len) + k_indx] * inv_sqrt_d;
      }
    }
  }
}

static inline void fused_attention_causal_backward_gt_block_size_unbatched(
    u32 m_len, u32 n_len, u32 k_len, const f32 *q_data,
    const f32 *restrict k_data, const f32 *restrict v_data,
    f32 *restrict dq_data, f32 *restrict dk_data, f32 *restrict dv_data,
    const f32 *restrict dr_data, f32 sqrt_d, const f32 *restrict max_data,
    const f32 *restrict sum_data, const f32 *restrict row_sum) {

  auto inv_sqrt_d = 1 / sqrt_d;

  MAIN_LEN(m);
  MAIN_LEN(n);

  MATMUL_OUTER_LOOP(m) {
    MATMUL_OUTER_LOOP(n) {

      fused_attention_causal_backward_gt_block_size_unbatched_fast(
          m_outer, n_outer, k_len, q_data, k_data, v_data, dq_data, dk_data,
          dv_data, dr_data, inv_sqrt_d, max_data, sum_data, row_sum);
    }
  }
}

static inline void fused_attention_causal_backward_lt_block_size(
    u32 m_len, u32 n_len, u32 k_len, const f32 *q_data,
    const f32 *restrict k_data, const f32 *restrict v_data,
    f32 *restrict dq_data, f32 *restrict dk_data, f32 *restrict dv_data,
    const f32 *restrict dr_data, f32 sqrt_d, const f32 *restrict max,
    const f32 *restrict sum, const f32 *restrict row_sum) {

  f32 inv_sqrt_d = 1 / sqrt_d;

  MATMUL_LOOP(n) {
    MATMUL_LOOP(m) {
      const f32 *restrict max_data = &max[m_indx];
      const f32 *restrict sum_data = &sum[m_indx];
      f32 score = 0;
      bool mask = n_indx > m_indx;
      MATMUL_CAUSAL_LOOP(k, mask) {
        auto q_val = q_data[(m_indx * k_len) + k_indx];
        score += q_val * k_data[(n_indx * k_len) + k_indx];
      }
      // normalize QK_T
      score = mask ? -INFINITY : score * inv_sqrt_d; // S_{m,n}
      auto e_score = expf(score - *max_data) / *sum_data;
      f32 de_score = 0;
      MATMUL_LOOP(k) {
        dv_data[(n_indx * k_len) + k_indx] +=
            e_score * dr_data[(m_indx * k_len) + k_indx];

        de_score += dr_data[(m_indx * k_len) + k_indx] *
                    v_data[(n_indx * k_len) + k_indx];
      }

      f32 dscore = e_score * (de_score - row_sum[m_indx]);
      MATMUL_LOOP(k) {
        dq_data[(m_indx * k_len) + k_indx] +=
            dscore * k_data[(n_indx * k_len) + k_indx] * inv_sqrt_d;

        dk_data[(n_indx * k_len) + k_indx] +=
            dscore * q_data[(m_indx * k_len) + k_indx] * inv_sqrt_d;
      }
    }
  }
}

#endif
