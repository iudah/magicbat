#ifndef FUSED_ATTENTION_CAUSAL_H
#define FUSED_ATTENTION_CAUSAL_H

#include "matmul.h"
#include "matmul_tile.h"
#include <math.h>

static inline void fused_attention_causal_gt_block_size_fast(
    u32 b_outer, u32 m_outer, u32 n_outer, u32 m_len, u32 n_len, u32 k_len,
    const f32 *restrict q_data, const f32 *restrict k_data,
    const f32 *restrict v_data, f32 *restrict r_data, f32 inv_sqrt_d, f32 *max,
    f32 *sum) {

  MATMUL_INNER_LOOP(b) {
    MATMUL_INNER_LOOP(m) {
      f32 *max_data = &max[(b_inner * m_len) + m_inner];
      f32 *sum_data = &sum[(b_inner * m_len) + m_inner];
      MATMUL_INNER_LOOP(n) {
        f32 score = 0;
        bool mask = n_inner > m_inner;
        MATMUL_CAUSAL_LOOP(k, mask) {
          auto m_val = q_data[((b_inner * m_len + m_inner) * k_len) + k_indx];
          score +=
              m_val * k_data[((b_inner * n_len + n_inner) * k_len) + k_indx];
        }
        // normalize QK_T
        score = mask ? -INFINITY : score * inv_sqrt_d;
        // update softtmax stability max
        auto new_max = fmaxf(score, *max_data);
        auto e_score = expf(score - new_max);
        // account for deviation from actual max
        auto alpha = expf(*max_data - new_max);
        // e^{x-max_{t-1}}* e^{max_t-1}-max_{t}}   hifts the eaarlier sum to
        // the correct t exp
        *sum_data = (*sum_data * alpha) + e_score;
        *max_data = new_max;
        MATMUL_LOOP(k) {
          r_data[((b_inner * m_len + m_inner) * k_len) + k_indx] =
              (r_data[((b_inner * m_len + m_inner) * k_len) + k_indx] * alpha) +
              (v_data[((b_inner * n_len + n_inner) * k_len) + k_indx] *
               e_score);
        }
      }
    }
  }
}

static inline f32 *fused_attention_causal_gt_block_size(
    u32 b_len, u32 m_len, u32 n_len, u32 k_len, const f32 *q_data,
    const f32 *k_data, const f32 *v_data, f32 *r_data, f32 inv_sqrt_d,
    f32 *max_data, f32 *sum_data) {

  MAIN_LEN(b);
  MAIN_LEN(m);
  MAIN_LEN(n);

  MATMUL_OUTER_LOOP(b) {
    MATMUL_OUTER_LOOP(m) {
      MATMUL_OUTER_LOOP(n) {

        fused_attention_causal_gt_block_size_fast(
            b_outer, m_outer, n_outer, m_len, n_len, k_len, q_data, k_data,
            v_data, r_data, inv_sqrt_d, max_data, sum_data);
      }
      MATMUL_INNER_LOOP(b) {
        MATMUL_INNER_LOOP(m) {
          f32 sum_val = sum_data[(b_inner * m_len) + m_inner];
          MATMUL_LOOP(k) {

            r_data[((b_inner * m_len) + m_inner * k_len) + k_indx] /= sum_val;
          }
        }
      }
    }
  }

  return r_data;
}

static inline void fused_attention_causal_gt_block_size_unbatched_fast(
    u32 m_outer, u32 n_outer, u32 k_len, const f32 *restrict q_data,
    const f32 *restrict k_data, const f32 *restrict v_data,
    f32 *restrict r_data, f32 inv_sqrt_d, f32 *max, f32 *sum) {

  MATMUL_INNER_LOOP(m) {
    f32 *max_data = &max[m_inner];
    f32 *sum_data = &sum[m_inner];
    MATMUL_INNER_LOOP(n) {
      f32 score = 0;
      bool mask = n_inner > m_inner;
      MATMUL_CAUSAL_LOOP(k, mask) {
        auto m_val = q_data[(m_inner * k_len) + k_indx];
        score += m_val * k_data[(n_inner * k_len) + k_indx];
      }
      // normalize QK_T
      score = mask ? -INFINITY : score * inv_sqrt_d;
      // update softtmax stability max
      auto new_max = fmaxf(score, *max_data);
      auto e_score = expf(score - new_max);
      // account for deviation from actual max
      auto alpha = expf(*max_data - new_max);
      // e^{x-max_{t-1}}* e^{max_t-1}-max_{t}}   hifts the eaarlier sum to
      // the correct t exp
      *sum_data = (*sum_data * alpha) + e_score;
      *max_data = new_max;
      MATMUL_LOOP(k) {
        r_data[(m_inner * k_len) + k_indx] =
            (r_data[(m_inner * k_len) + k_indx] * alpha) +
            (v_data[(n_inner * k_len) + k_indx] * e_score);
      }
    }
  }
}

static inline f32 *fused_attention_causal_gt_block_size_unbatched(
    u32 m_len, u32 n_len, u32 k_len, const f32 *q_data, const f32 *k_data,
    const f32 *v_data, f32 *r_data, f32 inv_sqrt_d, f32 *max_data,
    f32 *sum_data) {

  MAIN_LEN(m);
  MAIN_LEN(n);

  MATMUL_OUTER_LOOP(m) {
    MATMUL_OUTER_LOOP(n) {

      fused_attention_causal_gt_block_size_unbatched_fast(
          m_outer, n_outer, k_len, q_data, k_data, v_data, r_data, inv_sqrt_d,
          max_data, sum_data);
    }
    MATMUL_INNER_LOOP(m) {
      f32 sum_val = sum_data[m_inner];
      MATMUL_LOOP(k) { r_data[(m_inner * k_len) + k_indx] /= sum_val; }
    }
  }

  return r_data;
}

static inline f32 *fused_attention_causal_lt_block_size(
    u32 m_len, u32 n_len, u32 k_len, const f32 *q_data, const f32 *k_data,
    const f32 *v_data, f32 *r_data, f32 inv_sqrt_d, f32 *max_data,
    f32 *sum_data) {

  MATMUL_LOOP(m) {
    f32 *max = &max_data[m_indx];
    f32 *sum = &sum_data[m_indx];
    MATMUL_LOOP(n) {
      f32 score = 0;
      bool mask = n_indx > m_indx;
      MATMUL_CAUSAL_LOOP(k, mask) {
        auto m_val = q_data[(m_indx * k_len) + k_indx];
        score += m_val * k_data[(n_indx * k_len) + k_indx];
      }
      // normalize QK_T
      score = mask ? -INFINITY : score * inv_sqrt_d;
      // update softtmax stability max
      auto new_max = fmaxf(score, *max);
      auto e_score = expf(score - new_max);
      // account for deviation from actual max
      auto alpha = expf(*max - new_max);
      // e^{x-max_{t-1}}* e^{max_t-1}-max_{t}}   hifts the eaarlier sum to
      // the correct t exp
      *sum = (*sum * alpha) + e_score;
      *max = new_max;
      MATMUL_LOOP(k) {
        r_data[(m_indx * k_len) + k_indx] =
            (r_data[(m_indx * k_len) + k_indx] * alpha) +
            (v_data[(n_indx * k_len) + k_indx] * e_score);
      }
    }
    f32 sum_val = sum_data[m_indx];
    MATMUL_LOOP(k) { r_data[(m_indx * k_len) + k_indx] /= sum_val; }
  }
  // MATMUL_LOOP(m) {
  //  f32 sum_val = sum_data[m_indx];
  //  MATMUL_LOOP(k) { r_data[(m_indx * k_len) + k_indx] /= sum_val; }
  // }

  return r_data;
}

#endif
