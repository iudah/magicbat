#ifndef BMM_DISPATCH_H
#define BMM_DISPATCH_H

#include "matmul_tile.h"
#include "tensor.h"
#include "type_alias.h"

typedef f32 *(*bmm_batched_tile_fn)(u32, u32, u32, u32, const f32 *,
                                    const f32 *, f32 *);

typedef void (*bmm_unbatched_fn)(bool, bool, u32, u32, u32, u32, Tensor, Tensor,
                                 Tensor);

typedef void (*bm_non_contiguous_fn)(Tensor, Tensor, Tensor, u32, u32, u32, u32,
                                     bool, bool);

static inline void bmm_dispatch_contiguous(bmm_batched_tile_fn batched,
                                           bmm_unbatched_fn    lt_batch_sz,
                                           bmm_unbatched_fn    gt_batch_sz,
                                           bool broadcast_a, bool broadcast_b,
                                           u32 batch, u32 row, u32 col, u32 com,
                                           Tensor tensor_a, Tensor tensor_b,
                                           Tensor product) {
  const bool no_broadcast = !broadcast_a && !broadcast_b;
  const bool all_can_tile =
      row >= BLOCK_SIZE && col >= BLOCK_SIZE && com >= BLOCK_SIZE;
  const bool batch_can_tile = batch >= BLOCK_SIZE;

  if (no_broadcast && all_can_tile && batch_can_tile) {
    batched(batch, row, col, com, tensor_a->data->data, tensor_b->data->data,
            product->data->data);
  } else if (all_can_tile) {
    gt_batch_sz(broadcast_a, broadcast_b, batch, row, col, com, tensor_a,
                tensor_b, product);
  } else {
    lt_batch_sz(broadcast_a, broadcast_b, batch, row, col, com, tensor_a,
                tensor_b, product);
  }
}

static inline void bmm_dispatch_non_contiguous(
    bm_non_contiguous_fn no_bc, bm_non_contiguous_fn bc_3,
    bm_non_contiguous_fn bc_a_2, bm_non_contiguous_fn bc_b_2, bool broadcast_a,
    bool broadcast_b, u32 batch, u32 row, u32 col, u32 com, Tensor tensor_a,
    Tensor tensor_b, Tensor product) {
  const bool no_broadcast = !broadcast_a && !broadcast_b;
  const bool full_a       = tensor_a->ndims == 3;
  const bool full_b       = tensor_b->ndims == 3;

  if (full_a && full_b) {
    if (no_broadcast)
      no_bc(tensor_a, tensor_b, product, batch, row, col, com, broadcast_a,
            broadcast_b);
    else
      bc_3(tensor_a, tensor_b, product, batch, row, col, com, broadcast_a,
           broadcast_b);
  } else if (full_b) {
    bc_a_2(tensor_a, tensor_b, product, batch, row, col, com, broadcast_a,
           broadcast_b);
  } else if (full_a) {
    bc_b_2(tensor_a, tensor_b, product, batch, row, col, com, broadcast_a,
           broadcast_b);
  } else {
    no_bc(tensor_a, tensor_b, product, batch, row, col, com, broadcast_a,
          broadcast_b);
  }
}

#endif
