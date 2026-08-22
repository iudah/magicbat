#include "entropy.h"
#include "pcg_variants.h"
#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"
#include "type_alias.h"
#include <float.h>
#include <math.h>
#include <stdint.h>

#define RAND_SEED
#define BIT_SIZE 32

u64 urand() {
  static bool seeded = false;
  static pcg32_random_t rng;

  if (!seeded) {
    seeded = true;
#ifdef RAND_SEED
    uint64_t seeds[2];
    entropy_getbytes((void *)seeds, sizeof(seeds));
    pcg32_srandom_r(&rng, seeds[0], seeds[1]);
#else
    pcg32_srandom_r(&rng, 42u, 54u);

#endif
  }

  uint64_t high = pcg32_random_r(&rng);
  uint64_t low = pcg32_random_r(&rng);

  return (high << BIT_SIZE) | low;
}

void tensor_random(Tensor tensor) {
  f32 random_ratio;
  f32 value;
  if (tensor->is_contiguous) {
    for (u32 i = 0; i < tensor->data->nelements; ++i) {
      random_ratio = urand() / (f32)UINT64_MAX;
      tensor->data->data[i] =
          FLT_MIN + (FLT_MAX * random_ratio) - (FLT_MIN * random_ratio);
    }
  } else {
    u32 *index = tensor_odometer_new(tensor->ndims);
    do {
      random_ratio = urand() / (f32)UINT32_MAX;
      value = FLT_MIN + (FLT_MAX * random_ratio) - (FLT_MIN * random_ratio);
      tensor_set(tensor, index, value);
    } while (tensor_odometer_next(index, tensor->ndims, tensor->shape));
    tensor_odometer_destroy(index);
  }
}

void tensor_random_bound(Tensor tensor, f32 lower_bound, f32 upper_bound) {
  auto gap = upper_bound - lower_bound;
  f32 random_ratio;
  f32 value;
  if (tensor->is_contiguous) {
    for (u32 i = 0; i < tensor->data->nelements; ++i) {
      random_ratio = urand() / (f32)UINT64_MAX;
      tensor->data->data[i] = lower_bound + (gap * random_ratio);
    }
  } else {
    u32 *index = tensor_odometer_new(tensor->ndims);
    do {
      random_ratio = urand() / (f32)UINT64_MAX;
      value = lower_bound + (gap * random_ratio);
      tensor_set(tensor, index, value);
    } while (tensor_odometer_next(index, tensor->ndims, tensor->shape));
    tensor_odometer_destroy(index);
  }
}

void tensor_xavier(Tensor tensor, f32 fanin, f32 fanout) {
#define XAVIER_CONSTANT 6
  auto bound = sqrtf(XAVIER_CONSTANT / (fanin + fanout));
  tensor_random_bound(tensor, -bound, bound);
}

void tensor_kaiming(Tensor tensor, f32 fanin,
                    f32 __attribute__((unused)) fanout) {
#define KAIMING_CONSTANT 6
  auto bound = sqrtf(KAIMING_CONSTANT / (fanin));
  tensor_random_bound(tensor, -bound, bound);
}
