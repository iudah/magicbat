#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

const u32 *tensor_shape(Tensor t) { return t->shape; }
