#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include "../../include/adt/tensor/tensor_prot.h"
#include "tensor_memory.h"
#include <stdatomic.h>
#include <stdio.h>

static inline data_storage *data_storage_new(u32 nelements) {

  data_storage *data = tmalloc(sizeof(*data));
  data->nelements = nelements;
  data->data = nelements ? tcalloc(nelements, sizeof(f32)) : nullptr;
  data->refcount = 1;

  return data;
}

static inline bool data_storage_destroy(data_storage *data) {
  if (!data)
    return false;

  if (atomic_fetch_sub(&data->refcount, 1) != 1)
    return true;

  tfree(data->data);
  tfree(data);

  return true;
}

static inline data_storage *data_storage_deserialize(FILE *binary) {

  data_storage *data = tmalloc(sizeof(*data));
  fread(&data->nelements, sizeof(data->nelements), 1, binary);
  data->data =
      data->nelements ? tmalloc(data->nelements * sizeof(f32)) : nullptr;
  if (data->data)
    fread(data->data, sizeof(*data->data), data->nelements, binary);
  data->refcount = 1;

  return data;
}

static inline bool data_storage_serialize(data_storage *data, FILE *binary) {

  fwrite(&data->nelements, sizeof(data->nelements), 1, binary);
  fwrite(data->data, sizeof(*data->data), data->nelements, binary);

  return true;
}

#endif
