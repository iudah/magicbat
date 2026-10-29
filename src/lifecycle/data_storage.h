#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include "tensor_memory.h"
#include "tensor_prot.h"
#include <stdatomic.h>
#include <stdio.h>

static inline data_storage *data_storage_new(u32 nelements) {

  if ((u64)nelements * sizeof(f32) > UINT32_MAX)
    return nullptr;

  data_storage *data = tmalloc(sizeof(*data));
  if (!data)
    return nullptr;
  data->nelements = nelements;
  data->data      = nelements ? tcalloc(nelements, sizeof(f32)) : nullptr;
  data->refcount  = 1;

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

static inline data_storage *data_storage_read(FILE *binary) {

  data_storage *data = tmalloc(sizeof(*data));
  if (!data)
    return nullptr;
  fread(&data->nelements, sizeof(data->nelements), 1, binary);
  data->data =
      data->nelements ? tmalloc(data->nelements * sizeof(f32)) : nullptr;
  if (data->data)
    fread(data->data, sizeof(*data->data), data->nelements, binary);
  data->refcount = 1;

  return data;
}

static inline bool data_storage_write(data_storage *data, FILE *binary) {

  fwrite(&data->nelements, sizeof(data->nelements), 1, binary);
  fwrite(data->data, sizeof(*data->data), data->nelements, binary);

  return true;
}

static inline bool data_storage_write_offset(data_storage *data, u64 offset,
                                             FILE *binary) {
  if (offset >= data->nelements)
    return false;
  u32 remaining = data->nelements - offset;
  fwrite(&remaining, sizeof(remaining), 1, binary);
  fwrite(data->data + offset, sizeof(*data->data), remaining, binary);
  return true;
}

#endif
