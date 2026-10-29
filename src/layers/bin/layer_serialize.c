#include "dropout/dropout_layer.h"
#include "layer_bin.h"
#include "layer_norm.h"
#include "layers_prot.h"
#include "linear_layer.h"
#include "rnn_cell.h"
#include "tensor_memory.h"
#include "transformer_enc_layer.h"
#include "type_alias.h"
#include <stdint.h>
#include <stdio.h>

#define FILE_MEME 0x3A61C8A7
#define META_SIZE (16)
#define path_len  (2048)

mb_bin_stream *layers_bin_create(const char *path, const char *bin_name,
                                 u32 version) {
  char bin_path[path_len] = {0};
  snprintf(bin_path, path_len, "%s/%s.mbat", path, bin_name);
  mb_bin_stream *bins = tcalloc(1, sizeof(*bins));
  if (!bins)
    return nullptr;

  FILE *bin = fopen(bin_path, "w+b");

  if (!bin) {
    tfree(bins);
    return nullptr;
  }

  bins->fp = bin;

  u32 meta[META_SIZE] = {[0] = FILE_MEME, [1] = version, [META_SIZE - 1] = 0};
  fwrite(meta, sizeof(*meta), sizeof(meta) / sizeof(*meta), bin);
  return bins;
}

mb_bin_stream *layers_bin_open(const char *path, const char *bin_name) {
  char bin_path[path_len] = {0};
  snprintf(bin_path, path_len, "%s/%s.mbat", path, bin_name);
  mb_bin_stream *bins = tcalloc(1, sizeof(*bins));
  if (!bins)
    return nullptr;

  FILE *bin = fopen(bin_path, "rb");

  if (!bin) {
    tfree(bins);
    return nullptr;
  }

  bins->fp = bin;

  u32 meta[META_SIZE] = {0};
  fread(meta, sizeof(*meta), sizeof(meta) / sizeof(*meta), bin);

  if (meta[0] != FILE_MEME) {
    fclose(bin);
    tfree(bins);
    return nullptr;
  }

  return bins;
}

bool layers_bin_close(mb_bin_stream *bstream) {
  if (!bstream || !bstream->fp)
    return false;
  if (fclose(bstream->fp) != 0)
    return false;
  tfree(bstream);
  return true;
}
bool layer_write(LayerSerializerId serializer, mem layer, FILE *binary) {
  fwrite(&serializer, sizeof(serializer), 1, binary);
  switch (serializer) {
  case LINEAR_SERIALIZER:
    return linear_layer_write(layer, binary);
  case LAYERNORM_SERIALIZER:
    return layer_norm_write(layer, binary);
  case TRANSFORMER_ENC_SERIALIZER:
    return transformer_enc_layer_write(layer, binary);
  case RNNCELL_SERIALIZER:
    return rnn_cell_write(layer, binary);
  case DROPOUT_SERIALIZER:
    return dropout_layer_write(layer, binary);
  default:
    fprintf(stderr, "Unknown serializer id: %llu", serializer);
    return false;
  }
}

mem layer_read(FILE *binary) {
  enum serializer_enum serializer_id = 0;
  fread(&serializer_id, sizeof(serializer_id), 1, binary);
  mem layer = nullptr;
  switch (serializer_id) {
  case LINEAR_SERIALIZER:
    layer = tcalloc(1, sizeof(struct linear_layer));
    break;
  case LAYERNORM_SERIALIZER:
    layer = tcalloc(1, sizeof(struct layer_norm));
    break;
  case TRANSFORMER_ENC_SERIALIZER:
    layer = tcalloc(1, sizeof(struct transfomer_enc_layer));
    break;
  case RNNCELL_SERIALIZER:
    layer = tcalloc(1, sizeof(struct rnn_cell));
    break;
  case DROPOUT_SERIALIZER:
    layer = tcalloc(1, sizeof(struct dropout_layer));
    break;
  default:
    fprintf(stderr, "Unknown serializer id: %llu", serializer_id);
    return nullptr;
  }

  if (!layer)
    return nullptr;

  switch (serializer_id) {
  case LINEAR_SERIALIZER:
    return linear_layer_read(layer, binary);
  case LAYERNORM_SERIALIZER:
    return layer_norm_read(layer, binary);
  case TRANSFORMER_ENC_SERIALIZER:
    return transformer_enc_layer_read(layer, binary);
  case RNNCELL_SERIALIZER:
    return rnn_cell_read(layer, binary);
  case DROPOUT_SERIALIZER:
    return dropout_layer_read(layer, binary);
  default:
    fprintf(stderr, "Unknown serializer id: %llu", serializer_id);
    return nullptr;
  }
}
