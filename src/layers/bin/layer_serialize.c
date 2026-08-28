#include "layer_serialize.h"
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

FILE *layer_new_bin(char *path, char *bin_name, u32 version) {
#define path_len (2048)
  char bin_path[path_len] = {0};
  snprintf(bin_path, path_len, "%s/%s.mbat", path, bin_name);
  FILE *bin = fopen(bin_path, "w+b");

  if (!bin)
    return nullptr;

  u32 meta[META_SIZE] = {[0] = FILE_MEME, [1] = version, [META_SIZE - 1] = 0};
  fwrite(meta, sizeof(*meta), sizeof(meta) / sizeof(*meta), bin);
  return bin;
}

FILE *layer_load_bin(char *path, char *bin_name) {
  char bin_path[path_len] = {0};
  snprintf(bin_path, path_len, "%s/%s.mbat", path, bin_name);
  FILE *bin = fopen(bin_path, "rb");

  if (!bin)
    return nullptr;

  u32 meta[META_SIZE] = {0};
  fread(meta, sizeof(*meta), sizeof(meta) / sizeof(*meta), bin);

  if (meta[0] != FILE_MEME) {
    fclose(bin);
    return nullptr;
  }

  return bin;
}

bool layer_serialize(LayerSerializerId serializer, mem layer, FILE *binary) {
  fwrite(&serializer, sizeof(serializer), 1, binary);
  switch (serializer) {
  case LINEAR_SERIALIZER:
    return linear_layer_serialize(layer, binary);
  case LAYERNORM_SERIALIZER:
    return layer_norm_serialize(layer, binary);
  case TRANSFORMER_ENC_SERIALIZER:
    return transformer_enc_layer_serialize(layer, binary);
  case RNNCELL_SERIALIZER:
    return rnn_cell_serialize(layer, binary);
  default:
    fprintf(stderr, "Unknown serializer id: %llu", serializer);
    return false;
  }
}

mem layer_deserialize(FILE *binary) {
  enum serializer_enum serializer_id = 0;
  fread(&serializer_id, sizeof(serializer_id), 1, binary);
  switch (serializer_id) {
  case LINEAR_SERIALIZER:
    return linear_layer_deserialize(tmalloc(sizeof(struct linear_layer)),
                                    binary);
  case LAYERNORM_SERIALIZER:
    return layer_norm_deserialize(tmalloc(sizeof(struct layer_norm)), binary);
  case TRANSFORMER_ENC_SERIALIZER:
    return transformer_enc_layer_deserialize(
        tmalloc(sizeof(struct transfomer_enc_layer)), binary);
  case RNNCELL_SERIALIZER:
    return rnn_cell_deserialize(tmalloc(sizeof(struct rnn_cell)), binary);
  default:
    fprintf(stderr, "Unknown serializer id: %llu", serializer_id);
    return nullptr;
  }
}
