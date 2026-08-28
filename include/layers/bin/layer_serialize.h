#ifndef LAYER_BINARY
#define LAYER_BINARY

#include "type_alias.h"
#include <stdio.h>

typedef enum serializer_enum {
  UNKNOWN_LAYER_SERIALIZER = 0ULL,
  LINEAR_SERIALIZER = 0x114EA7ULL,
  LAYERNORM_SERIALIZER = 0x1A7E74073ULL,
  TRANSFORMER_ENC_SERIALIZER = 0x77A45F073E7ULL,
  RNNCELL_SERIALIZER = 0x744CE11ULL
} LayerSerializerId;
bool layer_serialize(LayerSerializerId serializer, mem layer, FILE *binary);
mem layer_deserialize(FILE *binary);
FILE *layer_new_bin(char *path, char *bin_name, u32 version);
FILE *layer_load_bin(char *path, char *bin_name);

#endif // !LAYER_BINARY
