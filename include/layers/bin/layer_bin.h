#ifndef LAYER_BIN_H
#define LAYER_BIN_H

#include "type_alias.h"
#include <stdio.h>

#define MBAT_MAGIC_STR "MBAT"
#define MBAT_MAGIC_U32 0x4D424154u /* 'M','B','A','T' */
#define MBAT_VERSION   2

typedef struct {
  u8  magic[4];
  u16 version;
  /* 0x0102 = LE, 0x0201 = BE */
  u16 endian_tag;
  u32 header_bytes;
  /* bit0 = has_crc32, reserved */
  u32 flags;
} mb_bin_header;

typedef struct {
  FILE         *fp;
  mb_bin_header hdr;
} mb_bin_stream;

typedef enum serializer_enum {
  UNKNOWN_LAYER_SERIALIZER   = 0ULL,
  LINEAR_SERIALIZER          = 0x114EA7ULL,
  LAYERNORM_SERIALIZER       = 0x1A7E74073ULL,
  TRANSFORMER_ENC_SERIALIZER = 0x77A45F073E7ULL,
  RNNCELL_SERIALIZER         = 0x744CE11ULL,
  DROPOUT_SERIALIZER         = 0xD709ULL
} LayerSerializerId;

bool layer_write(LayerSerializerId serializer, mem layer, FILE *binary);
mem  layer_read(FILE *binary);
mb_bin_stream *layers_bin_create(const char *path, const char *bin_name,
                                 u32 version);
mb_bin_stream *layers_bin_open(const char *path, const char *bin_name);
bool           layers_bin_close(mb_bin_stream *bstream);

#endif // !LAYER_BINARY
