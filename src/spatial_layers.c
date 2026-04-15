#include "spatial_layers.h"
#include <string.h>

static int is_space_byte(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int pos_is_content(PartOfSpeech pos) {
    return pos == POS_NOUN || pos == POS_VERB || pos == POS_ADJ || pos == POS_UNKNOWN;
}

static void seed_rg_token(SpatialGrid* grid, uint32_t idx, PartOfSpeech pos) {
    uint8_t r_seed = 0;
    uint8_t g_seed = 0;

    switch (pos) {
        case POS_NOUN:     r_seed = 40;  g_seed = 30;  break;
        case POS_VERB:     r_seed = 120; g_seed = 40;  break;
        case POS_ADJ:      r_seed = 170; g_seed = 35;  break;
        case POS_PARTICLE: r_seed = 8;   g_seed = 85;  break;
        case POS_ENDING:   r_seed = 12;  g_seed = 95;  break;
        case POS_PUNCT:    r_seed = 5;   g_seed = 120; break;
        case POS_UNKNOWN:  r_seed = 210; g_seed = 20;  break;
    }

    if (grid->R[idx] == 0) grid->R[idx] = r_seed;
    else grid->R[idx] = (uint8_t)(((uint16_t)grid->R[idx] + (uint16_t)r_seed) / 2u);

    if (grid->G[idx] == 0) grid->G[idx] = g_seed;
    else grid->G[idx] = (uint8_t)(((uint16_t)grid->G[idx] + (uint16_t)g_seed) / 2u);
}

LayerBitmaps* layers_create(void) {
    LayerBitmaps* lb = (LayerBitmaps*)malloc(sizeof(LayerBitmaps));
    if (!lb) return NULL;
    memset(lb, 0, sizeof(LayerBitmaps));
    return lb;
}

void layers_destroy(LayerBitmaps* lb) {
    free(lb);
}

/* Encode raw UTF-8 bytes into a 1D layer array with given weight */
static void layer_encode_bytes(const uint8_t* bytes, uint32_t len,
                               uint16_t* layer, uint16_t weight) {
    for (uint32_t i = 0; i < len; i++) {
        uint32_t x = bytes[i];
        uint32_t y = i % GRID_SIZE;
        uint32_t idx = y * GRID_SIZE + x;
        uint32_t new_val = (uint32_t)layer[idx] + weight;
        layer[idx] = (new_val > 65535) ? 65535 : (uint16_t)new_val;
    }
}

/* Split text by spaces and encode each word (with its position offset) */
static void layer_encode_words(const char* text, uint16_t* layer, uint16_t weight) {
    const uint8_t* bytes = (const uint8_t*)text;
    uint32_t total_len = (uint32_t)strlen(text);
    uint32_t pos = 0;

    while (pos < total_len) {
        /* Skip spaces */
        while (pos < total_len && bytes[pos] == ' ') pos++;
        if (pos >= total_len) break;

        /* Find word end */
        uint32_t word_start = pos;
        while (pos < total_len && bytes[pos] != ' ') pos++;

        /* Encode word bytes at their original positions */
        for (uint32_t i = word_start; i < pos; i++) {
            uint32_t x = bytes[i];
            uint32_t y = i % GRID_SIZE;
            uint32_t idx = y * GRID_SIZE + x;
            uint32_t new_val = (uint32_t)layer[idx] + weight;
            layer[idx] = (new_val > 65535) ? 65535 : (uint16_t)new_val;
        }
    }
}

/* Encode morphemes: analyze each word, then encode morpheme tokens
   at their original byte positions */
static void layer_encode_morphemes(const char* text, uint16_t* layer, uint16_t weight) {
    const uint8_t* bytes = (const uint8_t*)text;
    uint32_t total_len = (uint32_t)strlen(text);
    uint32_t pos = 0;

    while (pos < total_len) {
        while (pos < total_len && is_space_byte(bytes[pos])) pos++;
        if (pos >= total_len) break;

        uint32_t word_start = pos;
        while (pos < total_len && !is_space_byte(bytes[pos])) pos++;
        uint32_t word_end = pos;
        uint32_t word_len = word_end - word_start;
        if (word_len == 0 || word_len >= 255) continue;

        char word[256];
        memcpy(word, text + word_start, word_len);
        word[word_len] = '\0';

        Morpheme morphs[32];
        uint32_t n = morpheme_analyze(word, morphs, 32);
        uint32_t local = 0;

        for (uint32_t m = 0; m < n; m++) {
            uint32_t tlen = (uint32_t)strlen(morphs[m].token);
            if (tlen == 0 || tlen > word_len) continue;

            uint32_t found = UINT32_MAX;
            if (local + tlen <= word_len &&
                memcmp(word + local, morphs[m].token, tlen) == 0) {
                found = local;
            } else {
                for (uint32_t j = local; j + tlen <= word_len; j++) {
                    if (memcmp(word + j, morphs[m].token, tlen) == 0) {
                        found = j;
                        break;
                    }
                }
            }
            if (found == UINT32_MAX) continue;

            if (pos_is_content(morphs[m].pos)) {
                for (uint32_t i = 0; i < tlen; i++) {
                    uint32_t bi = word_start + found + i;
                    uint32_t x = bytes[bi];
                    uint32_t y = bi % GRID_SIZE;
                    uint32_t idx = y * GRID_SIZE + x;
                    uint32_t new_val = (uint32_t)layer[idx] + weight;
                    layer[idx] = (new_val > 65535) ? 65535 : (uint16_t)new_val;
                }
            }
            local = found + tlen;
        }
    }
}

/* Seed R/G channels from morpheme POS at original byte positions.
   This gives tile-level semantic/function priors before directional diffusion. */
static void seed_morpheme_rg(const char* text, SpatialGrid* out_combined) {
    const uint8_t* bytes = (const uint8_t*)text;
    uint32_t total_len = (uint32_t)strlen(text);
    uint32_t pos = 0;

    while (pos < total_len) {
        while (pos < total_len && is_space_byte(bytes[pos])) pos++;
        if (pos >= total_len) break;

        uint32_t word_start = pos;
        while (pos < total_len && !is_space_byte(bytes[pos])) pos++;
        uint32_t word_end = pos;
        uint32_t word_len = word_end - word_start;
        if (word_len == 0 || word_len >= 255) continue;

        char word[256];
        memcpy(word, text + word_start, word_len);
        word[word_len] = '\0';

        Morpheme morphs[32];
        uint32_t n = morpheme_analyze(word, morphs, 32);
        uint32_t local = 0;

        for (uint32_t m = 0; m < n; m++) {
            uint32_t tlen = (uint32_t)strlen(morphs[m].token);
            if (tlen == 0 || tlen > word_len) continue;

            uint32_t found = UINT32_MAX;
            if (local + tlen <= word_len &&
                memcmp(word + local, morphs[m].token, tlen) == 0) {
                found = local;
            } else {
                for (uint32_t j = local; j + tlen <= word_len; j++) {
                    if (memcmp(word + j, morphs[m].token, tlen) == 0) {
                        found = j;
                        break;
                    }
                }
            }
            if (found == UINT32_MAX) continue;

            for (uint32_t i = 0; i < tlen; i++) {
                uint32_t bi = word_start + found + i;
                uint32_t x = bytes[bi];
                uint32_t y = bi % GRID_SIZE;
                uint32_t idx = y * GRID_SIZE + x;
                if (out_combined->A[idx] > 0) {
                    seed_rg_token(out_combined, idx, morphs[m].pos);
                }
            }
            local = found + tlen;
        }
    }
}

void layers_encode_clause(const char* clause_text,
                          LayerBitmaps* out_layers,
                          SpatialGrid* out_combined) {
    if (!clause_text || !out_combined) return;

    LayerBitmaps local_layers;
    LayerBitmaps* lb = out_layers ? out_layers : &local_layers;
    memset(lb, 0, sizeof(LayerBitmaps));

    const uint8_t* bytes = (const uint8_t*)clause_text;
    uint32_t len = (uint32_t)strlen(clause_text);

    /* Layer 1: Base layer — all bytes, weight +1 */
    layer_encode_bytes(bytes, len, lb->base, 1);

    /* Layer 2: Word layer — space-separated words, weight +5 */
    layer_encode_words(clause_text, lb->word, 5);

    /* Layer 3: Morpheme layer — morpheme tokens, weight +3 */
    layer_encode_morphemes(clause_text, lb->morpheme, 3);

    /* Sum into combined grid: A = base + word + morpheme */
    grid_clear(out_combined);
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        uint32_t sum = (uint32_t)lb->base[i] + lb->word[i] + lb->morpheme[i];
        out_combined->A[i] = (sum > 65535) ? 65535 : (uint16_t)sum;
    }

    /* Seed R/G before diffusion, based on morpheme POS alignment.
     * B is intentionally left untouched here: the bitmap pattern (byte value =
     * X, byte order = Y, 3-layer sum = brightness) is already a unique
     * fingerprint. A per-clause B hash would stamp over that pattern rather
     * than refine it, so we keep B to the directional diffusion path only. */
    seed_morpheme_rg(clause_text, out_combined);
}
