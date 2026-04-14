#include "spatial_canvas.h"
#include "spatial_layers.h"
#include "spatial_match.h"
#include "spatial_morpheme.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
static void* cv_aligned(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static void cv_aligned_free(void* p) { _aligned_free(p); }
#else
static void* cv_aligned(size_t alignment, size_t size) {
    void* p = NULL;
    posix_memalign(&p, alignment, size);
    return p;
}
static void cv_aligned_free(void* p) { free(p); }
#endif

/* ── Lifecycle ─────────────────────────────────────────── */

SpatialCanvas* canvas_create(void) {
    SpatialCanvas* c = (SpatialCanvas*)calloc(1, sizeof(SpatialCanvas));
    if (!c) return NULL;

    c->A = (uint16_t*)cv_aligned(32, CV_TOTAL * sizeof(uint16_t));
    c->R = (uint8_t*) cv_aligned(32, CV_TOTAL);
    c->G = (uint8_t*) cv_aligned(32, CV_TOTAL);
    c->B = (uint8_t*) cv_aligned(32, CV_TOTAL);

    if (!c->A || !c->R || !c->G || !c->B) {
        canvas_destroy(c);
        return NULL;
    }
    memset(c->A, 0, CV_TOTAL * sizeof(uint16_t));
    memset(c->R, 0, CV_TOTAL);
    memset(c->G, 0, CV_TOTAL);
    memset(c->B, 0, CV_TOTAL);
    c->width  = CV_WIDTH;
    c->height = CV_HEIGHT;
    c->slot_count = 0;
    return c;
}

void canvas_destroy(SpatialCanvas* c) {
    if (!c) return;
    if (c->A) cv_aligned_free(c->A);
    if (c->R) cv_aligned_free(c->R);
    if (c->G) cv_aligned_free(c->G);
    if (c->B) cv_aligned_free(c->B);
    free(c);
}

void canvas_clear(SpatialCanvas* c) {
    if (!c) return;
    memset(c->A, 0, CV_TOTAL * sizeof(uint16_t));
    memset(c->R, 0, CV_TOTAL);
    memset(c->G, 0, CV_TOTAL);
    memset(c->B, 0, CV_TOTAL);
    c->slot_count = 0;
}

/* ── Slot → canvas-space coordinate helpers ────────────── */

uint32_t canvas_slot_byte_offset(uint32_t slot, uint32_t* out_x0, uint32_t* out_y0) {
    uint32_t col = slot % CV_COLS;
    uint32_t row = slot / CV_COLS;
    uint32_t x0  = col * CV_TILE;
    uint32_t y0  = row * CV_TILE;
    if (out_x0) *out_x0 = x0;
    if (out_y0) *out_y0 = y0;
    return y0 * CV_WIDTH + x0;
}

/* ── Tile placement ────────────────────────────────────── */

int canvas_add_clause(SpatialCanvas* c, const char* text) {
    if (!c || !text || c->slot_count >= CV_SLOTS) return -1;

    /* Encode into a temporary 256×256 tile */
    SpatialGrid* tile = grid_create();
    if (!tile) return -1;
    morpheme_init();
    layers_encode_clause(text, NULL, tile);
    /* Note: we intentionally skip update_rgb_directional on the tile;
       diffusion happens canvas-wide after placement (see
       canvas_update_rgb). B is already seeded with the clause's
       co-occurrence hash by layers_encode_clause. */

    /* Copy tile into the slot */
    uint32_t slot = c->slot_count;
    uint32_t x0, y0;
    canvas_slot_byte_offset(slot, &x0, &y0);

    for (uint32_t dy = 0; dy < CV_TILE; dy++) {
        for (uint32_t dx = 0; dx < CV_TILE; dx++) {
            uint32_t ti = dy * CV_TILE + dx;
            uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
            c->A[ci] = tile->A[ti];
            c->R[ci] = tile->R[ti];
            c->G[ci] = tile->G[ti];
            c->B[ci] = tile->B[ti];
        }
    }

    grid_destroy(tile);
    c->slot_count++;
    return (int)slot;
}

/* ── Canvas-wide RGB directional diffusion ─────────────── */

void canvas_update_rgb(SpatialCanvas* c) {
    if (!c) return;
    uint32_t W = c->width;
    uint32_t H = c->height;

    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            uint32_t i = y * W + x;
            if (c->A[i] == 0) continue;

            /* R: diagonal neighbours (semantic / morpheme relation) */
            static const int dx[4] = {1, 1, -1, -1};
            static const int dy[4] = {1, -1, 1, -1};
            for (int d = 0; d < 4; d++) {
                int nx = (int)x + dx[d], ny = (int)y + dy[d];
                if (nx < 0 || nx >= (int)W || ny < 0 || ny >= (int)H) continue;
                uint32_t ni = (uint32_t)ny * W + (uint32_t)nx;
                if (c->A[ni] == 0) continue;
                int diff = (int)c->R[ni] - (int)c->R[i];
                int new_v = (int)c->R[i] + (int)(ALPHA_R * diff);
                if (new_v < 0) new_v = 0;
                if (new_v > 255) new_v = 255;
                c->R[i] = (uint8_t)new_v;
            }
            /* G: vertical (substitution) — crosses tile row boundaries */
            for (int d = -1; d <= 1; d += 2) {
                int ny = (int)y + d;
                if (ny < 0 || ny >= (int)H) continue;
                uint32_t ni = (uint32_t)ny * W + x;
                if (c->A[ni] == 0) continue;
                int diff = (int)c->G[ni] - (int)c->G[i];
                int new_v = (int)c->G[i] + (int)(BETA_G * diff);
                if (new_v < 0) new_v = 0;
                if (new_v > 255) new_v = 255;
                c->G[i] = (uint8_t)new_v;
            }
            /* B: horizontal (clause order) — crosses tile column boundaries */
            for (int d = -1; d <= 1; d += 2) {
                int nx = (int)x + d;
                if (nx < 0 || nx >= (int)W) continue;
                uint32_t ni = y * W + (uint32_t)nx;
                if (c->A[ni] == 0) continue;
                int diff = (int)c->B[ni] - (int)c->B[i];
                int new_v = (int)c->B[i] + (int)(GAMMA_B * diff);
                if (new_v < 0) new_v = 0;
                if (new_v > 255) new_v = 255;
                c->B[i] = (uint8_t)new_v;
            }
        }
    }
}

/* ── Stats ─────────────────────────────────────────────── */

uint32_t canvas_active_count(const SpatialCanvas* c) {
    if (!c) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < CV_TOTAL; i++) if (c->A[i] > 0) n++;
    return n;
}

/* ── Slot → grid export ────────────────────────────────── */

void canvas_slot_to_grid(const SpatialCanvas* c, uint32_t slot, SpatialGrid* out) {
    if (!c || !out || slot >= CV_SLOTS) return;
    uint32_t x0, y0;
    canvas_slot_byte_offset(slot, &x0, &y0);
    for (uint32_t dy = 0; dy < CV_TILE; dy++) {
        for (uint32_t dx = 0; dx < CV_TILE; dx++) {
            uint32_t ti = dy * CV_TILE + dx;
            uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
            out->A[ti] = c->A[ci];
            out->R[ti] = c->R[ci];
            out->G[ti] = c->G[ci];
            out->B[ti] = c->B[ci];
        }
    }
}

/* ── Slot matching ─────────────────────────────────────── */

float canvas_match_slot(const SpatialCanvas* c, const SpatialGrid* query, uint32_t slot) {
    if (!c || !query || slot >= CV_SLOTS) return 0.0f;
    uint32_t x0, y0;
    canvas_slot_byte_offset(slot, &x0, &y0);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t dy = 0; dy < CV_TILE; dy++) {
        for (uint32_t dx = 0; dx < CV_TILE; dx++) {
            uint32_t qi = dy * CV_TILE + dx;
            uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
            double qa = (double)query->A[qi];
            double ca = (double)c->A[ci];
            if (qa > 0.0 && ca > 0.0) {
                /* RGB per-cell weight (same formula as rgb_weight) */
                double dr = fabs((double)query->R[qi] - c->R[ci]) / 255.0;
                double dg = fabs((double)query->G[qi] - c->G[ci]) / 255.0;
                double db = fabs((double)query->B[qi] - c->B[ci]) / 255.0;
                double w  = 1.0 - (0.5*dr + 0.3*dg + 0.2*db);
                if (w < 0.0) w = 0.0;
                dot += qa * ca * w;
            }
            na += qa * qa;
            nb += ca * ca;
        }
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

uint32_t canvas_best_slot(const SpatialCanvas* c, const SpatialGrid* query, float* out_sim) {
    if (!c || !query || c->slot_count == 0) {
        if (out_sim) *out_sim = 0.0f;
        return 0;
    }
    uint32_t best_slot = 0;
    float    best_sim  = -1.0f;
    for (uint32_t s = 0; s < c->slot_count; s++) {
        float v = canvas_match_slot(c, query, s);
        if (v > best_sim) { best_sim = v; best_slot = s; }
    }
    if (out_sim) *out_sim = best_sim;
    return best_slot;
}

/* ── Delta (sparse + RLE byte estimate) ────────────────── */

uint32_t canvas_delta_sparse(const SpatialCanvas* a, const SpatialCanvas* b,
                             CanvasDeltaEntry* out, uint32_t max_out) {
    if (!a || !b || !out) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < CV_TOTAL && n < max_out; i++) {
        int16_t diff = (int16_t)b->A[i] - (int16_t)a->A[i];
        if (diff != 0) {
            out[n].index  = i;
            out[n].diff_A = diff;
            n++;
        }
    }
    return n;
}

uint32_t canvas_delta_rle_bytes(const CanvasDeltaEntry* entries, uint32_t count) {
    if (!entries || count == 0) return 0;
    /* RLE record: (start:u32, length:u16, diff:i16) = 8 bytes per run.
       A run = consecutive indices where diff_A is identical. */
    uint32_t runs = 1;
    uint32_t prev_idx  = entries[0].index;
    int16_t  prev_diff = entries[0].diff_A;
    for (uint32_t k = 1; k < count; k++) {
        if (entries[k].index == prev_idx + 1 && entries[k].diff_A == prev_diff) {
            /* extend current run */
        } else {
            runs++;
        }
        prev_idx  = entries[k].index;
        prev_diff = entries[k].diff_A;
    }
    return runs * 8;
}
