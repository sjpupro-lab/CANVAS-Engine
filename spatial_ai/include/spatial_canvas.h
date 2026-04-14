#ifndef SPATIAL_CANVAS_H
#define SPATIAL_CANVAS_H

#include "spatial_grid.h"

/*
 * Multi-tile canvas per SPEC.md §6 "Scaling structure".
 *
 *   One canvas holds up to 32 clause tiles in an 8×4 grid:
 *
 *     +-------+-------+-------+-------+-------+-------+-------+-------+
 *     |slot 0 |slot 1 |slot 2 |slot 3 |slot 4 |slot 5 |slot 6 |slot 7 |   y 0..255
 *     +-------+-------+-------+-------+-------+-------+-------+-------+
 *     |slot 8 |slot 9 |slot10 |slot11 |slot12 |slot13 |slot14 |slot15 |   y 256..511
 *     +-------+-------+-------+-------+-------+-------+-------+-------+
 *     |slot16 |slot17 |slot18 |slot19 |slot20 |slot21 |slot22 |slot23 |   y 512..767
 *     +-------+-------+-------+-------+-------+-------+-------+-------+
 *     |slot24 |slot25 |slot26 |slot27 |slot28 |slot29 |slot30 |slot31 |   y 768..1023
 *     +-------+-------+-------+-------+-------+-------+-------+-------+
 *       x0-255  256-   512-   768-   1024-  1280-  1536-  1792-2047
 *
 *   Placement order: left → right, top → bottom.
 *
 *   Why one 2048×1024 canvas instead of 32 independent 256×256 frames?
 *
 *   (1)  update_rgb diffusion crosses clause boundaries. Slot k's right
 *        edge sits next to slot k+1's left edge; B-channel horizontal
 *        diffusion naturally flows between adjacent clauses.
 *
 *   (2)  Delta RLE works. Within a single tile, changes between
 *        similar clauses are scattered; across a full canvas tiled
 *        with related clauses, identical Y-rows contain contiguous
 *        regions where RLE compresses well (SPEC §D).
 *
 *   (3)  Retrieval becomes spatial. A query tile can be compared
 *        against every slot, and R/G/B from neighbor slots inform
 *        the match (context from surrounding clauses).
 */

#define CV_TILE       256
#define CV_COLS       8
#define CV_ROWS       4
#define CV_SLOTS      (CV_COLS * CV_ROWS)       /* 32 */
#define CV_WIDTH      (CV_TILE * CV_COLS)       /* 2048 */
#define CV_HEIGHT     (CV_TILE * CV_ROWS)       /* 1024 */
#define CV_TOTAL      (CV_WIDTH * CV_HEIGHT)    /* 2,097,152 cells */

typedef struct {
    uint16_t* A;            /* 32-byte aligned, 2 bytes per cell */
    uint8_t*  R;
    uint8_t*  G;
    uint8_t*  B;
    uint32_t  width;        /* CV_WIDTH */
    uint32_t  height;       /* CV_HEIGHT */
    uint32_t  slot_count;   /* clauses placed so far (0 … CV_SLOTS) */
} SpatialCanvas;

/* Lifecycle */
SpatialCanvas* canvas_create(void);
void           canvas_destroy(SpatialCanvas* c);
void           canvas_clear(SpatialCanvas* c);

/* Placement:
 *   Encodes `text` into a 256×256 tile via layers_encode_clause and
 *   copies it into slot `c->slot_count`. Returns the slot index that
 *   was used, or -1 if the canvas is full.
 *   Note: caller typically runs canvas_update_rgb AFTER all slots are
 *   filled so cross-boundary diffusion has data to work with. */
int            canvas_add_clause(SpatialCanvas* c, const char* text);

/* Canvas-wide directional diffusion (SPEC §4) over width × height.
 *   R diagonal, G vertical, B horizontal — same α/β/γ as grid version. */
void           canvas_update_rgb(SpatialCanvas* c);

/* Stats */
uint32_t       canvas_active_count(const SpatialCanvas* c);
uint32_t       canvas_slot_byte_offset(uint32_t slot, uint32_t* out_x0,
                                       uint32_t* out_y0);

/* Matching:
 *   Compare a 256×256 query grid against the tile occupying `slot`.
 *   Uses RGB-weighted cosine over the tile's 256×256 sub-region. */
float          canvas_match_slot(const SpatialCanvas* c,
                                 const SpatialGrid* query, uint32_t slot);

/* Best slot over all populated slots. Returns slot index and writes
 *   similarity. */
uint32_t       canvas_best_slot(const SpatialCanvas* c,
                                const SpatialGrid* query,
                                float* out_sim);

/* Export a tile back into a 256×256 grid (allocates channels in out). */
void           canvas_slot_to_grid(const SpatialCanvas* c, uint32_t slot,
                                   SpatialGrid* out);

/* Compute sparse A-delta between two canvases.
 *   Returns number of changed cells; writes index + diff into entries.
 *   Useful for comparing canvas_a → canvas_b (RLE friendliness is
 *   measurable via canvas_delta_rle_bytes). */
typedef struct {
    uint32_t index;    /* y * CV_WIDTH + x */
    int16_t  diff_A;
} CanvasDeltaEntry;

uint32_t       canvas_delta_sparse(const SpatialCanvas* a,
                                   const SpatialCanvas* b,
                                   CanvasDeltaEntry* out, uint32_t max_out);

/* Estimate RLE byte cost of the sparse delta: consecutive-index runs
 *   compressed into (start, length, diff) triplets of 8 bytes each
 *   vs. 6 bytes per sparse entry. Returns RLE byte count. */
uint32_t       canvas_delta_rle_bytes(const CanvasDeltaEntry* entries,
                                      uint32_t count);

#endif /* SPATIAL_CANVAS_H */
