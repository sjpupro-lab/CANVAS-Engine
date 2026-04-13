#ifndef SPATIAL_MATCH_H
#define SPATIAL_MATCH_H

#include "spatial_grid.h"
#include <math.h>

/* Directional RGB update parameters */
#define ALPHA_R 0.05f
#define BETA_G  0.08f
#define GAMMA_B 0.03f

/* Top-K and matching constants */
#define TOP_K 8
#define MAX_CANDIDATES 4096
#define BUCKET_THRESHOLD 100
#define NUM_BUCKETS 256

/* Matching candidate */
typedef struct {
    uint32_t id;
    float    score;
} Candidate;

/* Block summary for 16x16 block skip (SPEC-ENGINE Phase B) */
#define BLOCK 16
#define BLOCKS 16

typedef struct {
    uint32_t sum[BLOCKS][BLOCKS];
} BlockSummary;

/* Hash bucket for large-scale search (SPEC-ENGINE Phase C) */
typedef struct {
    uint32_t ids[256];
    uint32_t count;
} Bucket;

typedef struct {
    Bucket buckets[NUM_BUCKETS];
} BucketIndex;

/* ── Directional RGB ── */

/* Update RGB channels with directional diffusion */
void update_rgb_directional(SpatialGrid* grid);

/* ── Overlap (Coarse filter) ── */

/* Count active pixels that overlap between two grids */
uint32_t overlap_score(const SpatialGrid* a, const SpatialGrid* b);

/* ── Cosine similarity ── */

/* RGB weight factor for a single pixel pair */
float rgb_weight(uint8_t r1, uint8_t r2,
                 uint8_t g1, uint8_t g2,
                 uint8_t b1, uint8_t b2);

/* RGB-weighted cosine similarity */
float cosine_rgb_weighted(const SpatialGrid* a, const SpatialGrid* b);

/* A-channel only cosine similarity */
float cosine_a_only(const SpatialGrid* a, const SpatialGrid* b);

/* ── Block skip cosine (SPEC-ENGINE Phase B) ── */

/* Compute block sums for a grid */
void compute_block_sums(const SpatialGrid* g, BlockSummary* bs);

/* Cosine with block skip optimization */
float cosine_block_skip(const SpatialGrid* a, const SpatialGrid* b,
                        const BlockSummary* bs_a, const BlockSummary* bs_b);

/* ── Top-K selection ── */

/* Select top-k candidates in-place (partial sort) */
void topk_select(Candidate* pool, uint32_t pool_size, uint32_t k);

/* ── Hash bucket (SPEC-ENGINE Phase C) ── */

/* Compute grid hash based on active X coordinates */
uint32_t grid_hash(const SpatialGrid* g);

/* Initialize bucket index */
void bucket_index_init(BucketIndex* idx);

/* Add a keyframe to bucket index */
void bucket_index_add(BucketIndex* idx, const SpatialGrid* g, uint32_t kf_id);

/* Get candidates from adjacent buckets */
void bucket_candidates(BucketIndex* idx, uint32_t hash,
                       int expand, uint32_t* out, uint32_t* out_count);

#endif /* SPATIAL_MATCH_H */
