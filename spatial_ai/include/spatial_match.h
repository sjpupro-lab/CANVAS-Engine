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

/* ── Channel cascade matching ─────────────────────────────
 * Lego-block style staged matching: channels are combined in
 * ordered pairs rather than all at once. Top-K from one pair is
 * re-scored by another pair for the final match.
 *
 *   CASCADE_SEARCH    A-only baseline (overlap → cosine_a_only)
 *   CASCADE_QA        A → RG pair → BA rematch
 *                       R (diagonal/semantic) × G (vertical/substitution)
 *                       then fix top-K and re-score with
 *                       B (horizontal/co-occurrence) × A (activation).
 *   CASCADE_GENERATE  A → BG pair → RA rematch
 *                       B × G then R × A on top-K.
 *
 * Step 1 early-return: if the A-only cosine on the matched clause is
 * already ≥ CASCADE_STEP1_THRESHOLD (0.5), we've found a
 * structurally-identical clause and return immediately.
 */
typedef enum {
    CASCADE_SEARCH   = 0,
    CASCADE_QA       = 1,
    CASCADE_GENERATE = 2
} CascadeMode;

#define CASCADE_STEP1_THRESHOLD 0.5f

/* Forward declaration — full struct lives in spatial_keyframe.h.
 * Callers that invoke match_cascade must include spatial_keyframe.h
 * (which brings in the full SpatialAI definition).  C11 allows the
 * same typedef to appear in multiple headers. */
typedef struct SpatialAI_ SpatialAI;

/* Best-match cascade over ai->keyframes.
 * out_similarity:
 *   CASCADE_SEARCH       → A-only cosine of best match
 *   CASCADE_QA/GENERATE  → A-only cosine if step 1 fired; otherwise
 *                          RGB-weighted cosine of the final match. */
uint32_t match_cascade(
    SpatialAI* ai,
    SpatialGrid* input,
    CascadeMode mode,
    float* out_similarity
);

/* Top-K variant. Fills out_ids / out_scores (both capacity >= k) sorted
 * by final cascade score descending. Returns actual count written. */
uint32_t match_cascade_topk(
    SpatialAI* ai,
    SpatialGrid* input,
    CascadeMode mode,
    uint32_t k,
    uint32_t* out_ids,
    float* out_scores
);

/* Expose channel-pair scoring primitives (used by cascade, also useful
 * for tests and offline analysis). All iterate cells where BOTH a.A>0
 * and b.A>0. */
float rg_score(const SpatialGrid* a, const SpatialGrid* b);
float bg_score(const SpatialGrid* a, const SpatialGrid* b);
float ba_score(const SpatialGrid* a, const SpatialGrid* b);
float ra_score(const SpatialGrid* a, const SpatialGrid* b);

#endif /* SPATIAL_MATCH_H */
