#include "spatial_match.h"

/* ── Directional RGB update (§9.2) ── */

void update_rgb_directional(SpatialGrid* grid) {
    if (!grid) return;

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            uint32_t idx = (uint32_t)(y * GRID_SIZE + x);
            if (grid->A[idx] == 0) continue;

            /* R: diagonal (morpheme/semantic) */
            int dx[4] = {1, 1, -1, -1};
            int dy[4] = {1, -1, 1, -1};
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                    uint32_t nidx = (uint32_t)(ny * GRID_SIZE + nx);
                    if (grid->A[nidx] > 0) {
                        int diff = (int)grid->R[nidx] - (int)grid->R[idx];
                        int delta = (int)(ALPHA_R * diff);
                        int new_val = (int)grid->R[idx] + delta;
                        if (new_val < 0) new_val = 0;
                        if (new_val > 255) new_val = 255;
                        grid->R[idx] = (uint8_t)new_val;
                    }
                }
            }

            /* G: vertical (word substitution) */
            for (int d = -1; d <= 1; d += 2) {
                int ny = y + d;
                if (ny >= 0 && ny < GRID_SIZE) {
                    uint32_t nidx = (uint32_t)(ny * GRID_SIZE + x);
                    if (grid->A[nidx] > 0) {
                        int diff = (int)grid->G[nidx] - (int)grid->G[idx];
                        int delta = (int)(BETA_G * diff);
                        int new_val = (int)grid->G[idx] + delta;
                        if (new_val < 0) new_val = 0;
                        if (new_val > 255) new_val = 255;
                        grid->G[idx] = (uint8_t)new_val;
                    }
                }
            }

            /* B: horizontal (clause order) */
            for (int d = -1; d <= 1; d += 2) {
                int nx = x + d;
                if (nx >= 0 && nx < GRID_SIZE) {
                    uint32_t nidx = (uint32_t)(y * GRID_SIZE + nx);
                    if (grid->A[nidx] > 0) {
                        int diff = (int)grid->B[nidx] - (int)grid->B[idx];
                        int delta = (int)(GAMMA_B * diff);
                        int new_val = (int)grid->B[idx] + delta;
                        if (new_val < 0) new_val = 0;
                        if (new_val > 255) new_val = 255;
                        grid->B[idx] = (uint8_t)new_val;
                    }
                }
            }
        }
    }
}

/* ── Overlap score (Coarse filter §9.3) ── */

uint32_t overlap_score(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a->A[i] > 0 && b->A[i] > 0) count++;
    }
    return count;
}

/* ── RGB weight (§9.4) ── */

float rgb_weight(uint8_t r1, uint8_t r2,
                 uint8_t g1, uint8_t g2,
                 uint8_t b1, uint8_t b2) {
    float dr = fabsf((float)r1 - (float)r2) / 255.0f;
    float dg = fabsf((float)g1 - (float)g2) / 255.0f;
    float db = fabsf((float)b1 - (float)b2) / 255.0f;
    return 1.0f - (0.5f * dr + 0.3f * dg + 0.2f * db);
}

/* ── A-channel only cosine ── */

float cosine_a_only(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        double va = (double)a->A[i];
        double vb = (double)b->A[i];
        dot    += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── RGB-weighted cosine (§9.4) ── */

float cosine_rgb_weighted(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        double va = (double)a->A[i];
        double vb = (double)b->A[i];
        if (va > 0.0 && vb > 0.0) {
            float w = rgb_weight(a->R[i], b->R[i],
                                 a->G[i], b->G[i],
                                 a->B[i], b->B[i]);
            dot += va * vb * (double)w;
        }
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── Block summary (SPEC-ENGINE Phase B) ── */

void compute_block_sums(const SpatialGrid* g, BlockSummary* bs) {
    if (!g || !bs) return;

    for (int by = 0; by < BLOCKS; by++) {
        for (int bx = 0; bx < BLOCKS; bx++) {
            uint32_t s = 0;
            for (int y = 0; y < BLOCK; y++) {
                for (int x = 0; x < BLOCK; x++) {
                    s += g->A[(by * BLOCK + y) * GRID_SIZE + (bx * BLOCK + x)];
                }
            }
            bs->sum[by][bx] = s;
        }
    }
}

/* ── Block-skip cosine (SPEC-ENGINE Phase B.3) ── */

float cosine_block_skip(const SpatialGrid* a, const SpatialGrid* b,
                        const BlockSummary* bs_a, const BlockSummary* bs_b) {
    if (!a || !b || !bs_a || !bs_b) return 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;

    for (int by = 0; by < BLOCKS; by++) {
        for (int bx = 0; bx < BLOCKS; bx++) {
            /* Both blocks contribute to norms even if one is zero */
            int y_start = by * BLOCK;
            int x_start = bx * BLOCK;

            if (bs_a->sum[by][bx] == 0 && bs_b->sum[by][bx] == 0) {
                /* Both zero: no contribution to anything */
                continue;
            }

            for (int y = 0; y < BLOCK; y++) {
                for (int x = 0; x < BLOCK; x++) {
                    uint32_t idx = (uint32_t)((y_start + y) * GRID_SIZE + x_start + x);
                    double va = (double)a->A[idx];
                    double vb = (double)b->A[idx];
                    norm_a += va * va;
                    norm_b += vb * vb;

                    if (bs_a->sum[by][bx] == 0 || bs_b->sum[by][bx] == 0) {
                        /* One side zero: dot product contribution is 0 */
                        continue;
                    }
                    dot += va * vb;
                }
            }
        }
    }

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── Top-K selection (partial sort) ── */

void topk_select(Candidate* pool, uint32_t pool_size, uint32_t k) {
    if (!pool || pool_size <= k) return;

    /* Simple selection sort for top-k */
    for (uint32_t i = 0; i < k && i < pool_size; i++) {
        uint32_t max_idx = i;
        for (uint32_t j = i + 1; j < pool_size; j++) {
            if (pool[j].score > pool[max_idx].score) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            Candidate tmp = pool[i];
            pool[i] = pool[max_idx];
            pool[max_idx] = tmp;
        }
    }
}

/* ── Hash bucket (SPEC-ENGINE Phase C) ── */

uint32_t grid_hash(const SpatialGrid* g) {
    if (!g) return 0;
    uint32_t h = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (g->A[i] > 0) {
            uint32_t x = i % GRID_SIZE;
            h = h * 31 + x;
        }
    }
    return h % NUM_BUCKETS;
}

void bucket_index_init(BucketIndex* idx) {
    if (!idx) return;
    memset(idx, 0, sizeof(BucketIndex));
}

void bucket_index_add(BucketIndex* idx, const SpatialGrid* g, uint32_t kf_id) {
    if (!idx || !g) return;
    uint32_t h = grid_hash(g);
    Bucket* b = &idx->buckets[h];
    if (b->count < 256) {
        b->ids[b->count++] = kf_id;
    }
}

void bucket_candidates(BucketIndex* idx, uint32_t hash,
                       int expand, uint32_t* out, uint32_t* out_count) {
    if (!idx || !out || !out_count) return;
    *out_count = 0;

    for (int d = -expand; d <= expand; d++) {
        uint32_t bi = (hash + d + NUM_BUCKETS) % NUM_BUCKETS;
        Bucket* b = &idx->buckets[bi];
        for (uint32_t i = 0; i < b->count; i++) {
            if (*out_count < 1024) {
                out[(*out_count)++] = b->ids[i];
            }
        }
    }
}
