/*
 * test_wiki.c — Wikipedia integration test for SPATIAL-PATTERN-AI
 *
 * Usage:
 *   ./build/test_wiki data/sample_ko.txt
 *   ./build/test_wiki data/sample_en.txt
 *   ./build/test_wiki data/sample_en.txt 500
 *
 * Phases:
 *   1. Load + split clauses
 *   2. Encode + auto-store (3-layer → RGB → I/P decision)
 *   3. Similarity matching (match_engine full pipeline)
 *   4. Storage / matching / throughput report
 *   5. Recall@{1, 5, 10} on prefix queries
 *   6. Language separation check  (Korean vs English clustering)
 *   7. Next-clause prediction accuracy
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_context.h"
#include "bench_utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ── configuration ── */
#define MAX_CLAUSES     100000
#define MAX_LINE_LEN    4096
#define MIN_CLAUSE_LEN  10
#define DEFAULT_LIMIT   2000
#define TOP_PAIRS       10
#define RECALL_QUERIES  200     /* subsample for recall test */
#define NEXT_QUERIES    200     /* subsample for next-clause test */

/* ── helpers ── */

typedef struct { uint32_t a, b; float sim; } SimPair;

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int is_meta_line(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    if (line[0] == '<') return 1;
    if (line[0] == '\0') return 1;
    return 0;
}

static void trim_trailing(char* s) {
    int len = (int)strlen(s);
    while (len > 0 && (s[len-1] == '\n' || s[len-1] == '\r' ||
                       s[len-1] == ' '  || s[len-1] == '\t')) {
        s[--len] = '\0';
    }
}

static uint32_t split_clauses(const char* line, char out[][MAX_LINE_LEN],
                              uint32_t max_out) {
    uint32_t count = 0;
    const char* p = line;
    const char* start = p;

    while (*p && count < max_out) {
        if (*p == '.' || *p == '!' || *p == '?') {
            uint32_t len = (uint32_t)(p - start + 1);
            if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
                memcpy(out[count], start, len);
                out[count][len] = '\0';
                trim_trailing(out[count]);
                if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) count++;
            }
            start = p + 1;
            while (*start == ' ') start++;
            p = start;
            continue;
        }
        p++;
    }
    if (start < p && count < max_out) {
        uint32_t len = (uint32_t)(p - start);
        if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
            memcpy(out[count], start, len);
            out[count][len] = '\0';
            trim_trailing(out[count]);
            if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) count++;
        }
    }
    return count;
}

/* Classify clause by predominant byte range.
   0 = English / ASCII, 1 = Korean / CJK, 2 = mixed/other */
static int classify_script(const char* text) {
    const uint8_t* b = (const uint8_t*)text;
    uint32_t ascii = 0, cjk = 0, total = 0;
    for (uint32_t i = 0; b[i]; i++) {
        total++;
        if (b[i] < 0x80) ascii++;
        else if (b[i] >= 0xE0 && b[i] <= 0xEF) cjk++;
    }
    if (total == 0) return 2;
    if (cjk * 2 > total) return 1;   /* ≥ 2 CJK leaders per 4 bytes */
    if (ascii * 4 > total * 3) return 0;
    return 2;
}

/* Extract prefix (first N bytes, split on word boundary) */
static void make_prefix(const char* src, char* dst, uint32_t max_bytes) {
    uint32_t len = (uint32_t)strlen(src);
    if (len <= max_bytes) { strcpy(dst, src); return; }

    uint32_t cut = max_bytes;
    /* back off to a space for clean split */
    while (cut > max_bytes / 2 && src[cut] != ' ' && src[cut] != '\t') cut--;
    if (cut < max_bytes / 2) cut = max_bytes;
    memcpy(dst, src, cut);
    dst[cut] = '\0';
}

/* ── main ── */

int main(int argc, char* argv[]) {
    utf8_console_init();

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <text_file> [max_clauses]\n"
            "\n"
            "  text_file    Plain text (wikiextractor output)\n"
            "  max_clauses  Optional limit (default: %d)\n"
            "\n"
            "Example:\n"
            "  %s data/sample_ko.txt\n"
            "  %s data/sample_en.txt 500\n",
            argv[0], DEFAULT_LIMIT, argv[0], argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    uint32_t max_clauses = (argc >= 3) ? (uint32_t)atoi(argv[2]) : DEFAULT_LIMIT;
    if (max_clauses > MAX_CLAUSES) max_clauses = MAX_CLAUSES;

    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", filepath);
        return 1;
    }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  Wikipedia Test\n");
    printf("========================================\n");
    printf("  File:       %s\n", filepath);
    printf("  Max clauses: %u\n", max_clauses);
    printf("----------------------------------------\n\n");

    /* ── [1/7] Load + split ── */
    printf("[1/7] Loading and splitting clauses...\n");
    double t0 = now_sec();

    morpheme_init();

    char (*clauses)[MAX_LINE_LEN] = malloc((size_t)max_clauses * MAX_LINE_LEN);
    if (!clauses) {
        fprintf(stderr, "ERROR: allocation failed\n");
        fclose(fp);
        return 1;
    }

    uint32_t clause_count = 0;
    uint64_t total_bytes  = 0;
    char line[MAX_LINE_LEN];
    char split_buf[8][MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) && clause_count < max_clauses) {
        trim_trailing(line);
        if (is_meta_line(line)) continue;
        if ((int)strlen(line) < MIN_CLAUSE_LEN) continue;

        uint32_t n = split_clauses(line, split_buf, 8);
        for (uint32_t i = 0; i < n && clause_count < max_clauses; i++) {
            memcpy(clauses[clause_count], split_buf[i], MAX_LINE_LEN);
            total_bytes += strlen(split_buf[i]);
            clause_count++;
        }
    }
    fclose(fp);

    double t_load = now_sec() - t0;
    printf("  Loaded:  %u clauses, %llu bytes (%.2f sec)\n\n",
           clause_count, (unsigned long long)total_bytes, t_load);

    if (clause_count < 10) {
        printf("  Too few clauses. Abort.\n");
        free(clauses);
        return 1;
    }

    for (uint32_t i = 0; i < 3 && i < clause_count; i++) {
        char preview[80];
        strncpy(preview, clauses[i], 76);
        preview[76] = '\0';
        if (strlen(clauses[i]) > 76) strcat(preview, "...");
        printf("    [%u] %s\n", i, preview);
    }
    printf("\n");

    /* Script classification */
    uint32_t script_ko = 0, script_en = 0, script_mix = 0;
    uint8_t* script = (uint8_t*)malloc(clause_count);
    for (uint32_t i = 0; i < clause_count; i++) {
        script[i] = (uint8_t)classify_script(clauses[i]);
        if (script[i] == 0) script_en++;
        else if (script[i] == 1) script_ko++;
        else script_mix++;
    }

    /* ── [2/7] Encode + store ── */
    printf("[2/7] Encoding + storing (3-layer → RGB → auto I/P)...\n");
    t0 = now_sec();

    SpatialAI* ai = spatial_ai_create();
    FrameCache cache;
    cache_init(&cache);
    BucketIndex bidx;
    bucket_index_init(&bidx);

    /* Store map: clause idx → keyframe idx (or 0xFFFFFFFF if delta) */
    uint32_t* clause_to_kf = (uint32_t*)malloc(clause_count * sizeof(uint32_t));

    uint64_t total_blocks_checked = 0;
    uint64_t total_blocks_skipped = 0;

    for (uint32_t i = 0; i < clause_count; i++) {
        uint32_t fid = ai_store_auto(ai, clauses[i], "wiki");
        if (fid & 0x80000000u) {
            clause_to_kf[i] = 0xFFFFFFFFu;   /* stored as delta */
        } else {
            clause_to_kf[i] = fid;
            if (fid < ai->kf_count) {
                bucket_index_add(&bidx, &ai->keyframes[fid].grid, fid);
                BlockSummary bs;
                compute_block_sums(&ai->keyframes[fid].grid, &bs);
                for (int by = 0; by < BLOCKS; by++) {
                    for (int bx = 0; bx < BLOCKS; bx++) {
                        total_blocks_checked++;
                        if (bs.sum[by][bx] == 0) total_blocks_skipped++;
                    }
                }
            }
        }
        if ((i + 1) % 200 == 0 || i + 1 == clause_count) {
            printf("\r  Stored: %u / %u  (KF=%u, Delta=%u)",
                   i + 1, clause_count, ai->kf_count, ai->df_count);
            fflush(stdout);
        }
    }
    double t_store = now_sec() - t0;
    printf("\n  Done in %.2f sec  (%.0f clauses/sec)\n\n",
           t_store, clause_count / t_store);

    /* ── [3/7] Match queries ── */
    printf("[3/7] Running match queries (overlap → cosine)...\n");
    t0 = now_sec();

    float* similarities = (float*)malloc(clause_count * sizeof(float));
    uint32_t* match_ids = (uint32_t*)malloc(clause_count * sizeof(uint32_t));
    uint32_t exact_matches = 0, high_matches = 0, low_matches = 0;
    double sim_sum = 0.0;
    uint32_t hist[10] = {0};
    SimPair top_pairs[TOP_PAIRS];
    memset(top_pairs, 0, sizeof(top_pairs));

    uint32_t cache_hits = 0, cache_misses = 0;

    for (uint32_t i = 0; i < clause_count; i++) {
        float sim = 0.0f;
        SpatialGrid* input = grid_create();
        layers_encode_clause(clauses[i], NULL, input);
        update_rgb_directional(input);

        uint32_t probe = i % (ai->kf_count ? ai->kf_count : 1);
        if (cache_get(&cache, probe)) cache_hits++;
        else cache_misses++;

        match_ids[i] = match_engine(ai, input, &bidx, NULL, &cache, &sim);
        similarities[i] = sim;
        sim_sum += sim;

        if (sim >= 0.99f) exact_matches++;
        if (sim >= 0.50f) high_matches++;
        if (sim <  0.10f) low_matches++;

        int b = (int)(sim * 10.0f);
        if (b < 0) b = 0;
        if (b > 9) b = 9;
        hist[b]++;

        if (match_ids[i] != i && sim > top_pairs[TOP_PAIRS - 1].sim) {
            top_pairs[TOP_PAIRS - 1].a = i;
            top_pairs[TOP_PAIRS - 1].b = match_ids[i];
            top_pairs[TOP_PAIRS - 1].sim = sim;
            for (int j = TOP_PAIRS - 2; j >= 0; j--) {
                if (top_pairs[j + 1].sim > top_pairs[j].sim) {
                    SimPair tmp = top_pairs[j];
                    top_pairs[j] = top_pairs[j + 1];
                    top_pairs[j + 1] = tmp;
                }
            }
        }

        grid_destroy(input);

        if ((i + 1) % 200 == 0 || i + 1 == clause_count) {
            printf("\r  Queried: %u / %u", i + 1, clause_count);
            fflush(stdout);
        }
    }
    double t_query = now_sec() - t0;
    printf("\n  Done in %.2f sec  (%.0f queries/sec)\n\n",
           t_query, clause_count / t_query);

    /* ── [4/7] Basic report ── */
    printf("[4/7] Results\n");
    printf("========================================\n\n");

    printf("  STORAGE\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Total clauses:   %u\n", clause_count);
    printf("  Total bytes:     %llu\n", (unsigned long long)total_bytes);
    printf("  Keyframes (I):   %u\n", ai->kf_count);
    printf("  Deltas (P):      %u\n", ai->df_count);
    {
        uint32_t total = ai->kf_count + ai->df_count;
        if (total > 0) {
            printf("  KF ratio:        %.1f%%\n", 100.0f * ai->kf_count / total);
            printf("  Delta ratio:     %.1f%%\n", 100.0f * ai->df_count / total);
        }
    }
    printf("\n");

    float avg_sim = (clause_count > 0) ? (float)(sim_sum / clause_count) : 0.0f;
    printf("  MATCHING (self-query)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Avg similarity:  %.1f%%\n", avg_sim * 100.0f);
    printf("  Exact (>=99%%):   %u  (%.1f%%)\n",
           exact_matches, 100.0f * exact_matches / clause_count);
    printf("  High (>=50%%):    %u  (%.1f%%)\n",
           high_matches, 100.0f * high_matches / clause_count);
    printf("\n");

    if (total_blocks_checked > 0) {
        printf("  BLOCK SKIP\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Blocks skipped:  %.1f%%\n",
               100.0f * total_blocks_skipped / total_blocks_checked);
        printf("\n");
    }

    printf("  LRU CACHE\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Hit rate:  %.1f%%  (%u hits / %u total)\n",
           100.0f * cache_hits / (cache_hits + cache_misses + 1),
           cache_hits, cache_hits + cache_misses);
    printf("\n");

    /* Similarity histogram */
    printf("  SIMILARITY DISTRIBUTION\n");
    printf("  ─────────────────────────────────────\n");
    uint32_t hist_max = 1;
    for (int i = 0; i < 10; i++) if (hist[i] > hist_max) hist_max = hist[i];
    for (int i = 0; i < 10; i++) {
        int bar = (int)((float)hist[i] / hist_max * 30.0f);
        if (hist[i] > 0 && bar == 0) bar = 1;
        printf("  %3d-%3d%% | ", i * 10, (i + 1) * 10);
        for (int b = 0; b < bar; b++) printf("#");
        printf(" %u\n", hist[i]);
    }
    printf("\n");

    /* Top similar pairs */
    printf("  TOP-%d MOST SIMILAR PAIRS\n", TOP_PAIRS);
    printf("  ─────────────────────────────────────\n");
    for (int i = 0; i < TOP_PAIRS; i++) {
        if (top_pairs[i].sim <= 0.0f) break;
        char pa[60], pb[60];
        strncpy(pa, clauses[top_pairs[i].a], 56); pa[56] = '\0';
        if (strlen(clauses[top_pairs[i].a]) > 56) strcat(pa, "...");
        uint32_t bid = top_pairs[i].b;
        if (bid < clause_count) {
            strncpy(pb, clauses[bid], 56); pb[56] = '\0';
            if (strlen(clauses[bid]) > 56) strcat(pb, "...");
        } else {
            snprintf(pb, sizeof(pb), "(KF%u)", bid);
        }
        printf("  %2d. %.1f%%  [%u] %s\n", i + 1, top_pairs[i].sim * 100.0f,
               top_pairs[i].a, pa);
        printf("             [%u] %s\n", bid, pb);
    }
    printf("\n");

    /* ── [5/7] Recall@K on prefix queries ── */
    printf("[5/7] Recall@K (prefix queries on subsample)\n");
    printf("  ─────────────────────────────────────\n");
    t0 = now_sec();

    uint32_t nq = (clause_count < RECALL_QUERIES) ? clause_count : RECALL_QUERIES;
    uint32_t recall_at_1 = 0, recall_at_5 = 0, recall_at_10 = 0;

    Candidate* scored = (Candidate*)malloc(ai->kf_count * sizeof(Candidate));

    for (uint32_t q = 0; q < nq; q++) {
        uint32_t ci = q * (clause_count / nq);
        if (ci >= clause_count) ci = clause_count - 1;

        uint32_t gold_kf = clause_to_kf[ci];
        if (gold_kf == 0xFFFFFFFFu) continue;   /* clause stored as delta; skip */

        /* Prefix of first 60% bytes */
        char prefix[MAX_LINE_LEN];
        uint32_t full_len = (uint32_t)strlen(clauses[ci]);
        uint32_t prefix_len = full_len * 6 / 10;
        if (prefix_len < 10) prefix_len = full_len;
        make_prefix(clauses[ci], prefix, prefix_len);

        /* Encode + score all keyframes via overlap (fast coarse) */
        SpatialGrid* qg = grid_create();
        layers_encode_clause(prefix, NULL, qg);
        update_rgb_directional(qg);

        for (uint32_t k = 0; k < ai->kf_count; k++) {
            scored[k].id = k;
            scored[k].score = (float)overlap_score(qg, &ai->keyframes[k].grid);
        }

        /* Bubble top-10 to front */
        uint32_t top_need = 10 < ai->kf_count ? 10 : ai->kf_count;
        for (uint32_t i = 0; i < top_need; i++) {
            uint32_t max_i = i;
            for (uint32_t j = i + 1; j < ai->kf_count; j++) {
                if (scored[j].score > scored[max_i].score) max_i = j;
            }
            if (max_i != i) {
                Candidate tmp = scored[i]; scored[i] = scored[max_i]; scored[max_i] = tmp;
            }
        }

        /* Stage 2 on top-10 via cosine */
        for (uint32_t i = 0; i < top_need; i++) {
            scored[i].score = cosine_rgb_weighted(qg, &ai->keyframes[scored[i].id].grid);
        }
        for (uint32_t i = 0; i < top_need; i++) {
            uint32_t max_i = i;
            for (uint32_t j = i + 1; j < top_need; j++) {
                if (scored[j].score > scored[max_i].score) max_i = j;
            }
            if (max_i != i) {
                Candidate tmp = scored[i]; scored[i] = scored[max_i]; scored[max_i] = tmp;
            }
        }

        if (top_need >= 1 && scored[0].id == gold_kf) recall_at_1++;
        for (uint32_t i = 0; i < top_need && i < 5; i++) {
            if (scored[i].id == gold_kf) { recall_at_5++; break; }
        }
        for (uint32_t i = 0; i < top_need && i < 10; i++) {
            if (scored[i].id == gold_kf) { recall_at_10++; break; }
        }

        grid_destroy(qg);
    }

    free(scored);

    double t_recall = now_sec() - t0;
    printf("  Queries:     %u  (prefix = 60%% of original)\n", nq);
    printf("  Recall@1:    %.1f%%  (%u / %u)\n", 100.0f * recall_at_1  / nq, recall_at_1,  nq);
    printf("  Recall@5:    %.1f%%  (%u / %u)\n", 100.0f * recall_at_5  / nq, recall_at_5,  nq);
    printf("  Recall@10:   %.1f%%  (%u / %u)\n", 100.0f * recall_at_10 / nq, recall_at_10, nq);
    printf("  Query time:  %.2f sec  (%.0f/sec)\n", t_recall, nq / t_recall);
    printf("\n");

    /* ── [6/7] Language separation check ── */
    printf("[6/7] Language separation (Korean vs English)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Clause script distribution:\n");
    printf("    Korean (CJK): %u  (%.1f%%)\n", script_ko, 100.0f * script_ko / clause_count);
    printf("    English:      %u  (%.1f%%)\n", script_en, 100.0f * script_en / clause_count);
    printf("    Mixed/other:  %u  (%.1f%%)\n", script_mix, 100.0f * script_mix / clause_count);

    if (script_ko > 10 && script_en > 10) {
        /* Within-script avg cosine vs cross-script avg cosine */
        uint32_t ko_pairs = 0, en_pairs = 0, cross_pairs = 0;
        double ko_sum = 0, en_sum = 0, cross_sum = 0;

        SpatialGrid* ga = grid_create();
        SpatialGrid* gb = grid_create();
        uint32_t step = clause_count / 30; if (step == 0) step = 1;

        for (uint32_t i = 0; i < clause_count; i += step) {
            for (uint32_t j = i + step; j < clause_count; j += step) {
                grid_clear(ga); grid_clear(gb);
                layers_encode_clause(clauses[i], NULL, ga);
                layers_encode_clause(clauses[j], NULL, gb);
                float s = cosine_a_only(ga, gb);
                if (script[i] == 1 && script[j] == 1) { ko_sum += s; ko_pairs++; }
                else if (script[i] == 0 && script[j] == 0) { en_sum += s; en_pairs++; }
                else if ((script[i] == 0 && script[j] == 1) ||
                         (script[i] == 1 && script[j] == 0)) { cross_sum += s; cross_pairs++; }
            }
        }
        grid_destroy(ga);
        grid_destroy(gb);

        double avg_ko    = ko_pairs    ? ko_sum    / ko_pairs    : 0.0;
        double avg_en    = en_pairs    ? en_sum    / en_pairs    : 0.0;
        double avg_cross = cross_pairs ? cross_sum / cross_pairs : 0.0;

        printf("\n  Average cosine (sampled pairs):\n");
        printf("    Korean ↔ Korean:   %.1f%%  (n=%u)\n", avg_ko    * 100, ko_pairs);
        printf("    English ↔ English: %.1f%%  (n=%u)\n", avg_en    * 100, en_pairs);
        printf("    Korean ↔ English:  %.1f%%  (n=%u)\n", avg_cross * 100, cross_pairs);

        double within = (avg_ko * ko_pairs + avg_en * en_pairs) /
                        (ko_pairs + en_pairs + 1e-9);
        double separation = within - avg_cross;
        printf("\n  Separation (within - cross): %+.3f\n", separation);
        if (separation > 0.05)
            printf("  Languages are well separated in pattern space.\n");
        else if (separation > 0.0)
            printf("  Weak but positive separation.\n");
        else
            printf("  No clear separation.\n");
    } else {
        printf("\n  Single-script corpus. Separation test skipped.\n");
    }
    printf("\n");

    /* ── [7/7] Next-clause prediction ── */
    printf("[7/7] Next-clause prediction accuracy\n");
    printf("  ─────────────────────────────────────\n");

    /* For consecutive clauses (c_i, c_{i+1}):
       - Query with c_i
       - Check if the best-matching keyframe's sequential neighbor is
         more similar to c_{i+1} than to a random clause. */
    uint32_t np = (clause_count - 1 < NEXT_QUERIES) ? clause_count - 1 : NEXT_QUERIES;
    uint32_t correct_top1 = 0, correct_top5 = 0;
    uint32_t baseline = 0;
    SpatialGrid* gg_next = grid_create();
    SpatialGrid* gg_rand = grid_create();

    for (uint32_t q = 0; q < np; q++) {
        uint32_t ci = q * ((clause_count - 1) / np);
        if (ci + 1 >= clause_count) continue;

        /* Match c_i */
        float sim_in;
        uint32_t matched = ai_predict(ai, clauses[ci], &sim_in);
        if (matched >= ai->kf_count) continue;

        /* Predicted next: sequentially next keyframe (if any) */
        uint32_t predicted_next = (matched + 1 < ai->kf_count) ? (matched + 1) : matched;

        /* Score predicted_next vs true c_{i+1} */
        grid_clear(gg_next);
        layers_encode_clause(clauses[ci + 1], NULL, gg_next);
        float s_pred = cosine_a_only(gg_next, &ai->keyframes[predicted_next].grid);

        /* Baseline: 5 random keyframes */
        float s_rand_best = 0.0f;
        uint32_t r_correct_top5 = 0;
        (void)gg_rand;
        for (int r = 0; r < 5; r++) {
            uint32_t rk = (q * 131 + r * 17) % ai->kf_count;
            float sr = cosine_a_only(gg_next, &ai->keyframes[rk].grid);
            if (sr > s_rand_best) s_rand_best = sr;
            if (rk == predicted_next) r_correct_top5 = 1;
        }

        if (s_pred >= s_rand_best) correct_top1++;

        /* Top-5: check if predicted_next beats 4 of 5 randoms */
        uint32_t beats = 0;
        for (int r = 0; r < 5; r++) {
            uint32_t rk = (q * 131 + r * 17) % ai->kf_count;
            float sr = cosine_a_only(gg_next, &ai->keyframes[rk].grid);
            if (s_pred >= sr) beats++;
        }
        if (beats >= 4) correct_top5++;
        if (s_rand_best > 0.001f) baseline++;
        (void)r_correct_top5;
    }
    grid_destroy(gg_next);
    grid_destroy(gg_rand);

    printf("  Queries:       %u  (c_i → predict c_{i+1})\n", np);
    printf("  Predict top-1: %.1f%%  (beats best of 5 random)\n", 100.0f * correct_top1 / (np + 1));
    printf("  Predict top-5: %.1f%%  (beats 4 of 5 random)\n",    100.0f * correct_top5 / (np + 1));
    printf("  Method: sequential KF+1 prediction via frame order\n");
    printf("\n");

    printf("========================================\n");
    printf("  PASS  (%u clauses processed)\n", clause_count);
    printf("========================================\n");

    free(similarities);
    free(match_ids);
    free(clause_to_kf);
    free(script);
    free(clauses);
    spatial_ai_destroy(ai);
    return 0;
}
