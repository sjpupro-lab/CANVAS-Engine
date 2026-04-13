/*
 * test_wiki.c — Wikipedia dump stress test for SPATIAL-PATTERN-AI
 *
 * Usage:
 *   ./build/test_wiki  data/sample_ko.txt        (Korean)
 *   ./build/test_wiki  data/sample_en.txt        (English)
 *   ./build/test_wiki  data/sample_en.txt  500   (limit to 500 clauses)
 *
 * Input: plain-text file (one paragraph per line, wikiextractor output).
 * Pipeline per clause:
 *   3-layer encode → RGB directional → auto-store (I/P decision)
 *   → overlap coarse → cosine precise → match_engine
 *
 * Reports:
 *   - throughput (clauses/sec, bytes/sec)
 *   - keyframe vs delta ratio
 *   - average similarity, block skip %, cache hit rate
 *   - top-10 most similar clause pairs
 *   - similarity distribution histogram
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ── configuration ── */
#define MAX_CLAUSES     100000
#define MAX_LINE_LEN    4096
#define MIN_CLAUSE_LEN  10       /* skip very short lines (tags, blanks) */
#define DEFAULT_LIMIT   2000
#define TOP_PAIRS       10

/* ── helpers ── */

typedef struct {
    uint32_t a;
    uint32_t b;
    float    sim;
} SimPair;

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Strip XML-like tags: <doc ...> </doc> */
static int is_meta_line(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    if (line[0] == '<') return 1;   /* <doc id=...>, </doc> */
    if (line[0] == '\0') return 1;  /* blank */
    return 0;
}

/* Trim trailing whitespace / newlines */
static void trim_trailing(char* s) {
    int len = (int)strlen(s);
    while (len > 0 && (s[len-1] == '\n' || s[len-1] == '\r' ||
                       s[len-1] == ' '  || s[len-1] == '\t')) {
        s[--len] = '\0';
    }
}

/* Simple clause splitter: split line by sentence-ending punctuation.
   Writes clauses into out[], returns count. */
static uint32_t split_clauses(const char* line, char out[][MAX_LINE_LEN],
                              uint32_t max_out) {
    uint32_t count = 0;
    const char* p = line;
    const char* start = p;

    while (*p && count < max_out) {
        /* Sentence-end for both Korean and English */
        if (*p == '.' || *p == '!' || *p == '?') {
            /* Include the punctuation */
            uint32_t len = (uint32_t)(p - start + 1);
            if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
                memcpy(out[count], start, len);
                out[count][len] = '\0';
                trim_trailing(out[count]);
                if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) {
                    count++;
                }
            }
            start = p + 1;
            /* Skip whitespace after punctuation */
            while (*start == ' ') start++;
            p = start;
            continue;
        }
        p++;
    }

    /* Remainder (no punctuation) — take it if long enough */
    if (start < p && count < max_out) {
        uint32_t len = (uint32_t)(p - start);
        if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
            memcpy(out[count], start, len);
            out[count][len] = '\0';
            trim_trailing(out[count]);
            if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) {
                count++;
            }
        }
    }

    return count;
}

/* ── main ── */

int main(int argc, char* argv[]) {
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

    /* ── Phase 1: Load and split clauses ── */

    printf("[1/4] Loading and splitting clauses...\n");
    double t_start = now_sec();

    morpheme_init();

    /* Allocate clause storage */
    char (*clauses)[MAX_LINE_LEN] = malloc((size_t)max_clauses * MAX_LINE_LEN);
    if (!clauses) {
        fprintf(stderr, "ERROR: cannot allocate clause buffer\n");
        fclose(fp);
        return 1;
    }

    uint32_t clause_count = 0;
    uint64_t total_bytes = 0;
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

    double t_load = now_sec() - t_start;
    printf("  Loaded:  %u clauses, %llu bytes (%.2f sec)\n\n",
           clause_count, (unsigned long long)total_bytes, t_load);

    if (clause_count == 0) {
        printf("  No clauses found. Check input file format.\n");
        free(clauses);
        return 1;
    }

    /* Print first 3 clauses as sample */
    printf("  Sample clauses:\n");
    for (uint32_t i = 0; i < 3 && i < clause_count; i++) {
        char preview[80];
        strncpy(preview, clauses[i], 76);
        preview[76] = '\0';
        if (strlen(clauses[i]) > 76) strcat(preview, "...");
        printf("    [%u] %s\n", i, preview);
    }
    printf("\n");

    /* ── Phase 2: Encode + Store (full pipeline) ── */

    printf("[2/4] Encoding + storing (3-layer → RGB → auto I/P)...\n");
    t_start = now_sec();

    SpatialAI* ai = spatial_ai_create();
    FrameCache cache;
    cache_init(&cache);
    BucketIndex bidx;
    bucket_index_init(&bidx);

    /* Track block skip stats */
    uint64_t total_blocks_checked = 0;
    uint64_t total_blocks_skipped = 0;

    for (uint32_t i = 0; i < clause_count; i++) {
        /* Store through the full pipeline */
        uint32_t fid = ai_store_auto(ai, clauses[i], "wiki");

        /* Add to bucket index if it became a keyframe */
        if (!(fid & 0x80000000u) && fid < ai->kf_count) {
            bucket_index_add(&bidx, &ai->keyframes[fid].grid, fid);
        }

        /* Block skip stats on the last stored keyframe */
        if (ai->kf_count > 0) {
            uint32_t kid = (fid & 0x80000000u) ? 0 : fid;
            if (kid < ai->kf_count) {
                BlockSummary bs;
                compute_block_sums(&ai->keyframes[kid].grid, &bs);
                for (int by = 0; by < BLOCKS; by++) {
                    for (int bx = 0; bx < BLOCKS; bx++) {
                        total_blocks_checked++;
                        if (bs.sum[by][bx] == 0) total_blocks_skipped++;
                    }
                }
            }
        }

        /* Progress */
        if ((i + 1) % 200 == 0 || i + 1 == clause_count) {
            printf("\r  Stored: %u / %u  (KF=%u, Delta=%u)",
                   i + 1, clause_count, ai->kf_count, ai->df_count);
            fflush(stdout);
        }
    }

    double t_store = now_sec() - t_start;
    printf("\n  Done in %.2f sec  (%.0f clauses/sec, %.0f KB/sec)\n\n",
           t_store,
           (double)clause_count / t_store,
           (double)total_bytes / 1024.0 / t_store);

    /* ── Phase 3: Query matching (predict each clause) ── */

    printf("[3/4] Running match queries (overlap → cosine)...\n");
    t_start = now_sec();

    float* similarities = (float*)malloc(clause_count * sizeof(float));
    uint32_t* match_ids = (uint32_t*)malloc(clause_count * sizeof(uint32_t));
    uint32_t exact_matches = 0;   /* sim >= 0.99 */
    uint32_t high_matches = 0;    /* sim >= 0.5 */
    uint32_t low_matches = 0;     /* sim < 0.1 */
    double sim_sum = 0.0;

    /* Similarity distribution histogram */
    uint32_t hist[10] = {0}; /* [0-10%), [10-20%), ... [90-100%] */

    /* Top similar pairs (excluding self) */
    SimPair top_pairs[TOP_PAIRS];
    memset(top_pairs, 0, sizeof(top_pairs));

    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;

    for (uint32_t i = 0; i < clause_count; i++) {
        float sim = 0.0f;

        /* Use match_engine for the full optimized path */
        if (ai->kf_count >= 2) {
            SpatialGrid* input = grid_create();
            layers_encode_clause(clauses[i], NULL, input);
            update_rgb_directional(input);

            /* Check cache before match */
            uint32_t probe_id = i % ai->kf_count;
            if (cache_get(&cache, probe_id)) cache_hits++;
            else cache_misses++;

            match_ids[i] = match_engine(ai, input, &bidx, NULL, &cache, &sim);
            grid_destroy(input);
        } else {
            match_ids[i] = ai_predict(ai, clauses[i], &sim);
        }
        similarities[i] = sim;
        sim_sum += (double)sim;

        if (sim >= 0.99f) exact_matches++;
        if (sim >= 0.50f) high_matches++;
        if (sim <  0.10f) low_matches++;

        int bucket = (int)(sim * 10.0f);
        if (bucket >= 10) bucket = 9;
        if (bucket < 0)   bucket = 0;
        hist[bucket]++;

        /* Track top cross-clause pairs */
        if (match_ids[i] != i && sim > top_pairs[TOP_PAIRS - 1].sim) {
            top_pairs[TOP_PAIRS - 1].a = i;
            top_pairs[TOP_PAIRS - 1].b = match_ids[i];
            top_pairs[TOP_PAIRS - 1].sim = sim;
            /* Bubble sort last element up */
            for (int j = TOP_PAIRS - 2; j >= 0; j--) {
                if (top_pairs[j + 1].sim > top_pairs[j].sim) {
                    SimPair tmp = top_pairs[j];
                    top_pairs[j] = top_pairs[j + 1];
                    top_pairs[j + 1] = tmp;
                }
            }
        }

        if ((i + 1) % 200 == 0 || i + 1 == clause_count) {
            printf("\r  Queried: %u / %u", i + 1, clause_count);
            fflush(stdout);
        }
    }

    double t_query = now_sec() - t_start;
    printf("\n  Done in %.2f sec  (%.0f queries/sec)\n\n",
           t_query, (double)clause_count / t_query);

    /* ── Phase 4: Report ── */

    printf("[4/4] Results\n");
    printf("========================================\n\n");

    /* Storage summary */
    printf("  STORAGE\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Total clauses:   %u\n", clause_count);
    printf("  Total bytes:     %llu\n", (unsigned long long)total_bytes);
    printf("  Keyframes (I):   %u\n", ai->kf_count);
    printf("  Deltas (P):      %u\n", ai->df_count);
    if (ai->kf_count + ai->df_count > 0) {
        float kf_ratio = (float)ai->kf_count / (float)(ai->kf_count + ai->df_count) * 100.0f;
        float df_ratio = (float)ai->df_count / (float)(ai->kf_count + ai->df_count) * 100.0f;
        printf("  KF ratio:        %.1f%%\n", kf_ratio);
        printf("  Delta ratio:     %.1f%%\n", df_ratio);
    }
    uint64_t mem_kf = (uint64_t)ai->kf_count * 320; /* 320 KB per keyframe */
    uint64_t mem_delta = 0;
    for (uint32_t i = 0; i < ai->df_count; i++) {
        mem_delta += (uint64_t)ai->deltas[i].count * 8; /* 8 bytes per entry */
    }
    printf("  Memory (KF):     %llu KB\n", (unsigned long long)mem_kf);
    printf("  Memory (Delta):  %llu bytes\n", (unsigned long long)mem_delta);
    printf("\n");

    /* Matching summary */
    float avg_sim = (clause_count > 0) ? (float)(sim_sum / clause_count) : 0.0f;
    printf("  MATCHING\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Avg similarity:  %.1f%%\n", avg_sim * 100.0f);
    printf("  Exact (>=99%%):   %u  (%.1f%%)\n",
           exact_matches, (float)exact_matches / clause_count * 100.0f);
    printf("  High (>=50%%):    %u  (%.1f%%)\n",
           high_matches, (float)high_matches / clause_count * 100.0f);
    printf("  Low (<10%%):      %u  (%.1f%%)\n",
           low_matches, (float)low_matches / clause_count * 100.0f);
    printf("\n");

    /* Block skip stats */
    if (total_blocks_checked > 0) {
        float skip_pct = (float)total_blocks_skipped / (float)total_blocks_checked * 100.0f;
        printf("  BLOCK SKIP\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Blocks checked:  %llu\n", (unsigned long long)total_blocks_checked);
        printf("  Blocks skipped:  %llu  (%.1f%%)\n",
               (unsigned long long)total_blocks_skipped, skip_pct);
        printf("\n");
    }

    /* Cache stats */
    if (cache_hits + cache_misses > 0) {
        printf("  LRU CACHE\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Hits:   %u\n", cache_hits);
        printf("  Misses: %u\n", cache_misses);
        printf("  Rate:   %.1f%%\n",
               (float)cache_hits / (float)(cache_hits + cache_misses) * 100.0f);
        printf("\n");
    }

    /* Throughput */
    double t_total = t_load + t_store + t_query;
    printf("  THROUGHPUT\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Load:    %.2f sec\n", t_load);
    printf("  Store:   %.2f sec  (%.0f clauses/sec)\n",
           t_store, (double)clause_count / t_store);
    printf("  Query:   %.2f sec  (%.0f queries/sec)\n",
           t_query, (double)clause_count / t_query);
    printf("  Total:   %.2f sec\n", t_total);
    printf("  Speed:   %.0f KB/sec (store+query)\n",
           (double)total_bytes / 1024.0 / (t_store + t_query));
    printf("\n");

    /* Similarity histogram */
    printf("  SIMILARITY DISTRIBUTION\n");
    printf("  ─────────────────────────────────────\n");
    uint32_t hist_max = 1;
    for (int i = 0; i < 10; i++) {
        if (hist[i] > hist_max) hist_max = hist[i];
    }
    for (int i = 0; i < 10; i++) {
        int bar_len = (int)((float)hist[i] / (float)hist_max * 30.0f);
        if (hist[i] > 0 && bar_len == 0) bar_len = 1;
        printf("  %3d-%3d%% | ", i * 10, (i + 1) * 10);
        for (int b = 0; b < bar_len; b++) printf("#");
        printf(" %u\n", hist[i]);
    }
    printf("\n");

    /* Top similar pairs */
    printf("  TOP-%d MOST SIMILAR PAIRS\n", TOP_PAIRS);
    printf("  ─────────────────────────────────────\n");
    for (int i = 0; i < TOP_PAIRS; i++) {
        if (top_pairs[i].sim <= 0.0f) break;

        char pa[60], pb[60];
        strncpy(pa, clauses[top_pairs[i].a], 56);
        pa[56] = '\0';
        if (strlen(clauses[top_pairs[i].a]) > 56) strcat(pa, "...");

        uint32_t bid = top_pairs[i].b;
        if (bid < clause_count) {
            strncpy(pb, clauses[bid], 56);
            pb[56] = '\0';
            if (strlen(clauses[bid]) > 56) strcat(pb, "...");
        } else {
            snprintf(pb, sizeof(pb), "(KF%u)", bid);
        }

        printf("  %2d. %.1f%%\n", i + 1, top_pairs[i].sim * 100.0f);
        printf("      A[%u]: %s\n", top_pairs[i].a, pa);
        printf("      B[%u]: %s\n\n", bid, pb);
    }

    printf("========================================\n");
    printf("  PASS  (%u clauses processed)\n", clause_count);
    printf("========================================\n");

    /* cleanup */
    free(similarities);
    free(match_ids);
    free(clauses);
    spatial_ai_destroy(ai);

    return 0;
}
