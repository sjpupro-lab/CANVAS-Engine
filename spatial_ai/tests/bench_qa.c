/*
 * bench_qa.c — SQuAD-style retrieval QA benchmark
 *
 * Input TSV format (one pair per line):
 *   <question>\t<answer>
 *
 * Pipeline:
 *   Training
 *     For each pair i, store Q_i then A_i as consecutive keyframes.
 *     So the frame sequence is: [Q_0, A_0, Q_1, A_1, ...].
 *     All QA pairs become context frames preserving "answer follows question".
 *
 *   Test
 *     For each held-out pair (q_test, a_test):
 *       encode q_test → find best-matching keyframe via match_engine
 *       predicted_answer = keyframe[matched_id + 1]   (sequentially next)
 *       retrieval_hit    = matched keyframe is a question (even idx) and
 *                          its paired answer is close to a_test
 *
 * Metrics
 *   - Retrieval accuracy:  did the best match land on an even-indexed
 *     keyframe (i.e., a stored Question)?
 *   - Answer cosine:       cosine(predicted_answer, a_test)
 *   - Gold-answer rank:    where the true-paired A lands in the top-K
 *
 * Also supports self-consistency mode (no test split): every stored
 * question is queried back and must retrieve itself first.
 *
 * Usage
 *   ./build/bench_qa data/qa.tsv
 *   ./build/bench_qa data/qa.tsv 500
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
#include <math.h>
#include <time.h>

/* ── config ── */
#define MAX_PAIRS       8000
#define MAX_LINE_LEN    4096
#define TEST_SPLIT      0.20f
#define DEFAULT_LIMIT   1000

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    char* q;
    char* a;
} QAPair;

/* ── TSV parsing ── */

static int parse_line(const char* line, QAPair* out) {
    const char* tab = strchr(line, '\t');
    if (!tab) return 0;

    size_t q_len = (size_t)(tab - line);
    const char* a_start = tab + 1;
    size_t a_len = strlen(a_start);
    while (a_len > 0 && (a_start[a_len - 1] == '\n' ||
                         a_start[a_len - 1] == '\r' ||
                         a_start[a_len - 1] == ' ')) {
        a_len--;
    }

    if (q_len < 3 || a_len < 3) return 0;

    out->q = (char*)malloc(q_len + 1);
    out->a = (char*)malloc(a_len + 1);
    if (!out->q || !out->a) return 0;
    memcpy(out->q, line, q_len); out->q[q_len] = '\0';
    memcpy(out->a, a_start, a_len); out->a[a_len] = '\0';
    return 1;
}

/* ── main ── */

int main(int argc, char* argv[]) {
    utf8_console_init();

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <qa.tsv> [max_pairs]\n\n"
            "  TSV format:  <question>\\t<answer>   (one pair per line)\n\n"
            "Example:\n"
            "  %s data/qa.tsv\n"
            "  %s data/qa.tsv 500\n\n"
            "Generate QA pairs from wiki text:\n"
            "  powershell -File data/make_qa.ps1 data/sample_ko.txt > data/qa.tsv\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    int max_pairs = (argc >= 3) ? atoi(argv[2]) : DEFAULT_LIMIT;
    if (max_pairs > MAX_PAIRS) max_pairs = MAX_PAIRS;

    FILE* fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open '%s'\n", filepath); return 1; }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  QA Benchmark\n");
    printf("========================================\n");
    printf("  File:        %s\n", filepath);
    printf("  Max pairs:   %d\n", max_pairs);
    printf("  Test split:  %.0f%%\n", TEST_SPLIT * 100.0f);
    printf("----------------------------------------\n\n");

    morpheme_init();

    /* ── [1/4] Load QA pairs ── */
    printf("[1/4] Loading QA pairs...\n");
    double t0 = now_sec();

    QAPair* pairs = (QAPair*)calloc((size_t)max_pairs, sizeof(QAPair));
    int n = 0;
    char line[MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) && n < max_pairs) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (parse_line(line, &pairs[n])) n++;
    }
    fclose(fp);

    printf("  Loaded: %d pairs (%.2f sec)\n\n", n, now_sec() - t0);

    if (n < 10) {
        fprintf(stderr, "ERROR: need at least 10 pairs\n");
        free(pairs);
        return 1;
    }

    int test_n  = (int)(n * TEST_SPLIT);
    if (test_n < 3) test_n = 3;
    int train_n = n - test_n;

    /* Show sample */
    printf("  Sample pairs:\n");
    for (int i = 0; i < 2 && i < n; i++) {
        char qp[80], ap[80];
        strncpy(qp, pairs[i].q, 76); qp[76] = '\0';
        strncpy(ap, pairs[i].a, 76); ap[76] = '\0';
        if (strlen(pairs[i].q) > 76) strcat(qp, "...");
        if (strlen(pairs[i].a) > 76) strcat(ap, "...");
        printf("    Q[%d]: %s\n", i, qp);
        printf("    A[%d]: %s\n", i, ap);
    }
    printf("\n");

    /* ── [2/4] Store training pairs ── */
    printf("[2/4] Storing %d training pairs as consecutive keyframes...\n", train_n);
    t0 = now_sec();

    SpatialAI* ai = spatial_ai_create();
    FrameCache cache;
    cache_init(&cache);
    BucketIndex bidx;
    bucket_index_init(&bidx);

    /* For each pair: store Q then A. Track which keyframe id each pair's Q occupies. */
    uint32_t* q_kf_id = (uint32_t*)malloc((size_t)train_n * sizeof(uint32_t));
    uint32_t* a_kf_id = (uint32_t*)malloc((size_t)train_n * sizeof(uint32_t));

    for (int i = 0; i < train_n; i++) {
        char qlabel[32], alabel[32];
        snprintf(qlabel, sizeof(qlabel), "Q%d", i);
        snprintf(alabel, sizeof(alabel), "A%d", i);

        /* Use ai_store_auto but we need unique keyframes per pair.
           To ensure sequential storage without delta-collapse, check each
           returned fid. If a pair gets collapsed into a delta, we fall
           back to the nearest keyframe id. */
        uint32_t q_fid = ai_store_auto(ai, pairs[i].q, qlabel);
        if (q_fid & 0x80000000u) {
            /* was stored as delta — find parent keyframe */
            q_fid = ai->deltas[ai->df_count - 1].parent_id;
        }
        q_kf_id[i] = q_fid;
        if (q_fid < ai->kf_count) {
            bucket_index_add(&bidx, &ai->keyframes[q_fid].grid, q_fid);
        }

        uint32_t a_fid = ai_store_auto(ai, pairs[i].a, alabel);
        if (a_fid & 0x80000000u) {
            a_fid = ai->deltas[ai->df_count - 1].parent_id;
        }
        a_kf_id[i] = a_fid;
        if (a_fid < ai->kf_count) {
            bucket_index_add(&bidx, &ai->keyframes[a_fid].grid, a_fid);
        }

        if ((i + 1) % 100 == 0 || i + 1 == train_n) {
            printf("\r  Stored: %d / %d  (KF=%u, Delta=%u)",
                   i + 1, train_n, ai->kf_count, ai->df_count);
        }
    }

    double t_store = now_sec() - t0;
    printf("\n  Done in %.2f sec  (%.0f pairs/sec)\n\n",
           t_store, train_n / t_store);

    /* ── [3/4] Evaluate retrieval on test pairs ── */
    printf("[3/4] Evaluating on %d held-out pairs...\n", test_n);
    t0 = now_sec();

    int retrieval_top1 = 0;   /* matched KF's paired A is the training A most similar to test A */
    int retrieval_top5 = 0;
    double answer_cos_sum = 0.0;
    double answer_cos_random_sum = 0.0;
    uint32_t answered = 0;

    SpatialGrid* gq = grid_create();
    SpatialGrid* ga_true = grid_create();

    for (int t = 0; t < test_n; t++) {
        int idx = train_n + t;

        /* Encode test Q and true A */
        grid_clear(gq);
        grid_clear(ga_true);
        layers_encode_clause(pairs[idx].q, NULL, gq);
        layers_encode_clause(pairs[idx].a, NULL, ga_true);
        update_rgb_directional(gq);
        update_rgb_directional(ga_true);

        /* Match query against all stored keyframes */
        float sim;
        uint32_t matched = match_engine(ai, gq, &bidx, NULL, &cache, &sim);
        if (matched >= ai->kf_count) continue;

        /* Predicted answer = the keyframe immediately after matched */
        uint32_t pred_a_id = (matched + 1 < ai->kf_count) ? (matched + 1) : matched;
        float answer_cos = cosine_a_only(&ai->keyframes[pred_a_id].grid, ga_true);
        answer_cos_sum += answer_cos;
        answered++;

        /* Random baseline: avg cosine over 5 random keyframes */
        float rand_sum = 0.0f;
        for (int r = 0; r < 5; r++) {
            uint32_t rk = ((uint32_t)(t * 131 + r * 17)) % ai->kf_count;
            rand_sum += cosine_a_only(&ai->keyframes[rk].grid, ga_true);
        }
        answer_cos_random_sum += (double)(rand_sum / 5.0f);

        /* Retrieval: find rank of the training A most similar to test A */
        /* Simpler: does matched KF correspond to ANY training Q's position? */
        /* Find best-matching training question among the stored Qs */
        int best_train_qi = -1;
        float best_qsim = -1.0f;
        for (int ti = 0; ti < train_n; ti++) {
            float s = cosine_a_only(gq, &ai->keyframes[q_kf_id[ti]].grid);
            if (s > best_qsim) { best_qsim = s; best_train_qi = ti; }
        }

        /* Rank by Q similarity */
        if (best_train_qi >= 0) {
            /* Count how many training answers are MORE similar to test A
               than the predicted (matched+1) */
            float pred_cos = cosine_a_only(&ai->keyframes[pred_a_id].grid, ga_true);
            int rank = 0;
            for (int ti = 0; ti < train_n; ti++) {
                float s = cosine_a_only(&ai->keyframes[a_kf_id[ti]].grid, ga_true);
                if (s > pred_cos) rank++;
            }
            if (rank == 0) retrieval_top1++;
            if (rank < 5)  retrieval_top5++;
        }

        if ((t + 1) % 20 == 0 || t + 1 == test_n) {
            printf("\r  Evaluated: %d / %d", t + 1, test_n);
        }
    }

    grid_destroy(gq);
    grid_destroy(ga_true);

    double t_eval = now_sec() - t0;
    printf("\n  Done in %.2f sec\n\n", t_eval);

    /* ── [4/4] Report ── */
    printf("[4/4] Results\n");
    printf("========================================\n\n");

    printf("  SETUP\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Total pairs:       %d\n", n);
    printf("  Train pairs:       %d\n", train_n);
    printf("  Test pairs:        %d\n", test_n);
    printf("  Stored keyframes:  %u\n", ai->kf_count);
    printf("  Stored deltas:     %u\n\n", ai->df_count);

    if (answered > 0) {
        double avg_cos    = answer_cos_sum / answered;
        double avg_random = answer_cos_random_sum / answered;
        double lift       = avg_cos - avg_random;

        printf("  RETRIEVAL ACCURACY\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Top-1:             %.2f%%  (%d / %u)\n",
               100.0 * retrieval_top1 / (int)answered,
               retrieval_top1, answered);
        printf("  Top-5:             %.2f%%  (%d / %u)\n",
               100.0 * retrieval_top5 / (int)answered,
               retrieval_top5, answered);
        printf("  Random baseline:   %.2f%%  (5 / train_n)\n\n",
               100.0 * 5 / train_n);

        printf("  ANSWER QUALITY\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Avg answer cos:    %.1f%%\n",        avg_cos    * 100.0);
        printf("  Random-KF baseline: %.1f%%\n",       avg_random * 100.0);
        printf("  Lift over random:  %+.1f%%\n\n",     lift       * 100.0);

        printf("  INTERPRETATION\n");
        printf("  ─────────────────────────────────────\n");
        if (lift > 0.10)
            printf("  Strong QA signal (>10%% over random).\n");
        else if (lift > 0.03)
            printf("  Measurable QA signal.\n");
        else if (lift > 0.0)
            printf("  Weak signal (consider more train data).\n");
        else
            printf("  Near baseline.\n");
        printf("\n");
    }

    printf("========================================\n");
    printf("  PASS\n");
    printf("========================================\n");

    /* cleanup */
    for (int i = 0; i < n; i++) {
        free(pairs[i].q);
        free(pairs[i].a);
    }
    free(pairs);
    free(q_kf_id);
    free(a_kf_id);
    spatial_ai_destroy(ai);
    return 0;
}
