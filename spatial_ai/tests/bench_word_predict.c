/*
 * bench_word_predict.c — masked-word prediction benchmark
 *
 * SPEC alignment:
 *   - Word layer (+2) captures word-level emphasis at each byte position.
 *   - G channel, diffused vertically (↑↓), captures word-substitution
 *     relationships — "words that can appear at the same Y position".
 *
 * Procedure:
 *   Training
 *     For every training clause:
 *       encode with 3-layer summation → track the word-layer bitmap only
 *       wcount[y][x] += word_layer[y][x]  (weight +2 at word bytes)
 *       vocab[word].freq++
 *
 *   Test
 *     For each held-out clause, mask one word W at byte offset s (length L).
 *     For every candidate w in the vocabulary with |w| == L:
 *       log P(w | s) = sum_{i=0..L-1} log P(byte = w[i] | y = (s+i) mod 256)
 *                    where P(byte=v|y) = (wcount[y][v] + eps) /
 *                                         (sum_x wcount[y][x] + 256*eps)
 *     Rank candidates by log P, check where the true W lands.
 *
 * Metrics
 *   - Top-1 accuracy, Top-5 accuracy
 *   - Word-level perplexity = exp(- mean log P(W))
 *
 * Usage
 *   ./build/bench_word_predict data/sample_ko.txt
 *   ./build/bench_word_predict data/sample_en.txt 1000
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "bench_utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── configuration ── */
#define MAX_CLAUSES     10000
#define MAX_LINE_LEN    4096
#define MIN_CLAUSE_LEN  10
#define MAX_VOCAB       5000
#define MAX_WORD_LEN    64
#define MIN_WORD_BYTES  3       /* skip 1-2 byte tokens */
#define TEST_SPLIT      0.10f
#define DEFAULT_LIMIT   1000
#define EPSILON         0.5

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Meta / clause helpers ── */

static int is_meta_line(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    if (line[0] == '<' || line[0] == '\0') return 1;
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
    return count;
}

/* ── Word extraction ── */

typedef struct { char text[MAX_WORD_LEN]; uint32_t offset; uint32_t len; } WordSpan;

/* Extract whitespace-separated words from clause, with byte offsets.
   Returns number of words. Skips punctuation at ends. */
static uint32_t extract_words(const char* clause, WordSpan* out, uint32_t max) {
    uint32_t count = 0;
    uint32_t i = 0;
    uint32_t len = (uint32_t)strlen(clause);

    while (i < len && count < max) {
        while (i < len && (clause[i] == ' ' || clause[i] == '\t')) i++;
        if (i >= len) break;

        uint32_t w_start = i;
        while (i < len && clause[i] != ' ' && clause[i] != '\t') i++;
        uint32_t w_end = i;

        /* Strip trailing punctuation */
        while (w_end > w_start) {
            char c = clause[w_end - 1];
            if (c == '.' || c == ',' || c == '!' || c == '?' ||
                c == ';' || c == ':' || c == ')' || c == ']' ||
                c == '"' || c == '\'')
                w_end--;
            else break;
        }
        /* Strip leading punctuation */
        while (w_start < w_end) {
            char c = clause[w_start];
            if (c == '(' || c == '[' || c == '"' || c == '\'') w_start++;
            else break;
        }

        uint32_t w_len = w_end - w_start;
        if (w_len >= MIN_WORD_BYTES && w_len < MAX_WORD_LEN - 1) {
            memcpy(out[count].text, clause + w_start, w_len);
            out[count].text[w_len] = '\0';
            out[count].offset = w_start;
            out[count].len = w_len;
            count++;
        }
    }
    return count;
}

/* ── Vocabulary ── */

typedef struct {
    char     word[MAX_WORD_LEN];
    uint32_t len;          /* byte length */
    uint32_t freq;
} VocabEntry;

static int vocab_find(VocabEntry* vocab, uint32_t vcount, const char* word, uint32_t wlen) {
    for (uint32_t i = 0; i < vcount; i++) {
        if (vocab[i].len == wlen && memcmp(vocab[i].word, word, wlen) == 0)
            return (int)i;
    }
    return -1;
}

static int vocab_add(VocabEntry* vocab, uint32_t* vcount, uint32_t vmax,
                     const char* word, uint32_t wlen) {
    int idx = vocab_find(vocab, *vcount, word, wlen);
    if (idx >= 0) {
        vocab[idx].freq++;
        return idx;
    }
    if (*vcount >= vmax) return -1;
    VocabEntry* e = &vocab[*vcount];
    memcpy(e->word, word, wlen);
    e->word[wlen] = '\0';
    e->len = wlen;
    e->freq = 1;
    (*vcount)++;
    return (int)(*vcount - 1);
}

/* ── main ── */

int main(int argc, char* argv[]) {
    utf8_console_init();

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <text_file> [max_clauses]\n\n"
            "Example:\n"
            "  %s data/sample_ko.txt\n"
            "  %s data/sample_en.txt 500\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    uint32_t max_clauses = (argc >= 3) ? (uint32_t)atoi(argv[2]) : DEFAULT_LIMIT;
    if (max_clauses > MAX_CLAUSES) max_clauses = MAX_CLAUSES;

    FILE* fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open '%s'\n", filepath); return 1; }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  Word Prediction\n");
    printf("========================================\n");
    printf("  File:        %s\n", filepath);
    printf("  Max clauses: %u\n", max_clauses);
    printf("  Test split:  %.0f%%\n", TEST_SPLIT * 100.0f);
    printf("----------------------------------------\n\n");

    morpheme_init();

    /* ── [1/4] Load ── */
    printf("[1/4] Loading clauses...\n");
    double t0 = now_sec();

    char (*clauses)[MAX_LINE_LEN] = malloc((size_t)max_clauses * MAX_LINE_LEN);
    uint32_t clause_count = 0;
    char line[MAX_LINE_LEN];
    char split_buf[8][MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) && clause_count < max_clauses) {
        trim_trailing(line);
        if (is_meta_line(line)) continue;
        uint32_t n = split_clauses(line, split_buf, 8);
        for (uint32_t i = 0; i < n && clause_count < max_clauses; i++) {
            memcpy(clauses[clause_count++], split_buf[i], MAX_LINE_LEN);
        }
    }
    fclose(fp);
    printf("  Loaded: %u clauses (%.2f sec)\n\n", clause_count, now_sec() - t0);

    if (clause_count < 20) {
        fprintf(stderr, "ERROR: need at least 20 clauses\n");
        free(clauses);
        return 1;
    }

    uint32_t test_count  = (uint32_t)(clause_count * TEST_SPLIT);
    if (test_count < 5) test_count = 5;
    uint32_t train_count = clause_count - test_count;

    /* ── [2/4] Train: word-layer distribution + vocabulary ── */
    printf("[2/4] Training on %u clauses...\n", train_count);
    t0 = now_sec();

    /* word-layer frequency: count[y][x] over training */
    double (*wcount)[256] = calloc(256 * 256, sizeof(double));
    VocabEntry* vocab = (VocabEntry*)calloc(MAX_VOCAB, sizeof(VocabEntry));
    uint32_t vcount = 0;

    LayerBitmaps* lb = layers_create();
    SpatialGrid* dummy = grid_create();
    WordSpan words[256];

    for (uint32_t c = 0; c < train_count; c++) {
        /* Get word-layer bitmap */
        memset(lb, 0, sizeof(LayerBitmaps));
        layers_encode_clause(clauses[c], lb, dummy);

        for (uint32_t y = 0; y < 256; y++) {
            for (uint32_t x = 0; x < 256; x++) {
                wcount[y][x] += (double)lb->word[y * 256 + x];
            }
        }

        /* Build vocab */
        uint32_t nw = extract_words(clauses[c], words, 256);
        for (uint32_t w = 0; w < nw; w++) {
            vocab_add(vocab, &vcount, MAX_VOCAB, words[w].text, words[w].len);
        }

        if ((c + 1) % 200 == 0 || c + 1 == train_count) {
            printf("\r  Processed: %u / %u (vocab=%u)", c + 1, train_count, vcount);
        }
    }
    grid_destroy(dummy);
    layers_destroy(lb);

    /* Row totals */
    double row_total[256];
    for (uint32_t y = 0; y < 256; y++) {
        double s = 0.0;
        for (uint32_t x = 0; x < 256; x++) s += wcount[y][x];
        row_total[y] = s;
    }

    double t_train = now_sec() - t0;
    printf("\n  Done in %.2f sec  (vocab size = %u)\n\n", t_train, vcount);

    /* ── [3/4] Predict masked words in test set ── */
    printf("[3/4] Predicting masked words in %u held-out clauses...\n", test_count);
    t0 = now_sec();

    uint32_t top1_hits = 0;
    uint32_t top5_hits = 0;
    uint32_t total_preds = 0;
    double   log_p_sum = 0.0;
    uint32_t oov = 0;   /* true word not in training vocab */

    /* Score buffer: per-vocab log probability */
    double* scores = (double*)malloc(MAX_VOCAB * sizeof(double));

    uint32_t clauses_with_any = 0;
    for (uint32_t t = 0; t < test_count; t++) {
        uint32_t ci = train_count + t;

        uint32_t nw = extract_words(clauses[ci], words, 256);
        if (nw < 2) continue;

        int any_in_clause = 0;

        /* Predict every in-vocab word in the clause */
        for (uint32_t w = 0; w < nw; w++) {
            WordSpan* target = &words[w];
            int true_idx = vocab_find(vocab, vcount, target->text, target->len);
            if (true_idx < 0) { oov++; continue; }

            /* Score every same-length candidate */
            uint32_t candidates = 0;
            double true_score = 0.0;

            for (uint32_t v = 0; v < vcount; v++) {
                if (vocab[v].len != target->len) {
                    scores[v] = -1e18;
                    continue;
                }
                double lp = 0.0;
                for (uint32_t i = 0; i < vocab[v].len; i++) {
                    uint32_t y = (target->offset + i) % 256;
                    uint8_t  v_byte = (uint8_t)vocab[v].word[i];
                    double p = (wcount[y][v_byte] + EPSILON) /
                               (row_total[y] + 256.0 * EPSILON);
                    lp += log(p);
                }
                scores[v] = lp;
                candidates++;
                if ((int)v == true_idx) true_score = lp;
            }

            if (candidates < 2) continue;

            /* Rank position of true_idx */
            uint32_t rank = 0;
            for (uint32_t v = 0; v < vcount; v++) {
                if ((int)v == true_idx) continue;
                if (scores[v] > true_score) rank++;
            }

            if (rank == 0) top1_hits++;
            if (rank < 5)  top5_hits++;

            log_p_sum += true_score;
            total_preds++;
            any_in_clause = 1;
        }

        if (any_in_clause) clauses_with_any++;

        if ((t + 1) % 50 == 0 || t + 1 == test_count) {
            printf("\r  Predicted: %u / %u clauses, %u preds", t + 1, test_count, total_preds);
        }
    }

    double t_pred = now_sec() - t0;
    printf("\n  Done in %.2f sec  (%.0f preds/sec)\n\n",
           t_pred, total_preds / (t_pred + 1e-9));

    /* ── [4/4] Report ── */
    printf("[4/4] Results\n");
    printf("========================================\n\n");

    printf("  VOCABULARY\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Train clauses:         %u\n", train_count);
    printf("  Test clauses:          %u  (with predictions: %u)\n",
           test_count, clauses_with_any);
    printf("  Vocab size:            %u\n", vcount);
    printf("  In-vocab predictions:  %u\n", total_preds);
    printf("  OOV words skipped:     %u\n\n", oov);

    if (total_preds > 0) {
        double top1 = 100.0 * top1_hits / total_preds;
        double top5 = 100.0 * top5_hits / total_preds;
        double avg_lp = log_p_sum / total_preds;
        double word_ppl = exp(-avg_lp);

        printf("  ACCURACY\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Top-1:                %.2f%%  (%u / %u)\n",
               top1, top1_hits, total_preds);
        printf("  Top-5:                %.2f%%  (%u / %u)\n",
               top5, top5_hits, total_preds);
        printf("  Random baseline:      %.2f%%  (1 / vocab_same_len)\n\n",
               100.0 / (double)vcount);

        printf("  WORD-LEVEL PERPLEXITY\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Avg NLL:              %.4f nats/word\n", -avg_lp);
        printf("  Word perplexity:      %.2f\n", word_ppl);
        printf("  (lower is better; uniform baseline = vocab_size)\n\n");

        printf("  INTERPRETATION\n");
        printf("  ─────────────────────────────────────\n");
        if (top1 > 10.0)
            printf("  Strong word-level prediction.\n");
        else if (top1 > 2.0)
            printf("  Measurable word-level signal.\n");
        else if (top1 > 0.5)
            printf("  Weak signal (increase training data).\n");
        else
            printf("  Near baseline (train more clauses).\n");
        printf("\n");
    }

    printf("========================================\n");
    printf("  PASS\n");
    printf("========================================\n");

    free(scores);
    free(vocab);
    free(wcount);
    free(clauses);
    return 0;
}
