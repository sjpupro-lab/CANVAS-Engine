#include "spatial_io.h"
#include "spatial_grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Header struct (on-disk, 32 bytes) ── */
typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t kf_count;
    uint32_t df_count;
    uint32_t reserved[3];
} SpaiHeader;

const char* spai_status_str(SpaiStatus s) {
    switch (s) {
        case SPAI_OK:          return "OK";
        case SPAI_ERR_OPEN:    return "cannot open file";
        case SPAI_ERR_MAGIC:   return "bad magic (not a SPAI file)";
        case SPAI_ERR_VERSION: return "unsupported file version";
        case SPAI_ERR_READ:    return "read error / truncated file";
        case SPAI_ERR_WRITE:   return "write error";
        case SPAI_ERR_CORRUPT: return "file corrupted";
        case SPAI_ERR_ALLOC:   return "allocation failed";
        case SPAI_ERR_STATE:   return "engine has fewer entries than file";
    }
    return "unknown";
}

/* ── header helpers ── */

static SpaiStatus read_header(FILE* fp, SpaiHeader* h) {
    if (fread(h, sizeof(*h), 1, fp) != 1) return SPAI_ERR_READ;
    if (memcmp(h->magic, SPAI_MAGIC, 4) != 0) return SPAI_ERR_MAGIC;
    /* Accept any version ≤ current; older files just omit sections
     * added in later versions. */
    if (h->version == 0 || h->version > SPAI_VERSION) return SPAI_ERR_VERSION;
    return SPAI_OK;
}

static SpaiStatus write_header(FILE* fp, uint32_t kf_count, uint32_t df_count) {
    SpaiHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, SPAI_MAGIC, 4);
    h.version  = SPAI_VERSION;
    h.kf_count = kf_count;
    h.df_count = df_count;
    if (fwrite(&h, sizeof(h), 1, fp) != 1) return SPAI_ERR_WRITE;
    return SPAI_OK;
}

/* ── Record writers ── */

static SpaiStatus write_keyframe_record(FILE* fp, const Keyframe* kf) {
    uint8_t tag = SPAI_TAG_KEYFRAME;
    if (fwrite(&tag, 1, 1, fp) != 1) return SPAI_ERR_WRITE;

    if (fwrite(&kf->id, sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(kf->label, 1, 64, fp) != 64) return SPAI_ERR_WRITE;
    if (fwrite(&kf->text_byte_count, sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_WRITE;

    if (fwrite(kf->grid.A, sizeof(uint16_t), GRID_TOTAL, fp) != GRID_TOTAL)
        return SPAI_ERR_WRITE;
    if (fwrite(kf->grid.R, 1, GRID_TOTAL, fp) != GRID_TOTAL) return SPAI_ERR_WRITE;
    if (fwrite(kf->grid.G, 1, GRID_TOTAL, fp) != GRID_TOTAL) return SPAI_ERR_WRITE;
    if (fwrite(kf->grid.B, 1, GRID_TOTAL, fp) != GRID_TOTAL) return SPAI_ERR_WRITE;
    return SPAI_OK;
}

static SpaiStatus write_delta_record(FILE* fp, const DeltaFrame* df) {
    uint8_t tag = SPAI_TAG_DELTA;
    if (fwrite(&tag, 1, 1, fp) != 1) return SPAI_ERR_WRITE;

    if (fwrite(&df->id,        sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(&df->parent_id, sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(df->label,      1, 64, fp)            != 64) return SPAI_ERR_WRITE;
    if (fwrite(&df->count,     sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(&df->change_ratio, sizeof(float), 1, fp) != 1) return SPAI_ERR_WRITE;

    if (df->count > 0) {
        if (fwrite(df->entries, sizeof(DeltaEntry), df->count, fp) != df->count)
            return SPAI_ERR_WRITE;
    }
    return SPAI_OK;
}

/* ── Record readers ── */

/* Read a keyframe record body (tag already consumed). Allocates grid channels. */
static SpaiStatus read_keyframe_body(FILE* fp, Keyframe* kf) {
    memset(kf, 0, sizeof(*kf));

    if (fread(&kf->id, sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(kf->label, 1, 64, fp) != 64) return SPAI_ERR_READ;
    if (fread(&kf->text_byte_count, sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_READ;

    kf->grid.A = (uint16_t*)malloc(GRID_TOTAL * sizeof(uint16_t));
    kf->grid.R = (uint8_t*) malloc(GRID_TOTAL);
    kf->grid.G = (uint8_t*) malloc(GRID_TOTAL);
    kf->grid.B = (uint8_t*) malloc(GRID_TOTAL);
    if (!kf->grid.A || !kf->grid.R || !kf->grid.G || !kf->grid.B) {
        free(kf->grid.A); free(kf->grid.R); free(kf->grid.G); free(kf->grid.B);
        memset(&kf->grid, 0, sizeof(kf->grid));
        return SPAI_ERR_ALLOC;
    }

    if (fread(kf->grid.A, sizeof(uint16_t), GRID_TOTAL, fp) != GRID_TOTAL)
        return SPAI_ERR_READ;
    if (fread(kf->grid.R, 1, GRID_TOTAL, fp) != GRID_TOTAL) return SPAI_ERR_READ;
    if (fread(kf->grid.G, 1, GRID_TOTAL, fp) != GRID_TOTAL) return SPAI_ERR_READ;
    if (fread(kf->grid.B, 1, GRID_TOTAL, fp) != GRID_TOTAL) return SPAI_ERR_READ;
    return SPAI_OK;
}

static SpaiStatus read_delta_body(FILE* fp, DeltaFrame* df) {
    memset(df, 0, sizeof(*df));

    if (fread(&df->id,        sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(&df->parent_id, sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(df->label,      1, 64, fp)            != 64) return SPAI_ERR_READ;
    if (fread(&df->count,     sizeof(uint32_t), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(&df->change_ratio, sizeof(float), 1, fp) != 1) return SPAI_ERR_READ;

    df->entries = NULL;
    if (df->count > 0) {
        df->entries = (DeltaEntry*)malloc(df->count * sizeof(DeltaEntry));
        if (!df->entries) return SPAI_ERR_ALLOC;
        if (fread(df->entries, sizeof(DeltaEntry), df->count, fp) != df->count)
            return SPAI_ERR_READ;
    }
    return SPAI_OK;
}

/* ── Capacity helpers ── */

static int grow_kf_cap(SpatialAI* ai, uint32_t needed) {
    if (needed <= ai->kf_capacity) return 1;
    uint32_t cap = ai->kf_capacity ? ai->kf_capacity : 64;
    while (cap < needed) cap *= 2;
    Keyframe* n = (Keyframe*)realloc(ai->keyframes, cap * sizeof(Keyframe));
    if (!n) return 0;
    memset(&n[ai->kf_capacity], 0, (cap - ai->kf_capacity) * sizeof(Keyframe));
    ai->keyframes   = n;
    ai->kf_capacity = cap;
    return 1;
}

static int grow_df_cap(SpatialAI* ai, uint32_t needed) {
    if (needed <= ai->df_capacity) return 1;
    uint32_t cap = ai->df_capacity ? ai->df_capacity : 64;
    while (cap < needed) cap *= 2;
    DeltaFrame* n = (DeltaFrame*)realloc(ai->deltas, cap * sizeof(DeltaFrame));
    if (!n) return 0;
    memset(&n[ai->df_capacity], 0, (cap - ai->df_capacity) * sizeof(DeltaFrame));
    ai->deltas      = n;
    ai->df_capacity = cap;
    return 1;
}

/* ── Public: save ── */

static SpaiStatus write_weights_record(FILE* fp, const ChannelWeight* w) {
    uint8_t tag = SPAI_TAG_WEIGHTS;
    if (fwrite(&tag, 1, 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(&w->w_A, sizeof(float), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(&w->w_R, sizeof(float), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(&w->w_G, sizeof(float), 1, fp) != 1) return SPAI_ERR_WRITE;
    if (fwrite(&w->w_B, sizeof(float), 1, fp) != 1) return SPAI_ERR_WRITE;
    return SPAI_OK;
}

static SpaiStatus read_weights_body(FILE* fp, ChannelWeight* w) {
    if (fread(&w->w_A, sizeof(float), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(&w->w_R, sizeof(float), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(&w->w_G, sizeof(float), 1, fp) != 1) return SPAI_ERR_READ;
    if (fread(&w->w_B, sizeof(float), 1, fp) != 1) return SPAI_ERR_READ;
    return SPAI_OK;
}

SpaiStatus ai_save(const SpatialAI* ai, const char* path) {
    if (!ai || !path) return SPAI_ERR_OPEN;

    FILE* fp = fopen(path, "wb");
    if (!fp) return SPAI_ERR_OPEN;

    SpaiStatus s = write_header(fp, ai->kf_count, ai->df_count);
    if (s != SPAI_OK) { fclose(fp); return s; }

    /* Write all keyframes, then all deltas. Order within each type
       follows the in-memory array, which matches insertion order. */
    for (uint32_t i = 0; i < ai->kf_count; i++) {
        s = write_keyframe_record(fp, &ai->keyframes[i]);
        if (s != SPAI_OK) { fclose(fp); return s; }
    }
    for (uint32_t i = 0; i < ai->df_count; i++) {
        s = write_delta_record(fp, &ai->deltas[i]);
        if (s != SPAI_OK) { fclose(fp); return s; }
    }

    /* v2: adaptive channel weights as trailing record */
    s = write_weights_record(fp, &ai->global_weights);
    if (s != SPAI_OK) { fclose(fp); return s; }

    if (fclose(fp) != 0) return SPAI_ERR_WRITE;
    return SPAI_OK;
}

/* ── Public: load ── */

SpatialAI* ai_load(const char* path, SpaiStatus* out_status) {
    if (out_status) *out_status = SPAI_OK;
    if (!path) { if (out_status) *out_status = SPAI_ERR_OPEN; return NULL; }

    FILE* fp = fopen(path, "rb");
    if (!fp) { if (out_status) *out_status = SPAI_ERR_OPEN; return NULL; }

    SpaiHeader h;
    SpaiStatus s = read_header(fp, &h);
    if (s != SPAI_OK) { fclose(fp); if (out_status) *out_status = s; return NULL; }

    SpatialAI* ai = spatial_ai_create();
    if (!ai) { fclose(fp); if (out_status) *out_status = SPAI_ERR_ALLOC; return NULL; }

    if (!grow_kf_cap(ai, h.kf_count) || !grow_df_cap(ai, h.df_count)) {
        fclose(fp); spatial_ai_destroy(ai);
        if (out_status) *out_status = SPAI_ERR_ALLOC;
        return NULL;
    }

    /* Walk records. Read until both KF and Delta counts are satisfied,
     * then check for optional v2+ trailing records (weights). */
    uint32_t kfs_read = 0, dfs_read = 0;
    while (kfs_read < h.kf_count || dfs_read < h.df_count) {
        uint8_t tag;
        size_t got = fread(&tag, 1, 1, fp);
        if (got != 1) {
            fclose(fp); spatial_ai_destroy(ai);
            if (out_status) *out_status = SPAI_ERR_READ;
            return NULL;
        }

        if (tag == SPAI_TAG_KEYFRAME) {
            if (kfs_read >= h.kf_count) {
                fclose(fp); spatial_ai_destroy(ai);
                if (out_status) *out_status = SPAI_ERR_CORRUPT;
                return NULL;
            }
            s = read_keyframe_body(fp, &ai->keyframes[kfs_read]);
            if (s != SPAI_OK) {
                fclose(fp); spatial_ai_destroy(ai);
                if (out_status) *out_status = s;
                return NULL;
            }
            kfs_read++;
        } else if (tag == SPAI_TAG_DELTA) {
            if (dfs_read >= h.df_count) {
                fclose(fp); spatial_ai_destroy(ai);
                if (out_status) *out_status = SPAI_ERR_CORRUPT;
                return NULL;
            }
            s = read_delta_body(fp, &ai->deltas[dfs_read]);
            if (s != SPAI_OK) {
                fclose(fp); spatial_ai_destroy(ai);
                if (out_status) *out_status = s;
                return NULL;
            }
            dfs_read++;
        } else {
            fclose(fp); spatial_ai_destroy(ai);
            if (out_status) *out_status = SPAI_ERR_CORRUPT;
            return NULL;
        }
    }

    ai->kf_count = kfs_read;
    ai->df_count = dfs_read;

    /* Optional trailing records (v2+): weights */
    uint8_t tag;
    while (fread(&tag, 1, 1, fp) == 1) {
        if (tag == SPAI_TAG_WEIGHTS) {
            s = read_weights_body(fp, &ai->global_weights);
            if (s != SPAI_OK) {
                fclose(fp); spatial_ai_destroy(ai);
                if (out_status) *out_status = s;
                return NULL;
            }
            /* weight_normalize defensively in case the on-disk values
             * drifted from the sum = 4 invariant. */
            weight_normalize(&ai->global_weights);
        } else {
            /* Unknown trailing tag — stop cleanly, forward compatible. */
            break;
        }
    }
    /* If file had no weights block (v1 file), global_weights stays at
     * the default set by spatial_ai_create. */

    fclose(fp);
    if (out_status) *out_status = SPAI_OK;
    return ai;
}

/* ── Public: incremental save ── */

SpaiStatus ai_save_incremental(const SpatialAI* ai, const char* path) {
    if (!ai || !path) return SPAI_ERR_OPEN;

    /* 1) Probe for an existing valid file. */
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        /* No existing file — fall back to full save. */
        return ai_save(ai, path);
    }

    SpaiHeader h;
    SpaiStatus s = read_header(fp, &h);
    fclose(fp);
    if (s != SPAI_OK) {
        /* Unreadable / wrong magic / wrong version — overwrite with full save. */
        return ai_save(ai, path);
    }

    /* 2) In-memory engine must not have LESS than what the file recorded. */
    if (ai->kf_count < h.kf_count || ai->df_count < h.df_count) {
        return SPAI_ERR_STATE;
    }

    /* 3) Nothing new? no-op. */
    if (ai->kf_count == h.kf_count && ai->df_count == h.df_count) {
        return SPAI_OK;
    }

    /* 4) Full rewrite: incremental save cannot trivially update a
     *    trailing weights block in place (it sits after records).
     *    For simplicity and correctness we rewrite the whole file;
     *    callers with huge KF counts can still call ai_save_incremental
     *    — it's equivalent to ai_save when weights are present. */
    (void)h;  /* unused now */
    return ai_save(ai, path);
}

/* ── Public: peek header ── */

SpaiStatus ai_peek_header(const char* path,
                          uint32_t* out_kf_count,
                          uint32_t* out_df_count,
                          uint32_t* out_version) {
    if (!path) return SPAI_ERR_OPEN;
    FILE* fp = fopen(path, "rb");
    if (!fp) return SPAI_ERR_OPEN;

    SpaiHeader h;
    SpaiStatus s = read_header(fp, &h);
    fclose(fp);
    if (s != SPAI_OK) return s;

    if (out_kf_count) *out_kf_count = h.kf_count;
    if (out_df_count) *out_df_count = h.df_count;
    if (out_version)  *out_version  = h.version;
    return SPAI_OK;
}
