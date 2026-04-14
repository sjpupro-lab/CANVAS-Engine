#ifndef SPATIAL_SUBTITLE_H
#define SPATIAL_SUBTITLE_H

#include "spatial_canvas.h"

/*
 * SubtitleTrack + SpatialCanvasPool
 *
 * SubtitleTrack is a *metadata-only* index layer over canvas slots. It
 * never influences matching scores; its role is to let queries jump
 * directly to the slots of the relevant DataType instead of scanning
 * every stored slot.
 *
 *   Think of it as the subtitle track of a video — the pixel data
 *   (canvases) isn't touched, but the subtitles tell you where to
 *   seek.
 *
 * SpatialCanvasPool owns a growable array of SpatialCanvas objects
 * plus the SubtitleTrack that indexes them. pool_add_clause routes
 * each new clause to a canvas of the matching DataType (creating a
 * new canvas if none exists), preserving type-homogeneous tiling.
 */

/* ── SubtitleTrack ─────────────────────────────────────── */

typedef struct {
    DataType type;
    uint32_t topic_hash;
    uint32_t canvas_id;    /* index into pool->canvases */
    uint32_t slot_id;      /* slot within that canvas */
    uint32_t byte_length;  /* original clause length */
} SubtitleEntry;

typedef struct {
    SubtitleEntry* entries;
    uint32_t       count;
    uint32_t       capacity;

    /* Per-type indices: entries[ids[i]] for fast "slots of type T" lookup.
       Kept in sync on every subtitle_track_add. */
    uint32_t* by_type[DATA_TYPE_COUNT];
    uint32_t  by_type_count[DATA_TYPE_COUNT];
    uint32_t  by_type_cap[DATA_TYPE_COUNT];
} SubtitleTrack;

void     subtitle_track_init(SubtitleTrack* t);
void     subtitle_track_destroy(SubtitleTrack* t);

/* Append an entry. Returns the new entry index. */
uint32_t subtitle_track_add(SubtitleTrack* t,
                            DataType type, uint32_t topic_hash,
                            uint32_t canvas_id, uint32_t slot_id,
                            uint32_t byte_length);

/* Return pointer to ids[] array for a given type; count via out param. */
const uint32_t* subtitle_track_ids_of_type(const SubtitleTrack* t,
                                           DataType type, uint32_t* out_count);

/* ── SpatialCanvasPool ─────────────────────────────────── */

typedef struct SpatialCanvasPool_ {
    SpatialCanvas** canvases;
    uint32_t        count;
    uint32_t        capacity;
    SubtitleTrack   track;
} SpatialCanvasPool;

SpatialCanvasPool* pool_create(void);
void               pool_destroy(SpatialCanvasPool* p);

/* Place a clause. Detects DataType, finds a canvas of matching type
 * with an empty slot, or creates a new canvas. Also appends a subtitle
 * entry. Returns the new subtitle entry index, or -1 on failure. */
int                pool_add_clause(SpatialCanvasPool* p, const char* text);

/* Total populated slots across all canvases (== track.count) */
uint32_t           pool_total_slots(const SpatialCanvasPool* p);

/* ── 4-step matching (type jump → A → R×G → B×A → other types) ── */

typedef struct {
    uint32_t canvas_id;
    uint32_t slot_id;
    float    similarity;
    DataType query_type;
    int      fallback;   /* 1 if Step 4 (other-type) was taken */
    int      step_taken; /* 1=A, 2=RG, 3=BA, 4=fallback */
} PoolMatchResult;

/* Match a pre-encoded query grid against the pool. query_text is used
 * only for DataType detection of the query (no textual comparison). */
PoolMatchResult    pool_match(SpatialCanvasPool* p,
                              const SpatialGrid* query,
                              const char* query_text);

/* Slot-level scoring primitives (used by pool_match; also useful for
 * tests). All operate on a single 256x256 slot region of the canvas
 * directly without copying to a grid. */
float canvas_slot_cosine_a (const SpatialCanvas* c, uint32_t slot,
                             const SpatialGrid* q);
float canvas_slot_rg_score (const SpatialCanvas* c, uint32_t slot,
                             const SpatialGrid* q);
float canvas_slot_ba_score (const SpatialCanvas* c, uint32_t slot,
                             const SpatialGrid* q);

#endif /* SPATIAL_SUBTITLE_H */
