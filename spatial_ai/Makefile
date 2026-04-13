CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -Iinclude -std=c11
LDFLAGS = -lm

SRC_DIR   = src
INC_DIR   = include
TEST_DIR  = tests
BUILD_DIR = build

SRCS = $(SRC_DIR)/spatial_grid.c \
       $(SRC_DIR)/spatial_morpheme.c \
       $(SRC_DIR)/spatial_layers.c \
       $(SRC_DIR)/spatial_match.c \
       $(SRC_DIR)/spatial_keyframe.c \
       $(SRC_DIR)/spatial_context.c

OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

TESTS = test_grid test_morpheme test_layers test_match test_keyframe test_context test_integration

.PHONY: all clean test

all: $(BUILD_DIR) $(OBJS)
	@echo "Build complete."

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Test targets
test: $(addprefix $(BUILD_DIR)/,$(TESTS))
	@echo "=== Running all tests ==="
	@for t in $(TESTS); do \
		echo "--- $$t ---"; \
		./$(BUILD_DIR)/$$t || exit 1; \
		echo ""; \
	done
	@echo "=== ALL TESTS PASSED ==="

$(BUILD_DIR)/test_grid: $(TEST_DIR)/test_grid.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_morpheme: $(TEST_DIR)/test_morpheme.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_layers: $(TEST_DIR)/test_layers.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_match: $(TEST_DIR)/test_match.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_keyframe: $(TEST_DIR)/test_keyframe.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_context: $(TEST_DIR)/test_context.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_integration: $(TEST_DIR)/test_integration.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

# ─── Wikipedia integration test (manual run) ─────────────────
$(BUILD_DIR)/test_wiki: $(TEST_DIR)/test_wiki.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

wiki: $(BUILD_DIR)/test_wiki
	@echo "Built test_wiki. Run with:"
	@echo "  ./build/test_wiki data/sample_ko.txt"
	@echo "  ./build/test_wiki data/sample_en.txt"
	@echo "  ./build/test_wiki data/sample_en.txt 500"

# ─── Benchmarks: STS-B + Perplexity ──────────────────────────
$(BUILD_DIR)/bench_stsb: $(TEST_DIR)/bench_stsb.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_perplexity: $(TEST_DIR)/bench_perplexity.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

bench: $(BUILD_DIR)/bench_stsb $(BUILD_DIR)/bench_perplexity
	@echo "=== Benchmarks built ==="
	@echo "  ./build/bench_stsb        data/stsb.tsv"
	@echo "  ./build/bench_perplexity  data/sample_ko.txt"
	@echo ""
	@if [ -f data/stsb.tsv ]; then \
		echo "=== Running bench_stsb ==="; \
		./$(BUILD_DIR)/bench_stsb data/stsb.tsv || true; \
	else \
		echo "(skip bench_stsb: data/stsb.tsv not found — run data/download_stsb.ps1)"; \
	fi
	@echo ""
	@if [ -f data/sample_ko.txt ]; then \
		echo "=== Running bench_perplexity on sample_ko.txt ==="; \
		./$(BUILD_DIR)/bench_perplexity data/sample_ko.txt 500 || true; \
	elif [ -f data/sample_en.txt ]; then \
		echo "=== Running bench_perplexity on sample_en.txt ==="; \
		./$(BUILD_DIR)/bench_perplexity data/sample_en.txt 500 || true; \
	else \
		echo "(skip bench_perplexity: no sample_*.txt — run data/download_wiki.ps1)"; \
	fi

clean:
	rm -rf $(BUILD_DIR)
