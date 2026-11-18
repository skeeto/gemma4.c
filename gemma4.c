#if !defined(_WIN32) && !defined(__APPLE__) // macOS keeps its default namespace because sysctl needs the BSD types that _POSIX_C_SOURCE hides.
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include "win.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// The compiled-in architecture must match the exported checkpoint. E2B is the
// default; build with -DGEMMA4_E4B for the E4B checkpoint.
#ifdef GEMMA4_E4B
#define MODEL_FILE "gemma4-E4B-int8.bin"
#define NUM_LAYERS 42
#define HIDDEN_SIZE 2560
#define FULL_INTERVAL 6  // Every sixth layer uses full attention.
#define CACHE_SETS 4     // KV-writing full-attention layers: 5, 11, 17, 23.
#define SLIDING_KV 512   // Two KV heads of 256 channels.
#define FULL_KV 1024     // Two KV heads of 512 channels.
#else
#define MODEL_FILE "gemma4-E2B-int8.bin"
#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define FULL_INTERVAL 5  // Every fifth layer uses full attention.
#define CACHE_SETS 3     // KV-writing full-attention layers: 4, 9, 14.
#define SLIDING_KV 256   // One KV head of 256 channels.
#define FULL_KV 512      // One KV head of 512 channels.
#endif
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072
#define SLIDING_WINDOW 512
#define BATCH_SIZE 512

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct {
    char key[8]; // Holds one UTF-8 piece or a pair of 32-bit token IDs.
    int result, rank;
} LookupEntry;

typedef struct {
    char token[94]; // The longest vocabulary piece is 93 bytes plus the null terminator.
    int id;
} VocabEntry;

typedef struct {
    int merge_count;
    int encode_vocab_count;
    int special_count;
    char decoded_tokens[VOCAB_SIZE][94];
    VocabEntry specials[256]; // Reserves space for the checkpoint's 24 special tokens.
    LookupEntry encode_vocab[32768]; // Reserves space for 19,249 directly encoded pieces.
    LookupEntry merges[514906]; // This checkpoint contains 514,906 merge rules.
} Tokenizer;

int compare_lookup_keys(const void *key, const void *entry) {
    return memcmp(key, ((const LookupEntry *)entry)->key, 8);
}

// A candidate merge of the adjacent tokens at left and right, ordered by rank, then position.
typedef struct {
    int rank, left, right;
    const LookupEntry *merge;
} MergeCandidate;

int candidate_before(const MergeCandidate *a, const MergeCandidate *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->left < b->left;
}

// Pushes the merge for the pair at left and right onto the heap, if one exists.
void push_candidate(const Tokenizer *tokenizer, const int *tokens, MergeCandidate *heap, int *heap_count, int left, int right) {
    int pair[2] = {tokens[left], tokens[right]};
    const LookupEntry *merge = bsearch(pair, tokenizer->merges, tokenizer->merge_count, sizeof(LookupEntry), compare_lookup_keys);
    if (!merge) return;
    int i = (*heap_count)++;
    heap[i] = (MergeCandidate){merge->rank, left, right, merge};
    for (; i && candidate_before(&heap[i], &heap[(i - 1) / 2]); i = (i - 1) / 2) {
        MergeCandidate swap = heap[i]; heap[i] = heap[(i - 1) / 2]; heap[(i - 1) / 2] = swap;
    }
}

MergeCandidate pop_candidate(MergeCandidate *heap, int *heap_count) {
    MergeCandidate top = heap[0];
    heap[0] = heap[--*heap_count];
    for (int i = 0;;) {
        int least = i, child = 2 * i + 1;
        if (child < *heap_count && candidate_before(&heap[child], &heap[least])) least = child;
        if (child + 1 < *heap_count && candidate_before(&heap[child + 1], &heap[least])) least = child + 1;
        if (least == i) break;
        MergeCandidate swap = heap[i]; heap[i] = heap[least]; heap[least] = swap;
        i = least;
    }
    return top;
}

// Repeatedly applies the highest-priority learned merge, leftmost first, until no adjacent
// token pair matches. Tokens form a linked list, and a heap holds the candidate merges,
// discarding those whose pair has since changed.
int apply_bpe_merges(const Tokenizer *tokenizer, int *tokens, int count) {
    if (count < 2) return count;
    int *next = malloc((size_t)count * sizeof(int)), *prev = malloc((size_t)count * sizeof(int));
    MergeCandidate *heap = malloc((size_t)count * 3 * sizeof(MergeCandidate)); // Each merge adds at most two.
    if (!next || !prev || !heap) { perror("malloc"); exit(1); }
    int heap_count = 0;
    for (int i = 0; i < count; i++) { next[i] = i + 1; prev[i] = i - 1; }
    for (int i = 0; i + 1 < count; i++) push_candidate(tokenizer, tokens, heap, &heap_count, i, i + 1);

    while (heap_count) {
        MergeCandidate c = pop_candidate(heap, &heap_count);
        int pair[2] = {tokens[c.left], tokens[c.right]};
        if (tokens[c.left] < 0 || next[c.left] != c.right || memcmp(pair, c.merge->key, 8)) continue; // Stale.
        tokens[c.left] = c.merge->result;
        tokens[c.right] = -1; // Absorbed into left.
        next[c.left] = next[c.right];
        if (next[c.left] < count) prev[next[c.left]] = c.left;
        if (prev[c.left] >= 0) push_candidate(tokenizer, tokens, heap, &heap_count, prev[c.left], c.left);
        if (next[c.left] < count) push_candidate(tokenizer, tokens, heap, &heap_count, c.left, next[c.left]);
    }

    int out = 0;
    for (int i = 0; i < count; i = next[i]) tokens[out++] = tokens[i];
    free(next);
    free(prev);
    free(heap);
    return out;
}

// Converts the three prompt segments from UTF-8 into vocabulary pieces, falls back to byte tokens when needed, applies BPE, and prepends <bos>.
int tokenize(const Tokenizer *tokenizer, const char *segments[3], int *tokens, int capacity) {
    int count = 1;

    for (int segment = 0; segment < 3; segment++)
        for (const char *cursor = segments[segment]; *cursor;) {
            if (count >= capacity) return -1;
            int special = -1;
            if (*cursor == '<')
                for (int i = 0; i < tokenizer->special_count && special < 0; i++) {
                    int length = (int)strlen(tokenizer->specials[i].token);
                    if (!strncmp(cursor, tokenizer->specials[i].token, length)) { special = tokenizer->specials[i].id; cursor += length; }
                }
            if (special >= 0) { tokens[count++] = special; continue; }
            char piece[8] = {0};
            if (*cursor == ' ') { memcpy(piece, "\xE2\x96\x81", 3); cursor++; } // SentencePiece represents spaces with U+2581.
            else {
                piece[0] = *cursor++;
                if ((piece[0] & 0xC0) == 0xC0)
                    for (int i = 1; i < 4 && (*cursor & 0xC0) == 0x80; i++) piece[i] = *cursor++;
            }
            const LookupEntry *entry = bsearch(piece, tokenizer->encode_vocab, tokenizer->encode_vocab_count,
                                      sizeof(LookupEntry), compare_lookup_keys);
            if (entry) { tokens[count++] = entry->result; continue; }

            for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++) {
                if (count >= capacity) return -1;
                tokens[count++] = 238 + *byte; // Byte tokens occupy IDs 238 through 493.
            }
        }

    count = 1 + apply_bpe_merges(tokenizer, tokens + 1, count - 1);
    tokens[0] = 2; // Token 2 is <bos>.
    return count;
}

const char *token_text(const Tokenizer *tokenizer, int token) {
    return token >= 0 && token < VOCAB_SIZE ? tokenizer->decoded_tokens[token] : "";
}

// ----------------------------------------------------------------------------
// Model

// data and scales begin as file offsets and become pointers after the model is memory-mapped.
// How a tensor's data is stored. The exporter's files hold only TENSOR_PACKED
// and plain float data, and leave the type slot zero.
enum { TENSOR_PACKED, TENSOR_Q4_0, TENSOR_Q6_K };

typedef struct {
    void *data;
    uint16_t *scales;
    int shape[3];
    int type;
} Tensor;

typedef struct {
    Tensor input_layernorm;
    Tensor layer_scalar;
    Tensor pre_ffn_layernorm;
    Tensor post_attn_layernorm;
    Tensor post_ffn_layernorm;
    Tensor post_per_layer_input_norm;
    Tensor per_layer_input_gate;
    Tensor per_layer_projection;
    Tensor q_norm;
    Tensor k_norm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor rope_cos;
    Tensor rope_sin;
} LayerWeights;

typedef struct {
    Tensor embed;
    Tensor embed_per_layer;
    LayerWeights layers[NUM_LAYERS];
    Tensor norm;
    Tensor per_layer_model_projection;
    Tensor per_layer_projection_norm;
    Tensor gelu_table;
} ModelWeights;

typedef struct {
    float residual[BATCH_SIZE * HIDDEN_SIZE];                           // Carries each token's hidden state through every layer.
    float hidden[BATCH_SIZE * 8 * HIDDEN_SIZE];                         // Reused for intermediate results and sized for the largest MLP output.
    float auxiliary[BATCH_SIZE * 8 * HIDDEN_SIZE];                      // Holds a second intermediate when attention or the MLP needs two results at once.
    int8_t quantized[BATCH_SIZE * 8 * HIDDEN_SIZE];                     // Holds the current linear input after dynamic int8 quantization.
    float activation_scales[BATCH_SIZE * 8 * HIDDEN_SIZE / 64];         // Stores one float scale for every 64 quantized values.
    float per_layer_inputs[BATCH_SIZE * NUM_LAYERS * 256];              // Stores one 256-value conditioning vector for every token and layer.
    float sliding_cache[CACHE_SETS][FULL_INTERVAL - 1][2 * (SLIDING_WINDOW + BATCH_SIZE) * SLIDING_KV]; // Keeps the previous window and current batch for the sliding KV caches.
    float full_cache[CACHE_SETS][2 * MAX_CONTEXT * FULL_KV];            // Holds the complete context for the full-attention KV caches.
    int token_ids[MAX_CONTEXT];                                         // Holds the tokenized prompt before prefill.
} InferenceState;

typedef struct {
    char magic[4];
    Tokenizer tokenizer;
    ModelWeights weights;
} Model;

// Verifies that the compiler laid out the memory-mapped model exactly as the exporter expects.
_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void *) == 8 && sizeof(VocabEntry) == 100 && offsetof(VocabEntry, id) == 96 && sizeof(LookupEntry) == 16 && sizeof(Tokenizer) == 33429932 && sizeof(Tensor) == 32 && sizeof(ModelWeights) == (6 + 19 * NUM_LAYERS) * sizeof(Tensor) && offsetof(Model, weights) == 33429936 && sizeof(Model) == offsetof(Model, weights) + sizeof(ModelWeights) && BATCH_SIZE == SLIDING_WINDOW && !((SLIDING_WINDOW + BATCH_SIZE) & (SLIDING_WINDOW + BATCH_SIZE - 1)) && !(MAX_CONTEXT & (MAX_CONTEXT - 1)), "MOG ABI mismatch");

// ----------------------------------------------------------------------------
// Kernels

// Uses one OpenMP thread per physical core because each core already uses SIMD, unless OMP_NUM_THREADS overrides it.
static inline int thread_count(void) {
    if (getenv("OMP_NUM_THREADS")) return omp_get_max_threads();
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0xB, 0, eax, ebx, ecx, edx);
    int threads_per_core = ebx & 0xffff;
    return omp_get_num_procs() / (threads_per_core ? threads_per_core : 1);
#elif defined(__APPLE__)
    int performance_cores = 0; // Efficiency cores would stall the static OpenMP schedules, so use only performance cores.
    size_t size = sizeof(performance_cores);
    if (!sysctlbyname("hw.perflevel0.logicalcpu", &performance_cores, &size, NULL, 0) && performance_cores > 0)
        return performance_cores;
    return omp_get_num_procs();
#else
    return omp_get_num_procs();
#endif
}

// Weight scales are stored as IEEE half-precision floats.
static inline float half_to_float(uint16_t bits) {
#if defined(__ARM_NEON)
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
#else
    return _cvtsh_ss(bits);
#endif
}

// Multiplies dynamically quantized activations by packed int8 weights in blocks of 16
// output rows. VNNI handles eight input rows at once while AVX2 handles four. Both
// accumulate integer dot products before restoring float values with their scales.
#if defined(__AVX512VNNI__)
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 8) {
        size_t active_rows = rows - row_start < 8 ? rows - row_start : 8;
        __m512 result[8] = {0};
        for (size_t group = 0; group < groups_per_row; group++) {
            __m512i dot[8] = {0};
            __m512i correction = _mm512_setzero_si512();
            __m512i offset_bytes = _mm512_set1_epi8(-128); // Flip signed activations into the unsigned range required by VNNI.
            for (int chunk = 0; chunk < 16; chunk++) {
                __m512i weight_values = _mm512_load_si512((const __m512i *)(packed_weights + group * 1024 + chunk * 64));
                correction = _mm512_dpbusd_epi32(correction, offset_bytes, weight_values);
                for (size_t row = 0; row < active_rows; row++) {
                    __m512i input_values = _mm512_broadcastd_epi32(
                        _mm_loadu_si32(input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4));
                    input_values = _mm512_xor_si512(input_values, offset_bytes);
                    dot[row] = _mm512_dpbusd_epi32(dot[row], input_values, weight_values);
                }
            }
            __m512 weight_scales = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(weight->scales + (output_block * groups_per_row + group) * block_rows)));
            for (size_t row = 0; row < active_rows; row++)
                result[row] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(dot[row], correction)), _mm512_mul_ps(weight_scales, _mm512_set1_ps(input_scales[(row_start + row) * groups_per_row + group])), result[row]);
        }
        for (size_t row = 0; row < active_rows; row++)
            _mm512_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows, result[row]);
    }
}
#elif defined(__ARM_NEON)
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 4) {
        size_t active_rows = rows - row_start < 4 ? rows - row_start : 4;
        float32x4_t result[4][4] = {0};
        for (size_t group = 0; group < groups_per_row; group++) {
            int32x4_t dot[4][4] = {0};
            for (int chunk = 0; chunk < 16; chunk++) {
                int8x16_t weight_values[4]; // Each vector holds four consecutive input values for four of the sixteen output rows.
                for (int quad = 0; quad < 4; quad++)
                    weight_values[quad] = vld1q_s8(packed_weights + group * 1024 + chunk * 64 + quad * 16);
                for (size_t row = 0; row < active_rows; row++) {
                    int32_t input_word;
                    memcpy(&input_word, input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4, sizeof(input_word));
                    int8x16_t input_values = vreinterpretq_s8_s32(vdupq_n_s32(input_word));
                    for (int quad = 0; quad < 4; quad++)
                        dot[row][quad] = vdotq_s32(dot[row][quad], weight_values[quad], input_values);
                }
            }
            const float16_t *scale_halves = (const float16_t *)(weight->scales + (output_block * groups_per_row + group) * block_rows);
            for (size_t row = 0; row < active_rows; row++) {
                float32x4_t input_scale = vdupq_n_f32(input_scales[(row_start + row) * groups_per_row + group]);
                for (int quad = 0; quad < 4; quad++)
                    result[row][quad] = vfmaq_f32(result[row][quad], vcvtq_f32_s32(dot[row][quad]),
                                                  vmulq_f32(vcvt_f32_f16(vld1_f16(scale_halves + quad * 4)), input_scale));
            }
        }
        for (size_t row = 0; row < active_rows; row++)
            for (int quad = 0; quad < 4; quad++)
                vst1q_f32(output + (row_start + row) * weight->shape[0] + output_block * block_rows + quad * 4, result[row][quad]);
    }
}
#else
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 4) {
        size_t active_rows = rows - row_start < 4 ? rows - row_start : 4;
        for (int half = 0; half < 2; half++) {
            __m256 result[4] = {0};
            for (size_t group = 0; group < groups_per_row; group++) {
                __m256i dot[4] = {0};
                for (int chunk = 0; chunk < 16; chunk++) {
                    __m256i weight_values = _mm256_loadu_si256((const __m256i *)(packed_weights + group * 1024 + chunk * 64 + half * 32));
                    __m256i weight_magnitudes = _mm256_abs_epi8(weight_values);
                    for (size_t row = 0; row < active_rows; row++) {
                        __m256i input_values = _mm256_broadcastd_epi32(
                            _mm_loadu_si32(input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4));
                        dot[row] = _mm256_add_epi32(dot[row], _mm256_madd_epi16(_mm256_maddubs_epi16(weight_magnitudes, _mm256_sign_epi8(input_values, weight_values)), _mm256_set1_epi16(1)));
                    }
                }
                __m256 weight_scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(weight->scales + (output_block * groups_per_row + group) * block_rows + half * 8)));
                for (size_t row = 0; row < active_rows; row++)
                    result[row] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot[row]), _mm256_mul_ps(weight_scales, _mm256_set1_ps(input_scales[(row_start + row) * groups_per_row + group])), result[row]);
            }
            for (size_t row = 0; row < active_rows; row++)
                _mm256_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows + half * 8, result[row]);
        }
    }
}
#endif

void matmul_q4_0(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows);

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    if (weight->type == TENSOR_Q4_0) { matmul_q4_0(output, input_q, input_scales, weight, rows); return; }
    const size_t block_rows = 16;
    #pragma omp for schedule(static)
    for (size_t output_block = 0; output_block < (size_t)weight->shape[0] / block_rows; output_block++)
        matmul_block(output, input_q, input_scales, weight, rows, output_block);
}

// Converts each input row to int8 in groups of 64 values with a float scale recording each group's magnitude.
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 64); group_index++) {
        const float *group = input + group_index * 64;
        float max_abs = 0.0f;
        for (int j = 0; j < 64; j++) {
            float value = fabsf(group[j]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < 64; j++) quantized[group_index * 64 + j] = (int8_t)rintf(group[j] * inverse_scale);
        scales[group_index] = scale;
    }
}

// The caches interleave every KV head's channels per position, so entries are
// cache_stride floats apart while each head's slice spans head_dim of them.
#if defined(__ARM_NEON)
void attention_scores(float *scores, const float *query, const float *key_cache,
        int first_key, int num_keys, int cache_mask, int head_dim, int cache_stride) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + (size_t)((first_key + key_index) & cache_mask) * cache_stride;
        float32x4_t sum[4] = {0};
        for (int j = 0; j < head_dim; j += 16)
            for (int u = 0; u < 4; u++)
                sum[u] = vfmaq_f32(sum[u], vld1q_f32(query + j + u * 4), vld1q_f32(key + j + u * 4));
        scores[key_index] = vaddvq_f32(vaddq_f32(vaddq_f32(sum[0], sum[1]), vaddq_f32(sum[2], sum[3])));
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
        int first_key, int num_keys, int cache_mask, int head_dim, int cache_stride) {
    for (int j = 0; j < head_dim; j += 32) {
        float32x4_t sum[8] = {0};
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache + (size_t)((first_key + key_index) & cache_mask) * cache_stride + j;
            float32x4_t probability = vdupq_n_f32(probabilities[key_index]);
            for (int u = 0; u < 8; u++)
                sum[u] = vfmaq_f32(sum[u], probability, vld1q_f32(value + u * 4));
        }
        for (int u = 0; u < 8; u++) vst1q_f32(output + j + u * 4, sum[u]);
    }
}
#else
void attention_scores(float *scores, const float *query, const float *key_cache,
        int first_key, int num_keys, int cache_mask, int head_dim, int cache_stride) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + (size_t)((first_key + key_index) & cache_mask) * cache_stride;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        for (int j = 0; j < head_dim; j += 16) {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + j), _mm256_loadu_ps(key + j), sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(query + j + 8), _mm256_loadu_ps(key + j + 8), sum1);
        }
        __m256 sum8 = _mm256_add_ps(sum0, sum1);
        __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
        sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        scores[key_index] = _mm_cvtss_f32(_mm_add_ss(sum4, _mm_movehdup_ps(sum4)));
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
        int first_key, int num_keys, int cache_mask, int head_dim, int cache_stride) {
    for (int j = 0; j < head_dim; j += 64) {
        __m256 sum[8] = {0};
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache + (size_t)((first_key + key_index) & cache_mask) * cache_stride + j;
            __m256 probability = _mm256_set1_ps(probabilities[key_index]);
            for (int u = 0; u < 8; u++)
                sum[u] = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + u * 8), sum[u]);
        }
        for (int u = 0; u < 8; u++) _mm256_storeu_ps(output + j + u * 8, sum[u]);
    }
}
#endif

// Approximates GELU from the exported lookup table and multiplies it by the up projection to produce the MLP's gated activation.
void geglu(float *gate, const float *up, int rows, int width, int up_stride, const Tensor *gelu_table) {
    const float *table = (const float *)gelu_table->data;
    const int table_size = gelu_table->shape[0];
    const float lower = (float)gelu_table->shape[1];
    const float upper = (float)gelu_table->shape[2];
    const float scale = (float)(table_size - 1) / (upper - lower);
    #pragma omp for collapse(2) schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lower) {
                x = table[0];
            } else if (!(x >= upper)) {
                float position = (x - lower) * scale;
                int index = (int)position;
                float fraction = position - (float)index;
                x = table[index] + fraction * (table[index + 1] - table[index]);
            }
            gate[row * width + i] = x * up[row * up_stride + i];
        }
    }
}

// ----------------------------------------------------------------------------
// Transformer

// Looks up packed int8 embedding rows and dequantizes them directly without materializing the full embedding table.
void dequantize_q6_K(float *output, const void *blocks, int width, float multiplier);

void embedding(float *output, const Tensor *table, const int *tokens, size_t token_count, float multiplier) {
    const int block_rows = 16;
    int width = table->shape[1];
    if (table->type == TENSOR_Q6_K) {
        size_t row_bytes = (size_t)width / 256 * 210;
        #pragma omp for schedule(static)
        for (size_t token = 0; token < token_count; token++)
            dequantize_q6_K(output + token * width, (const uint8_t *)table->data + (size_t)tokens[token] * row_bytes, width, multiplier);
        return;
    }
    int groups = width / 64;
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        size_t block = (size_t)(tokens[token] / block_rows);
        int row = tokens[token] % block_rows;
        float *vector = output + token * width;
        const int8_t *block_data = (const int8_t *)table->data + block * block_rows * width;
        const uint16_t *block_scales = table->scales + block * groups * block_rows;
        for (size_t group_index = 0; group_index < (size_t)groups; group_index++) {
            const int8_t *group = block_data + group_index * block_rows * 64;
            float scale = half_to_float(block_scales[group_index * block_rows + row]) * multiplier;
            for (int j = 0; j < 64; j++) {
                int chunk = j / 4;
                int offset = j % 4;
                vector[group_index * 64 + j] = (float)group[chunk * block_rows * 4 + row * 4 + offset] * scale;
            }
        }
    }
}

void rmsnorm(float *output, const float *input, const Tensor *weight, int width, float epsilon, size_t row_count) {
    const float *weights = weight ? (const float *)weight->data : NULL;
    #pragma omp for schedule(static)
    for (size_t row = 0; row < row_count; row++) {
        const float *input_row = input + row * width;
        float *output_row = output + row * width;
        float sum_squares = 0.0f;
        for (int i = 0; i < width; i++)
            sum_squares += input_row[i] * input_row[i];
        float inverse_rms = 1.0f / sqrtf(sum_squares / (float)width + epsilon);
        for (int i = 0; i < width; i++)
            output_row[i] = (weights ? weights[i] : 1.0f) * (inverse_rms * input_row[i]);
    }
}

void add_and_scale(float *output, const float *addend, size_t count, float scale) {
    #pragma omp for schedule(static)
    for (size_t i = 0; i < count; i++) output[i] = (output[i] + addend[i]) * scale;
}

// Rotates pairs of query or key channels using each position's sine and cosine values so attention can distinguish token order.
void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors,
                int num_heads, int head_dim, int start_pos, size_t token_count) {
    int pairs = cosines->shape[1];
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        const float *cosine = (float *)cosines->data + (start_pos + token) * pairs;
        const float *sine = (float *)sines->data + (start_pos + token) * pairs;
        for (size_t head = 0; head < (size_t)num_heads; head++) {
            float *vector = vectors + (token * num_heads + head) * head_dim;
            for (int j = 0; j < pairs; j++) {
                float first = vector[j];
                float second = vector[j + head_dim / 2];
                vector[j] = first * cosine[j] - second * sine[j];
                vector[j + head_dim / 2] = second * cosine[j] + first * sine[j];
            }
        }
    }
}

void softmax(float *values, int count) {
    float max = values[0], sum = 1.0f;
    for (int i = 1; i < count; i++) {
        if (values[i] > max) { sum = sum * expf(max - values[i]) + 1.0f; max = values[i]; } // Rescale the sum when a new maximum appears so expf() stays in range.
        else sum += expf(values[i] - max);
    }
    for (int i = 0; i < count; i++) values[i] = expf(values[i] - max) / sum;
}

// Builds queries, updates the KV cache, and computes causal attention over 512 tokens or the full context while shared layers reuse the latest compatible cache.
void attention(InferenceState *state, const LayerWeights *layers, int layer,
               int start_pos, size_t token_count, float *scores) {
    const LayerWeights *weights = &layers[layer];
    int full_attention = layer % FULL_INTERVAL == FULL_INTERVAL - 1;
    int cache_len = full_attention ? MAX_CONTEXT : SLIDING_WINDOW + BATCH_SIZE;
    int cache_mask = cache_len - 1; // Both cache lengths are powers of two, so masking wraps positions without division.
    int head_dim = weights->q_norm.shape[0];
    int query_width = weights->q_proj.shape[0];
    int cache_owner = layer;
    while (!layers[cache_owner].k_proj.data || (cache_owner % FULL_INTERVAL == FULL_INTERVAL - 1) != full_attention) cache_owner--; // Shared layers reuse the latest cache of the same attention type.
    int kv_width = layers[cache_owner].k_proj.shape[0]; // Every KV head's channels, interleaved per cache position.
    int kv_heads = kv_width / head_dim;
    float *key_cache = full_attention ? state->full_cache[cache_owner / FULL_INTERVAL] : state->sliding_cache[cache_owner / FULL_INTERVAL][cache_owner % FULL_INTERVAL];
    float *value_cache = key_cache + (size_t)cache_len * kv_width;

    quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->q_proj.shape[1]);
    matmul_int8(state->auxiliary, state->quantized, state->activation_scales, &weights->q_proj, token_count);
    rmsnorm(state->auxiliary, state->auxiliary, &weights->q_norm, head_dim, 1e-6f, token_count * (query_width / head_dim));
    apply_rope(&weights->rope_cos, &weights->rope_sin, state->auxiliary, query_width / head_dim, head_dim, start_pos, token_count);

    if (weights->k_proj.data) {
        float *new_keys = key_cache + ((size_t)start_pos & cache_mask) * kv_width;
        float *new_values = value_cache + ((size_t)start_pos & cache_mask) * kv_width;
        matmul_int8(new_keys, state->quantized, state->activation_scales, &weights->k_proj, token_count);
        matmul_int8(new_values, state->quantized, state->activation_scales, &weights->v_proj, token_count);
        rmsnorm(new_keys, new_keys, &weights->k_norm, head_dim, 1e-6f, token_count * kv_heads);
        rmsnorm(new_values, new_values, NULL, head_dim, 1e-6f, token_count * kv_heads); // Value vectors are normalized without a learned weight.
        apply_rope(&weights->rope_cos, &weights->rope_sin, new_keys, kv_heads, head_dim, start_pos, token_count);
    }

    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (size_t head = 0; head < (size_t)(query_width / head_dim); head++) {
        for (size_t token = 0; token < token_count; token++) {
            int first_key = !full_attention && start_pos + (int)token + 1 > SLIDING_WINDOW ? start_pos + (int)token + 1 - SLIDING_WINDOW : 0;
            int num_keys = start_pos + (int)token + 1 - first_key;
            size_t kv_offset = head / (query_width / head_dim / kv_heads) * (size_t)head_dim; // Query heads share KV heads in contiguous groups.
            float *head_output = state->hidden + token * query_width + head * head_dim;
            const float *query = state->auxiliary + token * query_width + head * head_dim;
            attention_scores(scores, query, key_cache + kv_offset, first_key, num_keys, cache_mask, head_dim, kv_width);
            softmax(scores, num_keys);
            weighted_value_sum(head_output, scores, value_cache + kv_offset, first_key, num_keys, cache_mask, head_dim, kv_width);
        }
    }

    quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->o_proj.shape[1]);
    matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->o_proj, token_count);
}

void forward(Model *model, InferenceState *state, const int *tokens, size_t token_count, int start_pos) {
    int per_layer_width = model->weights.per_layer_projection_norm.shape[0];
    // One OpenMP team stays alive for the full forward pass while each kernel divides its own loop.
    #pragma omp parallel num_threads(thread_count())
    {
    float scores[(size_t)start_pos + token_count]; // Each thread needs private scratch large enough for every visible key.
    embedding(state->residual, &model->weights.embed, tokens, token_count, sqrtf((float)HIDDEN_SIZE));

    // Build the token-conditioned input that each transformer layer will receive.
    quantize(state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
    matmul_int8(state->per_layer_inputs, state->quantized, state->activation_scales, &model->weights.per_layer_model_projection, token_count);
    rmsnorm(state->per_layer_inputs, state->per_layer_inputs, &model->weights.per_layer_projection_norm, per_layer_width, 1e-6f * HIDDEN_SIZE, token_count * NUM_LAYERS);

    embedding(state->hidden, &model->weights.embed_per_layer, tokens, token_count, sqrtf((float)per_layer_width));
    add_and_scale(state->per_layer_inputs, state->hidden, token_count * NUM_LAYERS * per_layer_width,
               1.0f / sqrtf(2.0f)); // Keeps the variance of the combined inputs unchanged.

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights *weights = &model->weights.layers[layer];

        rmsnorm(state->hidden, state->residual, &weights->input_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        attention(state, model->weights.layers, layer, start_pos, token_count, scores);
        rmsnorm(state->hidden, state->hidden, &weights->post_attn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        rmsnorm(state->hidden, state->residual, &weights->pre_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->gate_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->gate_proj, token_count);
        matmul_int8(state->auxiliary, state->quantized, state->activation_scales, &weights->up_proj, token_count);
        geglu(state->hidden, state->auxiliary, token_count, weights->gate_proj.shape[0], weights->gate_proj.shape[0], &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->down_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->down_proj, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);
        // Gate and project this layer's conditioning input before adding it to the residual stream with a learned scale.
        quantize(state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->per_layer_input_gate, token_count);
        geglu(state->hidden, state->per_layer_inputs + layer * per_layer_width, token_count, per_layer_width, NUM_LAYERS * per_layer_width, &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->per_layer_projection.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->per_layer_projection, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_per_layer_input_norm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE,
                   ((float *)weights->layer_scalar.data)[0]);
    }
    }
}

// Reuses the embedding matrix to turn the final token representation into vocabulary logits, then applies Gemma's tanh soft cap.
float *logits(Model *model, InferenceState *state, size_t token) {
    #pragma omp parallel num_threads(thread_count())
    {
        rmsnorm(state->hidden, state->residual + token * HIDDEN_SIZE, &model->weights.norm, HIDDEN_SIZE, 1e-6f, 1);
        quantize(state->quantized, state->activation_scales, state->hidden, 1, HIDDEN_SIZE);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &model->weights.embed, 1);
        #pragma omp for schedule(static)
        for (int i = 0; i < VOCAB_SIZE; i++) state->hidden[i] = 30.0f * tanhf(state->hidden[i] / 30.0f);
    }
    return state->hidden;
}

// ----------------------------------------------------------------------------
// Generation

unsigned long long rng_state = 42;

float random_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((rng_state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

int sample(float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    struct { float score; int token; } top[64]; // Sampling considers only the 64 highest logits.
    for (int i = 0; i < 64; i++) top[i].score = -INFINITY;
    for (int token = 0; token < vocab_size; token++) {
        if (logits[token] <= top[63].score) continue;
        int i = 63;
        while (i > 0 && logits[token] > top[i - 1].score) { top[i] = top[i - 1]; i--; }
        top[i].score = logits[token]; top[i].token = token;
    }
    float sum = 0.0f, max = top[0].score / temperature;
    for (int i = 0; i < 64; i++) sum += top[i].score = expf(top[i].score / temperature - max);
    float mass = 0.0f;
    int count = 0;
    while (mass < 0.95f * sum) mass += top[count++].score; // Keep the smallest prefix containing 95% of the top-64 probability mass.
    float threshold = random_uniform() * mass;
    for (int i = 0; i < count; i++)
        if ((threshold -= top[i].score) <= 0.0f) return top[i].token;
    return top[count - 1].token;
}

void prefill(Model *model, InferenceState *state, const int *tokens, int token_count, int start_pos, int dump_logits) {
    for (int offset = 0; offset < token_count; offset += BATCH_SIZE) {
        int chunk = token_count - offset < BATCH_SIZE ? token_count - offset : BATCH_SIZE;
        forward(model, state, tokens + offset, chunk, start_pos + offset);
        if (dump_logits) {
            for (int i = 0; i < chunk; i++) {
                fwrite(logits(model, state, i), sizeof(float), VOCAB_SIZE, stdout);
            }
        }
    }
}

void generate(Model *model, InferenceState *state, const char *prompt,
              int max_new_tokens, float temperature, int dump_logits) {
    Tokenizer *tokenizer = &model->tokenizer;
    int styled = !dump_logits && isatty(STDOUT_FILENO);

    if (max_new_tokens < 0) {
        fprintf(stderr, "-n must be non-negative\n");
        exit(1);
    }
    const char *segments[3] = {dump_logits ? "" : "<|turn>user\n", prompt,
                               dump_logits ? "" : "<turn|>\n<|turn>model\n"};
    int prompt_tokens = tokenize(tokenizer, segments, state->token_ids, MAX_CONTEXT);
    if (prompt_tokens < 0) {
        fprintf(stderr, "prompt exceeds the %d-token context limit\n", MAX_CONTEXT);
        exit(1);
    }

    if (styled) {
        fputs("\n\033[2;36m────────────────────────────────\033[0m\n", stdout);
        fflush(stdout);
    }

    prefill(model, state, state->token_ids, prompt_tokens, 0, dump_logits);
    if (dump_logits) return;

    int end = prompt_tokens + max_new_tokens;
    if (end > MAX_CONTEXT || end < prompt_tokens) end = MAX_CONTEXT;
    for (int position = prompt_tokens; position < end; position++) {
        int next_token = sample(logits(model, state, position == prompt_tokens ? (prompt_tokens - 1) % BATCH_SIZE : 0), VOCAB_SIZE, temperature);
        if (next_token == 1 || next_token == 106) break; // Stop at <eos> or <turn|>.

        fputs(token_text(tokenizer, next_token), stdout);
        fflush(stdout);
        forward(model, state, &next_token, 1, position);
    }
    putchar('\n');
}

double time_seconds(void) {
#ifdef _WIN32
    static double frequency;
    LARGE_INTEGER ticks;
    if (!frequency) {
        QueryPerformanceFrequency(&ticks);
        frequency = (double)ticks.QuadPart;
    }
    QueryPerformanceCounter(&ticks);
    return (double)ticks.QuadPart / frequency;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#endif
}

void benchmark(Model *model, InferenceState *state, int prefill_tokens, int generated_tokens) {
    if (prefill_tokens > 0) {
        for (int i = 0; i < prefill_tokens; i++)
            state->token_ids[i] = 2 + i % 1000;
        double start = time_seconds();
        prefill(model, state, state->token_ids, prefill_tokens, 0, 0);
        printf("pp%d %.2f tok/s\n", prefill_tokens, (double)prefill_tokens / (time_seconds() - start));
    }
    if (generated_tokens > 0) {
        int token = 2;
        double start = time_seconds();
        for (int position = 0; position < generated_tokens; position++) {
            forward(model, state, &token, 1, position);
            token = sample(logits(model, state, 0), VOCAB_SIZE, 0.0f);
        }
        printf("tg%d %.2f tok/s\n", generated_tokens, (double)generated_tokens / (time_seconds() - start));
    }
}

// ----------------------------------------------------------------------------
// GGUF models
//
// Loads Google's quantization-aware checkpoint in llama.cpp's format,
// gemma-4-E2B_q4_0-it.gguf, rather than GGUF in general. Its transformer
// matrices are q4_0, multiplied in place. Its two embedding tables are q6_K:
// the per-layer table is read in place a row at a time, while the token
// table, which is also the output projection, is converted at load to the
// packed int8 layout, as is the f16 per-layer model projection. The
// tokenizer's lookup tables and the RoPE and GELU tables, which an exported
// file carries, are built at load.

typedef struct {
    uint16_t scale;
    uint8_t values[16]; // Low nibbles hold inputs 0-15, high nibbles 16-31, offset by 8.
} BlockQ4_0;

_Static_assert(sizeof(BlockQ4_0) == 18, "q4_0 block layout");

enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q6_K = 14 };

static inline uint16_t float_to_half(float value) {
#if defined(__ARM_NEON)
    __fp16 half = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
#else
    return _cvtss_sh(value, 0);
#endif
}

// Expands one row of q6_K super-blocks: 256 values each, as 4 low and 2 high
// bits, with a signed 8-bit scale per 16 values and an f16 scale overall.
void dequantize_q6_K(float *output, const void *blocks, int width, float multiplier) {
    const uint8_t *block = blocks;
    for (int b = 0; b < width / 256; b++, block += 210) {
        const uint8_t *low = block, *high = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        uint16_t bits;
        memcpy(&bits, block + 208, sizeof(bits));
        float scale = half_to_float(bits) * multiplier;
        float *y = output + b * 256;
        for (int half = 0; half < 2; half++, y += 128, low += 64, high += 32, scales += 8) {
            for (int l = 0; l < 32; l++) {
                int s = l / 16;
                y[l + 0]  = scale * scales[s + 0] * (((low[l] & 0xF)      | ((high[l] >> 0 & 3) << 4)) - 32);
                y[l + 32] = scale * scales[s + 2] * (((low[l + 32] & 0xF) | ((high[l] >> 2 & 3) << 4)) - 32);
                y[l + 64] = scale * scales[s + 4] * (((low[l] >> 4)       | ((high[l] >> 4 & 3) << 4)) - 32);
                y[l + 96] = scale * scales[s + 6] * (((low[l + 32] >> 4)  | ((high[l] >> 6 & 3) << 4)) - 32);
            }
        }
    }
}

// Multiplies int8 activations, quantized in groups of 64, by q4_0 weights
// rearranged at load (pack_q4) so that every 16 bytes hold 4 consecutive
// inputs for 8 output rows: rows 0-3 in the low nibbles, rows 4-7 in the
// high. One unpacked vector then serves a whole tile of tokens, and each
// row's dot product builds up in its own lane, with no horizontal sums.
// Per 8-row group and 32-input block: 8 such chunks, then elsewhere 8 f16
// scales, one per row.
#define Q4_ROWS 8

#if defined(__ARM_NEON)
#define Q4_TILE 4
static inline __attribute__((always_inline)) void matmul_q4_group(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t first_token, int tokens, size_t group) {
    size_t width = (size_t)weight->shape[1], blocks = width / 32, groups = width / 64;
    const uint8_t *packed = (const uint8_t *)weight->data + group * blocks * 128;
    const float16_t *scales = (const float16_t *)weight->scales + group * blocks * 8;
    const int8_t *input = input_q + first_token * width;
    const float *input_scale = input_scales + first_token * groups;
    float32x4_t low_sum[Q4_TILE], high_sum[Q4_TILE];
    for (int t = 0; t < tokens; t++) low_sum[t] = high_sum[t] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < blocks; b++) {
        int32x4_t low_dot[Q4_TILE], high_dot[Q4_TILE];
        for (int t = 0; t < tokens; t++) low_dot[t] = high_dot[t] = vdupq_n_s32(0);
        int8x16_t low[8], high[8];
        for (int c = 0; c < 8; c++) {
            uint8x16_t p = vld1q_u8(packed + b * 128 + c * 16);
            low[c] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p, vdupq_n_u8(0xF))), vdupq_n_s8(8));
            high[c] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p, 4)), vdupq_n_s8(8));
        }
        for (int t = 0; t < tokens; t++) {
            int8x16_t first = vld1q_s8(input + t * width + b * 32), second = vld1q_s8(input + t * width + b * 32 + 16);
#define Q4_LANE(c, half, lane) \
            low_dot[t] = vdotq_laneq_s32(low_dot[t], low[c], half, lane); \
            high_dot[t] = vdotq_laneq_s32(high_dot[t], high[c], half, lane);
            Q4_LANE(0, first, 0) Q4_LANE(1, first, 1) Q4_LANE(2, first, 2) Q4_LANE(3, first, 3)
            Q4_LANE(4, second, 0) Q4_LANE(5, second, 1) Q4_LANE(6, second, 2) Q4_LANE(7, second, 3)
#undef Q4_LANE
        }
        float32x4_t low_scale = vcvt_f32_f16(vld1_f16(scales + b * 8));
        float32x4_t high_scale = vcvt_f32_f16(vld1_f16(scales + b * 8 + 4));
        for (int t = 0; t < tokens; t++) {
            float s = input_scale[t * groups + b / 2];
            low_sum[t] = vfmaq_f32(low_sum[t], vcvtq_f32_s32(low_dot[t]), vmulq_n_f32(low_scale, s));
            high_sum[t] = vfmaq_f32(high_sum[t], vcvtq_f32_s32(high_dot[t]), vmulq_n_f32(high_scale, s));
        }
    }
    for (int t = 0; t < tokens; t++) {
        float *out = output + (first_token + t) * weight->shape[0] + group * Q4_ROWS;
        vst1q_f32(out, low_sum[t]);
        vst1q_f32(out + 4, high_sum[t]);
    }
}
#else
#define Q4_TILE 6
// maddubs needs one unsigned operand: the weights' magnitudes, with their
// signs moved onto the activations. Its 16-bit sums cannot saturate within
// a block (8 chunks of at most 2 * 8 * 127).
static inline __attribute__((always_inline)) void matmul_q4_group(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t first_token, int tokens, size_t group) {
    size_t width = (size_t)weight->shape[1], blocks = width / 32, groups = width / 64;
    const uint8_t *packed = (const uint8_t *)weight->data + group * blocks * 128;
    const uint16_t *scales = weight->scales + group * blocks * 8;
    const int8_t *input = input_q + first_token * width;
    const float *input_scale = input_scales + first_token * groups;
    __m256 sum[Q4_TILE];
    for (int t = 0; t < tokens; t++) sum[t] = _mm256_setzero_ps();
    for (size_t b = 0; b < blocks; b++) {
        __m256i dot[Q4_TILE];
        for (int t = 0; t < tokens; t++) dot[t] = _mm256_setzero_si256();
        for (int c = 0; c < 8; c++) {
            __m128i p = _mm_loadu_si128((const __m128i *)(packed + b * 128 + c * 16));
            __m128i low = _mm_and_si128(p, _mm_set1_epi8(0xF));
            __m128i high = _mm_and_si128(_mm_srli_epi16(p, 4), _mm_set1_epi8(0xF));
            __m256i w = _mm256_sub_epi8(_mm256_set_m128i(high, low), _mm256_set1_epi8(8));
            __m256i magnitude = _mm256_abs_epi8(w);
            for (int t = 0; t < tokens; t++) {
                int32_t four;
                memcpy(&four, input + t * width + b * 32 + c * 4, 4);
                __m256i a = _mm256_sign_epi8(_mm256_set1_epi32(four), w);
                dot[t] = _mm256_add_epi16(dot[t], _mm256_maddubs_epi16(magnitude, a));
            }
        }
        __m256 scale = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(scales + b * 8)));
        for (int t = 0; t < tokens; t++) {
            __m256 d = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot[t], _mm256_set1_epi16(1)));
            sum[t] = _mm256_fmadd_ps(d, _mm256_mul_ps(scale, _mm256_set1_ps(input_scale[t * groups + b / 2])), sum[t]);
        }
    }
    for (int t = 0; t < tokens; t++)
        _mm256_storeu_ps(output + (first_token + t) * weight->shape[0] + group * Q4_ROWS, sum[t]);
}
#endif

// Each thread keeps its share of the row groups and walks the tokens a tile
// at a time, so a tile's activations stay in L1 while its weights stream
// from L2. Called inside a parallel region; ends with a barrier like omp for.
void matmul_q4_0(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    size_t total = (size_t)weight->shape[0] / Q4_ROWS;
    size_t threads = (size_t)omp_get_num_threads(), id = (size_t)omp_get_thread_num();
    size_t first = total * id / threads, last = total * (id + 1) / threads;
    size_t token = 0;
    for (; token + Q4_TILE <= rows; token += Q4_TILE)
        for (size_t group = first; group < last; group++)
            matmul_q4_group(output, input_q, input_scales, weight, token, Q4_TILE, group);
    for (; token < rows; token++)
        for (size_t group = first; group < last; group++)
            matmul_q4_group(output, input_q, input_scales, weight, token, 1, group);
    #pragma omp barrier
}

// A cursor over the file, which flags rather than overruns a truncated one.
typedef struct {
    const uint8_t *at, *end;
    int bad;
} Reader;

static uint64_t read_uint(Reader *r, int size) {
    uint64_t value = 0;
    if (r->end - r->at < size) { r->bad = 1; return 0; }
    memcpy(&value, r->at, (size_t)size); // GGUF is little-endian, as are the CPUs this runs on.
    r->at += size;
    return value;
}

typedef struct {
    const char *text;
    size_t len;
} Text;

static Text read_text(Reader *r) {
    uint64_t len = read_uint(r, 8);
    if ((uint64_t)(r->end - r->at) < len) { r->bad = 1; return (Text){"", 0}; }
    Text text = {(const char *)r->at, (size_t)len};
    r->at += len;
    return text;
}

static int text_is(Text text, const char *s) { return text.len == strlen(s) && !memcmp(text.text, s, text.len); }

// Skips a metadata value of the given type.
static void skip_value(Reader *r, uint32_t type) {
    static const int sizes[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type == 8) read_text(r);
    else if (type == 9) {
        uint32_t element = (uint32_t)read_uint(r, 4);
        uint64_t count = read_uint(r, 8);
        if (element < 13 && element != 8 && element != 9) {
            if ((uint64_t)(r->end - r->at) / sizes[element] < count) r->bad = 1;
            else r->at += count * sizes[element];
        } else for (uint64_t i = 0; i < count && !r->bad; i++) skip_value(r, element);
    } else if (type < 13) read_uint(r, sizes[type]);
    else r->bad = 1;
}

// A metadata array: its element type, length, and where its elements begin.
typedef struct {
    uint32_t type;
    uint64_t count;
    const uint8_t *at, *end;
} Array;

typedef struct {
    Text name;
    int dims[4], dim_count;
    uint32_t type;
    const uint8_t *data;
} GgufTensor;

typedef struct {
    GgufTensor *tensors;
    int count;
    const char *path;
    int missing;
} TensorIndex;

static const GgufTensor *find_tensor(TensorIndex *index, const char *name) {
    for (int i = 0; i < index->count; i++)
        if (text_is(index->tensors[i].name, name)) return &index->tensors[i];
    return NULL;
}

// Rearranges q4_0 blocks into matmul_q4_0's layout: for each group of 8
// rows and block of 32 inputs, chunk c's byte j holds input c*4 + j%4 of row
// j/4 in its low nibble and of row 4 + j/4 in its high nibble.
static void pack_q4(Tensor *t) {
    size_t groups = (size_t)t->shape[0] / Q4_ROWS, blocks = (size_t)t->shape[1] / 32;
    const BlockQ4_0 *source = t->data;
    uint8_t *packed = malloc(groups * blocks * 128);
    uint16_t *scales = malloc(groups * blocks * Q4_ROWS * sizeof(*scales));
    if (!packed || !scales) { perror("malloc"); exit(1); }
    #pragma omp parallel for num_threads(thread_count()) schedule(static)
    for (size_t g = 0; g < groups; g++)
        for (size_t b = 0; b < blocks; b++) {
            uint8_t *out = packed + (g * blocks + b) * 128;
            for (int r = 0; r < Q4_ROWS; r++) scales[(g * blocks + b) * Q4_ROWS + r] = source[(g * Q4_ROWS + r) * blocks + b].scale;
            for (int c = 0; c < 8; c++)
                for (int j = 0; j < 16; j++) {
                    int input = c * 4 + j % 4;
                    const BlockQ4_0 *low = &source[(g * Q4_ROWS + j / 4) * blocks + b];
                    const BlockQ4_0 *high = &source[(g * Q4_ROWS + 4 + j / 4) * blocks + b];
                    int low_nibble = input < 16 ? low->values[input] & 15 : low->values[input - 16] >> 4;
                    int high_nibble = input < 16 ? high->values[input] & 15 : high->values[input - 16] >> 4;
                    out[c * 16 + j] = (uint8_t)(low_nibble | high_nibble << 4);
                }
        }
    t->data = packed;
    t->scales = scales;
}

// Points t at a tensor of the given type and shape (outputs, inputs; inputs
// only for a vector), or leaves it empty when optional and absent.
static void bind_tensor(TensorIndex *index, Tensor *t, const char *name, uint32_t type, int optional) {
    const GgufTensor *g = find_tensor(index, name);
    if (!g) {
        if (!optional) { fprintf(stderr, "%s: missing tensor %s\n", index->path, name); index->missing = 1; }
        return;
    }
    if (g->type != type) { fprintf(stderr, "%s: tensor %s has type %u, expected %u\n", index->path, name, g->type, type); index->missing = 1; return; }
    t->data = (void *)g->data;
    t->shape[0] = g->dim_count > 1 ? g->dims[1] : g->dims[0]; // GGUF lists the innermost dimension first.
    t->shape[1] = g->dim_count > 1 ? g->dims[0] : 0;
    t->type = type == GGML_Q4_0 ? TENSOR_Q4_0 : type == GGML_Q6_K ? TENSOR_Q6_K : TENSOR_PACKED;
    if (type == GGML_Q4_0) {
        if (t->shape[0] % Q4_ROWS || t->shape[1] % 64) {
            fprintf(stderr, "%s: tensor %s has an unsupported shape\n", index->path, name);
            index->missing = 1;
            return;
        }
        pack_q4(t);
    }
}

// Converts a matrix, one source row at a time, to the exporter's packed int8
// layout: groups of 64 inputs with an f16 scale each, interleaved in blocks
// of 16 rows by 4 inputs.
static void pack_int8(Tensor *t, const void *source, int source_type) {
    size_t rows = (size_t)t->shape[0], width = (size_t)t->shape[1], groups = width / 64;
    int8_t *values = malloc(rows * width);
    uint16_t *scales = malloc(rows * groups * sizeof(*scales));
    if (!values || !scales) { perror("malloc"); exit(1); }
    #pragma omp parallel num_threads(thread_count())
    {
    float *row_values = malloc(width * sizeof(float));
    #pragma omp for schedule(static)
    for (size_t row = 0; row < rows; row++) {
        if (source_type == GGML_Q6_K) dequantize_q6_K(row_values, (const uint8_t *)source + row * (width / 256 * 210), (int)width, 1.0f);
        else for (size_t i = 0; i < width; i++) row_values[i] = half_to_float(((const uint16_t *)source)[row * width + i]);
        size_t block = row / 16, in_block = row % 16;
        for (size_t g = 0; g < groups; g++) {
            float max_abs = 0.0f;
            for (int j = 0; j < 64; j++) if (fabsf(row_values[g * 64 + j]) > max_abs) max_abs = fabsf(row_values[g * 64 + j]);
            uint16_t bits = float_to_half(fmaxf(max_abs / 127.0f, 6.104e-05f)); // Quantize against the scale as stored.
            float scale = half_to_float(bits);
            scales[(block * groups + g) * 16 + in_block] = bits;
            for (int j = 0; j < 64; j++) {
                float q = rintf(row_values[g * 64 + j] / scale);
                q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
                values[block * 16 * width + g * 1024 + (size_t)(j / 4) * 64 + in_block * 4 + j % 4] = (int8_t)q;
            }
        }
    }
    free(row_values);
    }
    t->data = values;
    t->scales = scales;
    t->type = TENSOR_PACKED;
}

// Builds a cosine and sine table for every position: pairs rotated channel
// pairs of a head_dim-wide head, as the exporter computes them.
static void rope_table(Tensor *cosines, Tensor *sines, int head_dim, int pairs, float theta) {
    float *cosine = malloc((size_t)MAX_CONTEXT * pairs * sizeof(float));
    float *sine = malloc((size_t)MAX_CONTEXT * pairs * sizeof(float));
    if (!cosine || !sine) { perror("malloc"); exit(1); }
    #pragma omp parallel for num_threads(thread_count()) schedule(static)
    for (int position = 0; position < MAX_CONTEXT; position++)
        for (int j = 0; j < pairs; j++) {
            float frequency = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
            float angle = (float)position * frequency;
            cosine[(size_t)position * pairs + j] = cosf(angle);
            sine[(size_t)position * pairs + j] = sinf(angle);
        }
    *cosines = (Tensor){.data = cosine, .shape = {MAX_CONTEXT, pairs}};
    *sines = (Tensor){.data = sine, .shape = {MAX_CONTEXT, pairs}};
}

// A string-to-id table over the vocabulary, by open addressing.
#define VOCAB_SLOTS (1 << 19)

static uint32_t text_hash(const char *text, size_t len) {
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; i++) hash = (hash ^ (uint8_t)text[i]) * 16777619u;
    return hash;
}

static int vocab_find(const int *slots, const Text *tokens, const char *text, size_t len) {
    for (uint32_t i = text_hash(text, len) & (VOCAB_SLOTS - 1);; i = (i + 1) & (VOCAB_SLOTS - 1)) {
        int id = slots[i];
        if (id < 0) return -1;
        if (tokens[id].len == len && !memcmp(tokens[id].text, text, len)) return id;
    }
}

static int compare_entries(const void *a, const void *b) { return memcmp(a, b, 8); }

static int special_longer(const void *a, const void *b) {
    const VocabEntry *x = a, *y = b;
    size_t lx = strlen(x->token), ly = strlen(y->token);
    return lx != ly ? (lx < ly) - (lx > ly) : strcmp(x->token, y->token);
}

// Builds the tokenizer's tables from the vocabulary as the exporter does:
// special tokens (control tokens, and the few llama.cpp marks user-defined
// so that they print), single pieces of up to 4 bytes that no merge makes,
// and the merges by token-id pair. Returns 0 on success.
static int build_tokenizer(Tokenizer *tokenizer, Array tokens_array, Array types_array, Array merges_array) {
    if (tokens_array.type != 8 || tokens_array.count != VOCAB_SIZE || types_array.type != 5 ||
        types_array.count != VOCAB_SIZE || merges_array.type != 8 || merges_array.count > 514906) return 1;
    Text *tokens = malloc(VOCAB_SIZE * sizeof(*tokens));
    int *slots = malloc(VOCAB_SLOTS * sizeof(*slots));
    uint8_t *made = calloc(VOCAB_SIZE, 1); // Whether a merge produces the token.
    if (!tokens || !slots || !made) { perror("malloc"); exit(1); }
    memset(slots, -1, VOCAB_SLOTS * sizeof(*slots));
    Reader r = {tokens_array.at, tokens_array.end, 0};
    for (int id = 0; id < VOCAB_SIZE; id++) {
        tokens[id] = read_text(&r);
        uint32_t i = text_hash(tokens[id].text, tokens[id].len) & (VOCAB_SLOTS - 1);
        while (slots[i] >= 0) i = (i + 1) & (VOCAB_SLOTS - 1);
        slots[i] = id;
    }

    int bad = 0;
    tokenizer->merge_count = 0;
    r = (Reader){merges_array.at, merges_array.end, 0};
    for (uint64_t rank = 0; rank < merges_array.count && !bad; rank++) {
        Text merge = read_text(&r);
        int left = -1, right = -1, result = -1;
        // Split at whichever space leaves a token on both sides.
        for (size_t at = 0; at < merge.len && result < 0; at++) {
            if (merge.text[at] != ' ') continue;
            left = vocab_find(slots, tokens, merge.text, at);
            right = vocab_find(slots, tokens, merge.text + at + 1, merge.len - at - 1);
            if (left < 0 || right < 0) continue;
            char joined[192];
            if (merge.len - 1 > sizeof(joined)) break;
            memcpy(joined, merge.text, at);
            memcpy(joined + at, merge.text + at + 1, merge.len - at - 1);
            result = vocab_find(slots, tokens, joined, merge.len - 1);
        }
        if (result < 0) { bad = 1; break; }
        LookupEntry *entry = &tokenizer->merges[tokenizer->merge_count++];
        memcpy(entry->key, &left, 4);
        memcpy(entry->key + 4, &right, 4);
        entry->result = result;
        entry->rank = (int)rank;
        made[result] = 1;
    }
    qsort(tokenizer->merges, (size_t)tokenizer->merge_count, sizeof(LookupEntry), compare_entries);

    static const char *visible[] = {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
                                    "<|tool_response>", "<tool_response|>", "<|\"|>"};
    tokenizer->special_count = tokenizer->encode_vocab_count = 0;
    for (int id = 0; id < VOCAB_SIZE && !bad; id++) {
        int32_t type;
        memcpy(&type, types_array.at + (size_t)id * 4, 4);
        Text token = tokens[id];
        int special = type == 3;
        for (int i = 0; i < 7; i++) special |= type == 4 && text_is(token, visible[i]);
        char *decoded = tokenizer->decoded_tokens[id];
        if (token.len >= 94) { bad = 1; break; }
        if (special) {
            if (tokenizer->special_count == 256) { bad = 1; break; }
            VocabEntry *entry = &tokenizer->specials[tokenizer->special_count++];
            memcpy(entry->token, token.text, token.len);
            entry->id = id;
            decoded[0] = 0;
            continue;
        }
        if (type == 6) { // A byte token, <0xNN>.
            decoded[0] = (char)strtol(token.text + 3, NULL, 16);
            decoded[1] = 0;
        } else {
            size_t n = 0;
            for (size_t i = 0; i < token.len; i++) {
                if (i + 2 < token.len && !memcmp(token.text + i, "\xE2\x96\x81", 3)) { decoded[n++] = ' '; i += 2; }
                else decoded[n++] = token.text[i];
            }
            decoded[n] = 0;
        }
        if (!made[id] && token.len <= 4) {
            if (tokenizer->encode_vocab_count == 32768) { bad = 1; break; }
            LookupEntry *entry = &tokenizer->encode_vocab[tokenizer->encode_vocab_count++];
            memset(entry->key, 0, 8);
            memcpy(entry->key, token.text, token.len);
            entry->result = id;
            entry->rank = 0;
        }
    }
    qsort(tokenizer->encode_vocab, (size_t)tokenizer->encode_vocab_count, sizeof(LookupEntry), compare_entries);
    qsort(tokenizer->specials, (size_t)tokenizer->special_count, sizeof(VocabEntry), special_longer);
    free(tokens);
    free(slots);
    free(made);
    return bad;
}

Model *load_gguf(const uint8_t *file, size_t size, const char *path) {
    Reader r = {file + 4, file + size, 0};
    uint32_t version = (uint32_t)read_uint(&r, 4);
    uint64_t tensor_count = read_uint(&r, 8), metadata_count = read_uint(&r, 8);
    if (version != 3 || tensor_count > 4096) { fprintf(stderr, "%s: unsupported GGUF file\n", path); return NULL; }

    // Keep the metadata this model needs, checking it against the constants compiled in.
    Array tokens = {0}, types = {0}, merges = {0}, pattern = {0};
    uint64_t alignment = 32;
    int layers = 0, width = 0, window = 0, context = 0, per_layer = 0, shared = 0;
    float theta = 0.0f, theta_sliding = 0.0f;
    for (uint64_t i = 0; i < metadata_count && !r.bad; i++) {
        Text key = read_text(&r);
        uint32_t type = (uint32_t)read_uint(&r, 4);
        const uint8_t *value = r.at;
        if (type == 9) {
            Array a = {(uint32_t)read_uint(&r, 4), read_uint(&r, 8), r.at, r.end};
            if (text_is(key, "tokenizer.ggml.tokens")) tokens = a;
            else if (text_is(key, "tokenizer.ggml.token_type")) types = a;
            else if (text_is(key, "tokenizer.ggml.merges")) merges = a;
            else if (text_is(key, "gemma4.attention.sliding_window_pattern")) pattern = a;
            r.at = value;
            skip_value(&r, type);
            continue;
        }
        skip_value(&r, type);
        if (r.bad) break;
        uint32_t u32 = 0;
        float f32 = 0.0f;
        if (type == 4 || type == 5) memcpy(&u32, value, 4);
        if (type == 6) memcpy(&f32, value, 4);
        if (text_is(key, "general.architecture") && !(type == 8 && text_is((Text){(const char *)value + 8, (size_t)(r.at - value - 8)}, "gemma4"))) {
            fprintf(stderr, "%s: not a Gemma 4 model\n", path);
            return NULL;
        }
        if (text_is(key, "general.alignment")) alignment = u32;
        else if (text_is(key, "gemma4.block_count")) layers = (int)u32;
        else if (text_is(key, "gemma4.embedding_length")) width = (int)u32;
        else if (text_is(key, "gemma4.attention.sliding_window")) window = (int)u32;
        else if (text_is(key, "gemma4.context_length")) context = (int)u32;
        else if (text_is(key, "gemma4.embedding_length_per_layer_input")) per_layer = (int)u32;
        else if (text_is(key, "gemma4.attention.shared_kv_layers")) shared = (int)u32;
        else if (text_is(key, "gemma4.rope.freq_base")) theta = f32;
        else if (text_is(key, "gemma4.rope.freq_base_swa")) theta_sliding = f32;
    }
    int pattern_ok = pattern.type == 7 && pattern.count == NUM_LAYERS;
    for (int i = 0; pattern_ok && i < NUM_LAYERS; i++)
        pattern_ok = pattern.at[i] == (i % FULL_INTERVAL != FULL_INTERVAL - 1); // True marks a sliding layer.
    if (r.bad || layers != NUM_LAYERS || width != HIDDEN_SIZE || window != SLIDING_WINDOW || context != MAX_CONTEXT ||
        per_layer != 256 || shared != NUM_LAYERS - CACHE_SETS * FULL_INTERVAL || !pattern_ok || !theta || !theta_sliding ||
        !alignment || alignment & (alignment - 1)) {
        fprintf(stderr, "%s: not the Gemma 4 variant this build supports\n", path);
        return NULL;
    }

    GgufTensor *list = calloc(tensor_count, sizeof(*list));
    if (!list) { perror("calloc"); return NULL; }
    for (uint64_t i = 0; i < tensor_count && !r.bad; i++) {
        list[i].name = read_text(&r);
        list[i].dim_count = (int)read_uint(&r, 4);
        if (list[i].dim_count < 1 || list[i].dim_count > 4) { r.bad = 1; break; }
        for (int d = 0; d < list[i].dim_count; d++) {
            uint64_t dim = read_uint(&r, 8);
            if (dim < 1 || dim > (1u << 30)) r.bad = 1;
            list[i].dims[d] = (int)dim;
        }
        list[i].type = (uint32_t)read_uint(&r, 4);
        list[i].data = (const uint8_t *)(uintptr_t)read_uint(&r, 8); // An offset into the data, for now.
    }
    size_t data_start = ((size_t)(r.at - file) + alignment - 1) & ~(size_t)(alignment - 1);
    for (uint64_t i = 0; i < tensor_count && !r.bad; i++) {
        GgufTensor *t = &list[i];
        uint64_t count = 1;
        for (int d = 0; d < t->dim_count; d++) count *= (uint64_t)t->dims[d];
        uint64_t bytes = t->type == GGML_F32 ? count * 4 : t->type == GGML_F16 ? count * 2 :
                         t->type == GGML_Q4_0 ? count / 32 * 18 : t->type == GGML_Q6_K ? count / 256 * 210 : 0;
        uint64_t offset = (uint64_t)(uintptr_t)t->data;
        if (offset > size || data_start > size - offset || bytes > size - data_start - offset) r.bad = 1;
        t->data = file + data_start + offset;
    }
    if (r.bad) { fprintf(stderr, "%s: truncated or malformed GGUF file\n", path); free(list); return NULL; }

    Model *model = calloc(1, sizeof(*model));
    if (!model) { perror("calloc"); free(list); return NULL; }
    memcpy(model->magic, "GGUF", 4);
    if (build_tokenizer(&model->tokenizer, tokens, types, merges)) {
        fprintf(stderr, "%s: unexpected tokenizer\n", path);
        free(list);
        free(model);
        return NULL;
    }

    TensorIndex index = {list, (int)tensor_count, path, 0};
    ModelWeights *w = &model->weights;
    bind_tensor(&index, &w->embed, "token_embd.weight", GGML_Q6_K, 0);
    bind_tensor(&index, &w->embed_per_layer, "per_layer_token_embd.weight", GGML_Q6_K, 0);
    bind_tensor(&index, &w->norm, "output_norm.weight", GGML_F32, 0);
    bind_tensor(&index, &w->per_layer_model_projection, "per_layer_model_proj.weight", GGML_F16, 0);
    bind_tensor(&index, &w->per_layer_projection_norm, "per_layer_proj_norm.weight", GGML_F32, 0);
    Tensor rope_frequencies = {0};
    bind_tensor(&index, &rope_frequencies, "rope_freqs.weight", GGML_F32, 0);
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights *l = &w->layers[layer];
        int writes_kv = layer < NUM_LAYERS - shared;
        struct { Tensor *t; const char *name; uint32_t type; int optional; } names[] = {
            {&l->input_layernorm, "attn_norm", GGML_F32, 0},
            {&l->layer_scalar, "layer_output_scale", GGML_F32, 0},
            {&l->pre_ffn_layernorm, "ffn_norm", GGML_F32, 0},
            {&l->post_attn_layernorm, "post_attention_norm", GGML_F32, 0},
            {&l->post_ffn_layernorm, "post_ffw_norm", GGML_F32, 0},
            {&l->post_per_layer_input_norm, "post_norm", GGML_F32, 0},
            {&l->per_layer_input_gate, "inp_gate", GGML_Q4_0, 0},
            {&l->per_layer_projection, "proj", GGML_Q4_0, 0},
            {&l->q_norm, "attn_q_norm", GGML_F32, 0},
            {&l->k_norm, "attn_k_norm", GGML_F32, !writes_kv},
            {&l->q_proj, "attn_q", GGML_Q4_0, 0},
            {&l->k_proj, "attn_k", GGML_Q4_0, !writes_kv},
            {&l->v_proj, "attn_v", GGML_Q4_0, !writes_kv},
            {&l->o_proj, "attn_output", GGML_Q4_0, 0},
            {&l->gate_proj, "ffn_gate", GGML_Q4_0, 0},
            {&l->up_proj, "ffn_up", GGML_Q4_0, 0},
            {&l->down_proj, "ffn_down", GGML_Q4_0, 0},
        };
        for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, names[i].name);
            bind_tensor(&index, names[i].t, name, names[i].type, names[i].optional);
        }
        if (!writes_kv && l->k_proj.data) { fprintf(stderr, "%s: layer %d has its own KV\n", path, layer); index.missing = 1; }
    }
    free(list);
    if (index.missing || w->embed.shape[0] != VOCAB_SIZE || w->embed.shape[1] != HIDDEN_SIZE) {
        fprintf(stderr, "%s: not the Gemma 4 variant this build supports\n", path);
        free(model);
        return NULL;
    }

    // Proportional RoPE: the full-attention layers rotate only the channel
    // pairs whose frequency factor is 1 (the rest are 1e30, unrotated), at
    // frequencies set by the full head width.
    const float *factors = rope_frequencies.data;
    int full_pairs = 0;
    while (full_pairs < rope_frequencies.shape[0] && factors[full_pairs] == 1.0f) full_pairs++;
    int sliding_dim = w->layers[0].q_norm.shape[0], full_dim = w->layers[FULL_INTERVAL - 1].q_norm.shape[0];
    Tensor sliding_cos, sliding_sin, full_cos, full_sin;
    rope_table(&sliding_cos, &sliding_sin, sliding_dim, sliding_dim / 2, theta_sliding);
    rope_table(&full_cos, &full_sin, full_dim, full_pairs, theta);
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int full = layer % FULL_INTERVAL == FULL_INTERVAL - 1;
        w->layers[layer].rope_cos = full ? full_cos : sliding_cos;
        w->layers[layer].rope_sin = full ? full_sin : sliding_sin;
    }

    float *gelu = malloc(4096 * sizeof(float));
    if (!gelu) { perror("malloc"); exit(1); }
    for (int i = 0; i < 4096; i++) {
        float x = -8.0f + 16.0f * (float)i / 4095.0f;
        gelu[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
    w->gelu_table = (Tensor){.data = gelu, .shape = {4096, -8, 8}};

    pack_int8(&w->embed, w->embed.data, GGML_Q6_K);
    pack_int8(&w->per_layer_model_projection, w->per_layer_model_projection.data, GGML_F16);
    return model;
}

// Memory-maps a model file and rewrites its stored tensor offsets into pointers.
Model *load_model(const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); return NULL; }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return NULL; }
    if ((size_t)st.st_size >= 24 && !memcmp(model->magic, "GGUF", 4)) return load_gguf((const uint8_t *)model, (size_t)st.st_size, path);
    if (memcmp(model->magic, "MOG", 4) != 0) { fprintf(stderr, "%s: bad model file\n", path); return NULL; }
    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }
    if (model->weights.embed.shape[0] != VOCAB_SIZE || model->weights.embed.shape[1] != HIDDEN_SIZE) {
        fprintf(stderr, "%s: model was exported for a different Gemma 4 variant than this build\n", path);
        return NULL;
    }
    return model;
}

#ifndef GEMMA4_LIB
int main(int argc, char **argv) {
#ifdef _WIN32
    argv_utf8(&argc, &argv);
    disable_power_throttling();
#endif
    const char *model_path = MODEL_FILE;
    const char *prompt = "Why is the sky blue?";
    float temperature = 1.0f;
    int max_new_tokens = 1024;
    int benchmark_mode = 0, dump_logits = 0, prefill_tokens = 0, generated_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) {
            benchmark_mode = 1;
            if (i + 1 < argc) prefill_tokens = atoi(argv[++i]);
            if (i + 1 < argc) generated_tokens = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--dump-logits")) dump_logits = 1;
        else prompt = argv[i];
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    if (dump_logits) _setmode(_fileno(stdout), _O_BINARY);
#endif
    Model *model = load_model(model_path);
    if (!model) return 1;
    InferenceState *state = calloc(1, sizeof(*state));

    rng_state = (unsigned long long)(time_seconds() * 1e9);
    if (benchmark_mode) benchmark(model, state, prefill_tokens, generated_tokens);
    else generate(model, state, prompt, max_new_tokens, temperature, dump_logits);
    free(state);
    return 0;
}
#endif // GEMMA4_LIB

//      |\__/,|   (`\_
//    *.|o o  |*   ) )
//---(((---(((------------------
//|                            |
//|          gemma4.c          |
//|____________________________|
