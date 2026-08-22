#include "ggml-common.h"
#include "ggml-quants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    float values[QK_K];
    for (int i = 0; i < QK_K; ++i) {
        values[i] = 0.6f * sinf((float) i * 0.17f) + 0.2f * cosf((float) i * 0.031f);
    }
    block_q4_K block;
    quantize_row_q4_K_ref(values, &block, QK_K);
    const uint8_t * bytes = (const uint8_t *) &block;
    for (size_t i = 0; i < sizeof(block); ++i) {
        printf("%02x", bytes[i]);
    }
    printf("\n");
    return 0;
}
