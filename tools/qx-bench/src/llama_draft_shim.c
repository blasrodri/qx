#include "llama.h"

#include <stdint.h>
#include <stdlib.h>

typedef struct qx_llama_draft {
    struct llama_model * model;
    struct llama_context * context;
    int32_t n_vocab;
    uint32_t position;
    char error[256];
} qx_llama_draft;

static void qx_set_error(qx_llama_draft * draft, const char * message) {
    if (!draft) {
        return;
    }
    size_t index = 0;
    while (message[index] && index + 1 < sizeof(draft->error)) {
        draft->error[index] = message[index];
        ++index;
    }
    draft->error[index] = '\0';
}

qx_llama_draft * qx_llama_draft_open(
        const char * path,
        uint32_t context_length,
        int32_t gpu_layers,
        int32_t threads) {
    qx_llama_draft * draft = calloc(1, sizeof(*draft));
    if (!draft) {
        return NULL;
    }

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    draft->model = llama_model_load_from_file(path, model_params);
    if (!draft->model) {
        qx_set_error(draft, "llama_model_load_from_file failed");
        free(draft);
        return NULL;
    }

    struct llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = context_length;
    context_params.n_batch = 4;
    context_params.n_ubatch = 4;
    context_params.n_outputs_max = 1;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    context_params.offload_kqv = true;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    draft->context = llama_init_from_model(draft->model, context_params);
    if (!draft->context) {
        qx_set_error(draft, "llama_init_from_model failed");
        llama_model_free(draft->model);
        free(draft);
        return NULL;
    }

    draft->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(draft->model));
    draft->position = 0;
    return draft;
}

void qx_llama_draft_close(qx_llama_draft * draft) {
    if (!draft) {
        return;
    }
    if (draft->context) {
        llama_free(draft->context);
    }
    if (draft->model) {
        llama_model_free(draft->model);
    }
    free(draft);
}

int32_t qx_llama_draft_vocab(const qx_llama_draft * draft) {
    return draft ? draft->n_vocab : 0;
}

uint32_t qx_llama_draft_position(const qx_llama_draft * draft) {
    return draft ? draft->position : 0;
}

const char * qx_llama_draft_error(const qx_llama_draft * draft) {
    return draft ? draft->error : "null draft";
}

int qx_llama_draft_step(qx_llama_draft * draft, int32_t input, int32_t * output) {
    if (!draft || !draft->context || !output) {
        return -1;
    }
    llama_token token = input;
    struct llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(draft->context, batch) != 0) {
        qx_set_error(draft, "llama_decode failed");
        return -2;
    }
    float * logits = llama_get_logits(draft->context);
    if (!logits) {
        qx_set_error(draft, "llama_get_logits returned null");
        return -3;
    }
    llama_synchronize(draft->context);
    int32_t best = 0;
    float best_value = logits[0];
    for (int32_t index = 1; index < draft->n_vocab; ++index) {
        if (logits[index] > best_value) {
            best = index;
            best_value = logits[index];
        }
    }
    *output = best;
    ++draft->position;
    return 0;
}

int qx_llama_draft_rewind(qx_llama_draft * draft, uint32_t position) {
    if (!draft || !draft->context || position > draft->position) {
        return -1;
    }
    llama_memory_t memory = llama_get_memory(draft->context);
    if (!llama_memory_seq_rm(memory, 0, (llama_pos) position, -1)) {
        qx_set_error(draft, "llama_memory_seq_rm failed");
        return -2;
    }
    draft->position = position;
    return 0;
}
