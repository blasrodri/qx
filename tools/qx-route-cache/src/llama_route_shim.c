#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct qx_llama_route {
    struct llama_model * model;
    struct llama_context * context;
    int32_t n_vocab;
    int32_t layers;
    int32_t top_k;
    int32_t intermediate;
    uint32_t position;
    int32_t * active_routes;
    float * active_activations;
    uint8_t * seen_layers;
    uint8_t * seen_activations;
    char error[256];
} qx_llama_route;

static void qx_set_error(qx_llama_route * route, const char * message) {
    if (!route) {
        return;
    }
    snprintf(route->error, sizeof(route->error), "%s", message);
}

static void qx_route_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static bool qx_route_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    qx_llama_route * route = (qx_llama_route *) user_data;
    static const char route_prefix[] = "ffn_moe_topk-";
    static const char activation_prefix[] = "ffn_moe_swiglu-";
    if (!route) {
        return false;
    }
    const bool is_route = strncmp(tensor->name, route_prefix, sizeof(route_prefix) - 1) == 0;
    const bool is_activation = route->intermediate > 0 &&
        strncmp(tensor->name, activation_prefix, sizeof(activation_prefix) - 1) == 0;
    if (!is_route && !is_activation) {
        return false;
    }
    const char * suffix = tensor->name +
        (is_route ? sizeof(route_prefix) - 1 : sizeof(activation_prefix) - 1);
    int layer = atoi(suffix);
    if (layer < 0 || layer >= route->layers) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (is_route) {
        if (!route->active_routes || tensor->type != GGML_TYPE_I32 || tensor->ne[0] != route->top_k) {
            qx_set_error(route, "unexpected ffn_moe_topk tensor shape or type");
            return true;
        }
        ggml_backend_tensor_get(
            tensor,
            route->active_routes + layer * route->top_k,
            0,
            (size_t) route->top_k * sizeof(int32_t));
        route->seen_layers[layer] = 1;
        return true;
    }
    if (!route->active_activations || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != route->intermediate || tensor->ne[1] != route->top_k) {
        qx_set_error(route, "unexpected ffn_moe_swiglu tensor shape or type");
        return true;
    }
    ggml_backend_tensor_get(
        tensor,
        route->active_activations + (size_t) layer * route->top_k * route->intermediate,
        0,
        (size_t) route->top_k * route->intermediate * sizeof(float));
    route->seen_activations[layer] = 1;
    return true;
}

qx_llama_route * qx_llama_route_open(
        const char * path,
        uint32_t context_length,
        int32_t layers,
        int32_t top_k,
        int32_t gpu_layers,
        int32_t threads,
        int32_t intermediate) {
    if (!path || layers <= 0 || top_k <= 0 || intermediate < 0) {
        return NULL;
    }
    qx_llama_route * route = calloc(1, sizeof(*route));
    if (!route) {
        return NULL;
    }
    route->layers = layers;
    route->top_k = top_k;
    route->intermediate = intermediate;
    route->seen_layers = calloc((size_t) layers, 1);
    route->seen_activations = intermediate > 0 ? calloc((size_t) layers, 1) : NULL;
    if (!route->seen_layers || (intermediate > 0 && !route->seen_activations)) {
        free(route->seen_layers);
        free(route->seen_activations);
        free(route);
        return NULL;
    }

    llama_log_set(qx_route_log, NULL);
    llama_backend_init();
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    route->model = llama_model_load_from_file(path, model_params);
    if (!route->model) {
        free(route->seen_layers);
        free(route->seen_activations);
        free(route);
        return NULL;
    }
    if (llama_model_n_layer(route->model) != layers) {
        llama_model_free(route->model);
        free(route->seen_layers);
        free(route->seen_activations);
        free(route);
        return NULL;
    }

    struct llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = context_length;
    context_params.n_batch = 1;
    context_params.n_ubatch = 1;
    context_params.n_outputs_max = 1;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    context_params.offload_kqv = true;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.cb_eval = qx_route_callback;
    context_params.cb_eval_user_data = route;
    route->context = llama_init_from_model(route->model, context_params);
    if (!route->context) {
        llama_model_free(route->model);
        free(route->seen_layers);
        free(route->seen_activations);
        free(route);
        return NULL;
    }

    route->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(route->model));
    return route;
}

void qx_llama_route_close(qx_llama_route * route) {
    if (!route) {
        return;
    }
    if (route->context) {
        llama_free(route->context);
    }
    if (route->model) {
        llama_model_free(route->model);
    }
    free(route->seen_layers);
    free(route->seen_activations);
    free(route);
}

const char * qx_llama_route_error(const qx_llama_route * route) {
    return route ? route->error : "failed to open llama route capture";
}

int32_t qx_llama_route_tokenize(
        const qx_llama_route * route,
        const char * text,
        int32_t text_length,
        int32_t * tokens,
        int32_t capacity) {
    if (!route || !route->model || !text) {
        return 0;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(route->model);
    return llama_tokenize(
        vocab,
        text,
        text_length,
        tokens,
        capacity,
        llama_vocab_get_add_bos(vocab),
        true);
}

int32_t qx_llama_route_eos(const qx_llama_route * route) {
    if (!route || !route->model) {
        return -1;
    }
    return llama_vocab_eos(llama_model_get_vocab(route->model));
}

int qx_llama_route_step(
        qx_llama_route * route,
        int32_t input,
        int32_t * output,
        int32_t * routes,
        int32_t route_count,
        float * activations,
        int32_t activation_count) {
    if (!route || !route->context || !output || !routes ||
        route_count != route->layers * route->top_k ||
        (route->intermediate > 0 && (!activations ||
            activation_count != route->layers * route->top_k * route->intermediate)) ||
        (route->intermediate == 0 && (activations || activation_count != 0))) {
        return -1;
    }
    memset(route->seen_layers, 0, (size_t) route->layers);
    if (route->intermediate > 0) {
        memset(route->seen_activations, 0, (size_t) route->layers);
        memset(activations, 0, (size_t) activation_count * sizeof(float));
    }
    memset(routes, 0xff, (size_t) route_count * sizeof(int32_t));
    route->active_routes = routes;
    route->active_activations = activations;
    route->error[0] = '\0';

    llama_token token = input;
    struct llama_batch batch = llama_batch_get_one(&token, 1);
    int decode_status = llama_decode(route->context, batch);
    llama_synchronize(route->context);
    route->active_routes = NULL;
    route->active_activations = NULL;
    if (decode_status != 0) {
        qx_set_error(route, "llama_decode failed");
        return -2;
    }
    for (int32_t layer = 0; layer < route->layers; ++layer) {
        if (!route->seen_layers[layer]) {
            qx_set_error(route, "route callback missed one or more layers");
            return -3;
        }
        if (route->intermediate > 0 && !route->seen_activations[layer]) {
            qx_set_error(route, "activation callback missed one or more layers");
            return -5;
        }
    }
    float * logits = llama_get_logits(route->context);
    if (!logits) {
        qx_set_error(route, "llama_get_logits returned null");
        return -4;
    }
    int32_t best = 0;
    float best_value = logits[0];
    for (int32_t index = 1; index < route->n_vocab; ++index) {
        if (logits[index] > best_value) {
            best = index;
            best_value = logits[index];
        }
    }
    *output = best;
    ++route->position;
    return 0;
}
