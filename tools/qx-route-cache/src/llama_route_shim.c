#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct qx_llama_route {
    struct llama_model * model;
    struct llama_context * context;
    int32_t n_vocab;
    int32_t layers;
    int32_t experts;
    int32_t top_k;
    int32_t intermediate;
    int32_t hidden;
    int32_t contribution_size_count;
    int32_t contribution_total_packets;
    int32_t contribution_finest_neurons;
    int32_t sample_layer;
    int32_t sample_expert;
    bool capture_all_routes;
    bool capture_all_activations;
    int32_t * contribution_packet_neurons;
    uint32_t position;
    int32_t * active_routes;
    float * active_activations;
    float * active_contribution_norms;
    float * active_contribution_residuals;
    uint16_t * active_contribution_order;
    float * active_contribution_reconstruction;
    uint8_t * seen_layers;
    uint8_t * seen_activations;
    uint8_t * seen_contributions;
    uint8_t * neuron_mask;
    float * sample_input;
    float * sample_activation;
    float * sample_down_output;
    float * sample_all_down_outputs;
    float * sample_down_weights;
    float * sample_student_down_weights;
    int32_t sample_joint_width;
    float * sample_joint_gate_weights;
    float * sample_joint_up_weights;
    float * sample_joint_down_weights;
    float * sample_joint_activation;
    int32_t joint_count;
    int32_t joint_active_count;
    int32_t * joint_layers;
    int32_t * joint_experts;
    int32_t * joint_widths;
    float ** joint_gate_weights;
    float ** joint_up_weights;
    float ** joint_down_weights;
    float ** joint_activations;
    float * joint_output;
    int32_t student_count;
    int32_t * student_layers;
    int32_t * student_experts;
    float ** student_down_weights;
    float * student_output;
    bool sample_ready;
    bool sample_input_ready;
    bool sample_weights_ready;
    bool sample_weights_taken;
    uint8_t * scratch_quantized;
    size_t scratch_quantized_bytes;
    float * scratch_row;
    float * scratch_output;
    float * scratch_base_vectors;
    float * scratch_packet_vectors;
    float * scratch_residual;
    float * scratch_norms;
    uint8_t * scratch_selected;
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

static bool qx_resize_bytes(uint8_t ** pointer, size_t * capacity, size_t required) {
    if (*capacity >= required) {
        return true;
    }
    void * resized = realloc(*pointer, required);
    if (!resized) {
        return false;
    }
    *pointer = resized;
    *capacity = required;
    return true;
}

static double qx_squared_norm(const float * values, int32_t count) {
    double sum = 0.0;
    for (int32_t index = 0; index < count; ++index) {
        const double value = values[index];
        sum += value * value;
    }
    return sum;
}

static bool qx_capture_down_contribution(
        qx_llama_route * route,
        struct ggml_tensor * tensor,
        int32_t layer) {
    if (!route->active_contribution_norms || !route->active_contribution_residuals ||
        !route->active_contribution_order ||
        !route->active_contribution_reconstruction || !route->seen_layers[layer] ||
        !route->seen_activations[layer] || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != route->hidden || tensor->ne[1] != route->top_k ||
        tensor->ne[2] != 1 || !tensor->src[0]) {
        qx_set_error(route, "unexpected ffn_moe_down tensor state, shape, or type");
        return false;
    }

    const struct ggml_tensor * weights = tensor->src[0];
    const struct ggml_type_traits * traits = ggml_get_type_traits(weights->type);
    if (weights->ne[0] != route->intermediate || weights->ne[1] != route->hidden ||
        weights->ne[2] <= 0 || !traits || !traits->to_float ||
        weights->nb[1] == 0 || weights->nb[2] < weights->nb[1] * (size_t) route->hidden) {
        qx_set_error(route, "unsupported ffn_moe_down weight tensor");
        return false;
    }

    const int32_t base_packets = route->intermediate / route->contribution_finest_neurons;
    const size_t vector_values = (size_t) base_packets * route->hidden;
    if (!route->scratch_row) {
        route->scratch_row = malloc((size_t) route->intermediate * sizeof(float));
        route->scratch_output = malloc((size_t) route->top_k * route->hidden * sizeof(float));
        route->scratch_base_vectors = malloc(vector_values * sizeof(float));
        route->scratch_packet_vectors = malloc(vector_values * sizeof(float));
        route->scratch_residual = malloc((size_t) route->hidden * sizeof(float));
        route->scratch_norms = malloc((size_t) base_packets * sizeof(float));
        route->scratch_selected = malloc((size_t) base_packets);
        if (!route->scratch_row || !route->scratch_output || !route->scratch_base_vectors ||
            !route->scratch_packet_vectors || !route->scratch_residual ||
            !route->scratch_norms || !route->scratch_selected) {
            qx_set_error(route, "failed to allocate contribution scratch buffers");
            return false;
        }
    }
    if (!qx_resize_bytes(
            &route->scratch_quantized,
            &route->scratch_quantized_bytes,
            weights->nb[2])) {
        qx_set_error(route, "failed to allocate quantized expert scratch buffer");
        return false;
    }

    ggml_backend_tensor_get(
        tensor,
        route->scratch_output,
        0,
        (size_t) route->top_k * route->hidden * sizeof(float));

    for (int32_t slot = 0; slot < route->top_k; ++slot) {
        const int32_t expert = route->active_routes[layer * route->top_k + slot];
        if (expert < 0 || expert >= weights->ne[2]) {
            qx_set_error(route, "selected expert is outside ffn_moe_down weights");
            return false;
        }
        ggml_backend_tensor_get(
            weights,
            route->scratch_quantized,
            (size_t) expert * weights->nb[2],
            weights->nb[2]);

        const float * activation = route->active_activations +
            ((size_t) layer * route->top_k + slot) * route->intermediate;
        memset(route->scratch_base_vectors, 0, vector_values * sizeof(float));
        for (int32_t output = 0; output < route->hidden; ++output) {
            traits->to_float(
                route->scratch_quantized + (size_t) output * weights->nb[1],
                route->scratch_row,
                route->intermediate);
            for (int32_t packet = 0; packet < base_packets; ++packet) {
                double contribution = 0.0;
                const int32_t start = packet * route->contribution_finest_neurons;
                const int32_t end = start + route->contribution_finest_neurons;
                for (int32_t neuron = start; neuron < end; ++neuron) {
                    contribution += (double) route->scratch_row[neuron] * activation[neuron];
                }
                route->scratch_base_vectors[(size_t) packet * route->hidden + output] =
                    (float) contribution;
            }
        }

        for (int32_t output = 0; output < route->hidden; ++output) {
            double reconstructed = 0.0;
            for (int32_t packet = 0; packet < base_packets; ++packet) {
                reconstructed += route->scratch_base_vectors[(size_t) packet * route->hidden + output];
            }
            route->scratch_residual[output] = (float) reconstructed;
        }
        const float * actual = route->scratch_output + (size_t) slot * route->hidden;
        double actual_norm = qx_squared_norm(actual, route->hidden);
        double reconstruction_error = 0.0;
        for (int32_t output = 0; output < route->hidden; ++output) {
            const double difference = (double) route->scratch_residual[output] - actual[output];
            reconstruction_error += difference * difference;
        }
        route->active_contribution_reconstruction[layer * route->top_k + slot] =
            (float) (reconstruction_error / fmax(actual_norm, DBL_MIN));

        int32_t packet_offset = 0;
        for (int32_t size_index = 0; size_index < route->contribution_size_count; ++size_index) {
            const int32_t packet_neurons = route->contribution_packet_neurons[size_index];
            const int32_t packets = route->intermediate / packet_neurons;
            const int32_t base_per_packet = packet_neurons / route->contribution_finest_neurons;
            for (int32_t packet = 0; packet < packets; ++packet) {
                float * destination = route->scratch_packet_vectors + (size_t) packet * route->hidden;
                memset(destination, 0, (size_t) route->hidden * sizeof(float));
                for (int32_t base = 0; base < base_per_packet; ++base) {
                    const float * source = route->scratch_base_vectors +
                        (size_t) (packet * base_per_packet + base) * route->hidden;
                    for (int32_t output = 0; output < route->hidden; ++output) {
                        destination[output] += source[output];
                    }
                }
                route->scratch_norms[packet] = (float) qx_squared_norm(destination, route->hidden);
            }

            float * norm_output = route->active_contribution_norms +
                ((size_t) layer * route->top_k + slot) * route->contribution_total_packets +
                packet_offset;
            float * residual_output = route->active_contribution_residuals +
                ((size_t) layer * route->top_k + slot) * route->contribution_total_packets +
                packet_offset;
            uint16_t * order_output = route->active_contribution_order +
                ((size_t) layer * route->top_k + slot) * route->contribution_total_packets +
                packet_offset;
            memcpy(norm_output, route->scratch_norms, (size_t) packets * sizeof(float));
            memset(route->scratch_selected, 0, (size_t) packets);
            memcpy(
                route->scratch_residual,
                route->scratch_output + (size_t) slot * route->hidden,
                (size_t) route->hidden * sizeof(float));
            const double full_norm = qx_squared_norm(route->scratch_residual, route->hidden);

            const bool use_greedy_residual = packets <= 64;
            for (int32_t step = 0; step < packets; ++step) {
                int32_t best_packet = -1;
                double best_error = DBL_MAX;
                for (int32_t packet = 0; packet < packets; ++packet) {
                    if (route->scratch_selected[packet]) {
                        continue;
                    }
                    if (use_greedy_residual) {
                        const float * candidate = route->scratch_packet_vectors +
                            (size_t) packet * route->hidden;
                        double error = 0.0;
                        for (int32_t output = 0; output < route->hidden; ++output) {
                            const double difference =
                                (double) route->scratch_residual[output] - candidate[output];
                            error += difference * difference;
                        }
                        if (error < best_error) {
                            best_error = error;
                            best_packet = packet;
                        }
                    } else if (best_packet < 0 ||
                            route->scratch_norms[packet] > route->scratch_norms[best_packet]) {
                        best_packet = packet;
                    }
                }
                if (best_packet < 0) {
                    qx_set_error(route, "contribution greedy search failed");
                    return false;
                }
                const float * selected = route->scratch_packet_vectors +
                    (size_t) best_packet * route->hidden;
                for (int32_t output = 0; output < route->hidden; ++output) {
                    route->scratch_residual[output] -= selected[output];
                }
                route->scratch_selected[best_packet] = 1;
                order_output[step] = (uint16_t) best_packet;
                if (!use_greedy_residual) {
                    best_error = qx_squared_norm(route->scratch_residual, route->hidden);
                }
                residual_output[step] =
                    (float) (best_error / fmax(full_norm, DBL_MIN));
            }
            packet_offset += packets;
        }
    }
    route->seen_contributions[layer] = 1;
    return true;
}

static bool qx_capture_expert_sample(
        qx_llama_route * route,
        struct ggml_tensor * tensor,
        int32_t layer) {
    if (layer != route->sample_layer) {
        return true;
    }
    if (!route->active_activations || !route->seen_layers[layer] ||
        !route->sample_input_ready ||
        tensor->type != GGML_TYPE_F32 || tensor->ne[0] != route->hidden ||
        tensor->ne[1] != route->top_k) {
        qx_set_error(route, "unexpected expert sample tensor state, shape, or type");
        return false;
    }
    if (route->sample_expert < 0) {
        for (int32_t slot = 0; slot < route->top_k; ++slot) {
            ggml_backend_tensor_get(
                tensor,
                route->sample_all_down_outputs + (size_t) slot * route->hidden,
                (size_t) slot * tensor->nb[1],
                (size_t) route->hidden * sizeof(float));
        }
        route->sample_ready = true;
        return true;
    }
    for (int32_t slot = 0; slot < route->top_k; ++slot) {
        if (route->active_routes[layer * route->top_k + slot] != route->sample_expert) {
            continue;
        }
        memcpy(
            route->sample_activation,
            route->active_activations +
                ((size_t) layer * route->top_k + slot) * route->intermediate,
            (size_t) route->intermediate * sizeof(float));
        ggml_backend_tensor_get(
            tensor,
            route->sample_down_output,
            (size_t) slot * tensor->nb[1],
            (size_t) route->hidden * sizeof(float));
        if (!route->sample_weights_ready) {
            const struct ggml_tensor * weights = tensor->src[0];
            const struct ggml_type_traits * traits =
                weights ? ggml_get_type_traits(weights->type) : NULL;
            if (!weights || weights->ne[0] != route->intermediate ||
                weights->ne[1] != route->hidden || route->sample_expert >= weights->ne[2] ||
                !traits || !traits->to_float ||
                !qx_resize_bytes(
                    &route->scratch_quantized,
                    &route->scratch_quantized_bytes,
                    weights->nb[2])) {
                qx_set_error(route, "unsupported expert sample down weights");
                return false;
            }
            ggml_backend_tensor_get(
                weights,
                route->scratch_quantized,
                (size_t) route->sample_expert * weights->nb[2],
                weights->nb[2]);
            for (int32_t output = 0; output < route->hidden; ++output) {
                traits->to_float(
                    route->scratch_quantized + (size_t) output * weights->nb[1],
                    route->sample_down_weights + (size_t) output * route->intermediate,
                    route->intermediate);
            }
            route->sample_weights_ready = true;
        }
        route->sample_ready = true;
        if (route->sample_joint_width > 0) {
            for (int32_t neuron = 0; neuron < route->sample_joint_width; ++neuron) {
                float gate = 0.0f;
                float up = 0.0f;
                const float * gate_row = route->sample_joint_gate_weights +
                    (size_t) neuron * route->hidden;
                const float * up_row = route->sample_joint_up_weights +
                    (size_t) neuron * route->hidden;
                for (int32_t input = 0; input < route->hidden; ++input) {
                    gate += gate_row[input] * route->sample_input[input];
                    up += up_row[input] * route->sample_input[input];
                }
                const float sigmoid = gate >= 0.0f
                    ? 1.0f / (1.0f + expf(-gate))
                    : expf(gate) / (1.0f + expf(gate));
                route->sample_joint_activation[neuron] = gate * sigmoid * up;
            }
            for (int32_t output = 0; output < route->hidden; ++output) {
                float value = 0.0f;
                const float * row = route->sample_joint_down_weights +
                    (size_t) output * route->sample_joint_width;
                for (int32_t neuron = 0; neuron < route->sample_joint_width; ++neuron) {
                    value += row[neuron] * route->sample_joint_activation[neuron];
                }
                route->sample_down_output[output] = value;
            }
            ggml_backend_tensor_set(
                tensor,
                route->sample_down_output,
                (size_t) slot * tensor->nb[1],
                (size_t) route->hidden * sizeof(float));
        }
        if (route->sample_student_down_weights) {
            for (int32_t output = 0; output < route->hidden; ++output) {
                float value = 0.0f;
                const float * row = route->sample_student_down_weights +
                    (size_t) output * route->intermediate;
                for (int32_t neuron = 0; neuron < route->intermediate; ++neuron) {
                    value += row[neuron] * route->sample_activation[neuron];
                }
                route->sample_down_output[output] = value;
            }
            ggml_backend_tensor_set(
                tensor,
                route->sample_down_output,
                (size_t) slot * tensor->nb[1],
                (size_t) route->hidden * sizeof(float));
        }
        break;
    }
    return true;
}

static const struct ggml_tensor * qx_find_expert_input(
        const struct ggml_tensor * tensor,
        int32_t hidden,
        int32_t remaining_depth) {
    if (!tensor) {
        return NULL;
    }
    if (tensor->ne[0] == hidden && tensor->ne[1] == 1 &&
        tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        return tensor;
    }
    if (remaining_depth == 0) {
        return NULL;
    }
    for (int32_t index = 0; index < GGML_MAX_SRC; ++index) {
        const struct ggml_tensor * found =
            qx_find_expert_input(tensor->src[index], hidden, remaining_depth - 1);
        if (found) {
            return found;
        }
    }
    return NULL;
}

static bool qx_capture_expert_input(
        qx_llama_route * route,
        struct ggml_tensor * tensor,
        int32_t layer) {
    (void) layer;
    const struct ggml_tensor * input = qx_find_expert_input(tensor, route->hidden, 3);
    const struct ggml_type_traits * traits = input ? ggml_get_type_traits(input->type) : NULL;
    const size_t bytes = input ? ggml_nbytes(input) : 0;
    if (!input || input->ne[0] != route->hidden || input->ne[1] != 1 ||
        input->ne[2] != 1 || input->ne[3] != 1 ||
        (input->type != GGML_TYPE_F32 && (!traits || !traits->to_float)) ||
        !qx_resize_bytes(&route->scratch_quantized, &route->scratch_quantized_bytes, bytes)) {
        qx_set_error(route, "unsupported expert input tensor");
        return false;
    }
    if (input->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(
            input,
            route->sample_input,
            0,
            (size_t) route->hidden * sizeof(float));
    } else {
        ggml_backend_tensor_get(input, route->scratch_quantized, 0, bytes);
        traits->to_float(route->scratch_quantized, route->sample_input, route->hidden);
    }
    route->sample_input_ready = true;
    return true;
}

static bool qx_has_student_layer(const qx_llama_route * route, int32_t layer) {
    for (int32_t index = 0; index < route->student_count; ++index) {
        if (route->student_layers[index] == layer) {
            return true;
        }
    }
    return false;
}

static bool qx_has_joint_student_layer(const qx_llama_route * route, int32_t layer) {
    for (int32_t index = 0; index < route->joint_active_count; ++index) {
        if (route->joint_layers[index] == layer) {
            return true;
        }
    }
    return false;
}

static bool qx_apply_joint_expert_students(
        qx_llama_route * route,
        struct ggml_tensor * tensor,
        int32_t layer) {
    if (!route->sample_input_ready || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != route->hidden || tensor->ne[1] != route->top_k) {
        qx_set_error(route, "unexpected joint expert student tensor state, shape, or type");
        return false;
    }
    for (int32_t student = 0; student < route->joint_active_count; ++student) {
        if (route->joint_layers[student] != layer) {
            continue;
        }
        int32_t slot = -1;
        for (int32_t candidate = 0; candidate < route->top_k; ++candidate) {
            if (route->active_routes[layer * route->top_k + candidate] ==
                    route->joint_experts[student]) {
                slot = candidate;
                break;
            }
        }
        if (slot < 0) {
            continue;
        }
        const int32_t width = route->joint_widths[student];
        float * activation = route->joint_activations[student];
        for (int32_t neuron = 0; neuron < width; ++neuron) {
            float gate = 0.0f;
            float up = 0.0f;
            const float * gate_row = route->joint_gate_weights[student] +
                (size_t) neuron * route->hidden;
            const float * up_row = route->joint_up_weights[student] +
                (size_t) neuron * route->hidden;
            for (int32_t input = 0; input < route->hidden; ++input) {
                gate += gate_row[input] * route->sample_input[input];
                up += up_row[input] * route->sample_input[input];
            }
            const float sigmoid = gate >= 0.0f
                ? 1.0f / (1.0f + expf(-gate))
                : expf(gate) / (1.0f + expf(gate));
            activation[neuron] = gate * sigmoid * up;
        }
        for (int32_t output = 0; output < route->hidden; ++output) {
            float value = 0.0f;
            const float * row = route->joint_down_weights[student] +
                (size_t) output * width;
            for (int32_t neuron = 0; neuron < width; ++neuron) {
                value += row[neuron] * activation[neuron];
            }
            route->joint_output[output] = value;
        }
        ggml_backend_tensor_set(
            tensor,
            route->joint_output,
            (size_t) slot * tensor->nb[1],
            (size_t) route->hidden * sizeof(float));
    }
    return true;
}

static bool qx_apply_expert_students(
        qx_llama_route * route,
        struct ggml_tensor * tensor,
        int32_t layer) {
    if (!qx_has_student_layer(route, layer)) {
        return true;
    }
    if (!route->active_activations || !route->seen_layers[layer] ||
        tensor->type != GGML_TYPE_F32 || tensor->ne[0] != route->hidden ||
        tensor->ne[1] != route->top_k) {
        qx_set_error(route, "unexpected multi-student down tensor state");
        return false;
    }
    for (int32_t slot = 0; slot < route->top_k; ++slot) {
        const int32_t active_expert = route->active_routes[layer * route->top_k + slot];
        for (int32_t student = 0; student < route->student_count; ++student) {
            if (route->student_layers[student] != layer ||
                route->student_experts[student] != active_expert) {
                continue;
            }
            const float * activation = route->active_activations +
                ((size_t) layer * route->top_k + slot) * route->intermediate;
            const float * weights = route->student_down_weights[student];
            for (int32_t output = 0; output < route->hidden; ++output) {
                float value = 0.0f;
                const float * row = weights + (size_t) output * route->intermediate;
                for (int32_t neuron = 0; neuron < route->intermediate; ++neuron) {
                    value += row[neuron] * activation[neuron];
                }
                route->student_output[output] = value;
            }
            ggml_backend_tensor_set(
                tensor,
                route->student_output,
                (size_t) slot * tensor->nb[1],
                (size_t) route->hidden * sizeof(float));
            break;
        }
    }
    return true;
}

static bool qx_route_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    qx_llama_route * route = (qx_llama_route *) user_data;
    static const char route_prefix[] = "ffn_moe_topk-";
    static const char activation_prefix[] = "ffn_moe_swiglu-";
    static const char contribution_prefix[] = "ffn_moe_down-";
    if (!route) {
        return false;
    }
    const bool is_route = strncmp(tensor->name, route_prefix, sizeof(route_prefix) - 1) == 0;
    const bool is_activation = route->intermediate > 0 &&
        strncmp(tensor->name, activation_prefix, sizeof(activation_prefix) - 1) == 0;
    const bool is_down = (route->contribution_size_count > 0 || route->sample_layer >= 0 ||
        route->student_count > 0 || route->joint_count > 0) &&
        strncmp(tensor->name, contribution_prefix, sizeof(contribution_prefix) - 1) == 0;
    if (!is_route && !is_activation && !is_down) {
        return false;
    }
    const char * suffix = tensor->name + (is_route ? sizeof(route_prefix) - 1 :
        (is_activation ? sizeof(activation_prefix) - 1 : sizeof(contribution_prefix) - 1));
    int layer = atoi(suffix);
    if (layer < 0 || layer >= route->layers) {
        return false;
    }
    if (is_route && !route->capture_all_routes && layer != route->sample_layer &&
        !qx_has_joint_student_layer(route, layer)) {
        return false;
    }
    if (is_activation && !route->capture_all_activations && layer != route->sample_layer &&
        !qx_has_joint_student_layer(route, layer)) {
        return false;
    }
    if (is_down && route->contribution_size_count == 0 && layer != route->sample_layer &&
        !qx_has_student_layer(route, layer) && !qx_has_joint_student_layer(route, layer)) {
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
    if (is_down) {
        if (route->contribution_size_count > 0) {
            qx_capture_down_contribution(route, tensor, layer);
        }
        if (route->sample_layer >= 0) {
            qx_capture_expert_sample(route, tensor, layer);
        }
        if (route->student_count > 0) {
            qx_apply_expert_students(route, tensor, layer);
        }
        if (qx_has_joint_student_layer(route, layer)) {
            qx_apply_joint_expert_students(route, tensor, layer);
        }
        return true;
    }
    if (!route->active_activations || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != route->intermediate || tensor->ne[1] != route->top_k) {
        qx_set_error(route, "unexpected ffn_moe_swiglu tensor shape or type");
        return true;
    }
    if ((layer == route->sample_layer || qx_has_joint_student_layer(route, layer)) &&
        !qx_capture_expert_input(route, tensor, layer)) {
        return true;
    }
    ggml_backend_tensor_get(
        tensor,
        route->active_activations + (size_t) layer * route->top_k * route->intermediate,
        0,
        (size_t) route->top_k * route->intermediate * sizeof(float));
    if (route->neuron_mask) {
        float * layer_activations = route->active_activations +
            (size_t) layer * route->top_k * route->intermediate;
        for (int32_t slot = 0; slot < route->top_k; ++slot) {
            const int32_t expert = route->active_routes[layer * route->top_k + slot];
            if (!route->seen_layers[layer] || expert < 0 || expert >= route->experts) {
                qx_set_error(route, "cannot apply neuron mask before valid routes are available");
                return true;
            }
            float * activation = layer_activations + (size_t) slot * route->intermediate;
            const uint8_t * mask = route->neuron_mask +
                ((size_t) layer * route->experts + expert) * route->intermediate;
            for (int32_t neuron = 0; neuron < route->intermediate; ++neuron) {
                if (!mask[neuron]) {
                    activation[neuron] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(
            tensor,
            layer_activations,
            0,
            (size_t) route->top_k * route->intermediate * sizeof(float));
    }
    route->seen_activations[layer] = 1;
    return true;
}

qx_llama_route * qx_llama_route_open(
        const char * path,
        uint32_t context_length,
        int32_t layers,
        int32_t experts,
        int32_t top_k,
        int32_t gpu_layers,
        int32_t threads,
        int32_t intermediate,
        int32_t hidden,
        int32_t sample_layer,
        int32_t sample_expert,
        int32_t capture_all_routes,
        int32_t capture_all_activations,
        const int32_t * contribution_packet_neurons,
        int32_t contribution_size_count) {
    if (!path || layers <= 0 || experts <= 0 || top_k <= 0 || intermediate < 0 || hidden < 0 ||
        contribution_size_count < 0 ||
        (sample_expert < -1 ||
            (sample_layer < 0 && sample_expert >= 0) ||
            (sample_layer >= 0 &&
                (sample_layer >= layers || sample_expert >= experts ||
                 intermediate <= 0 || hidden <= 0))) ||
        (contribution_size_count > 0 &&
            (!contribution_packet_neurons || intermediate <= 0 || hidden <= 0))) {
        return NULL;
    }
    qx_llama_route * route = calloc(1, sizeof(*route));
    if (!route) {
        return NULL;
    }
    route->layers = layers;
    route->experts = experts;
    route->top_k = top_k;
    route->intermediate = intermediate;
    route->hidden = hidden;
    route->sample_layer = sample_layer;
    route->sample_expert = sample_expert;
    route->capture_all_routes = capture_all_routes != 0;
    route->capture_all_activations = capture_all_activations != 0;
    route->contribution_size_count = contribution_size_count;
    if (contribution_size_count > 0) {
        route->contribution_packet_neurons = malloc(
            (size_t) contribution_size_count * sizeof(int32_t));
        if (!route->contribution_packet_neurons) {
            free(route);
            return NULL;
        }
        route->contribution_finest_neurons = intermediate;
        for (int32_t index = 0; index < contribution_size_count; ++index) {
            const int32_t packet_neurons = contribution_packet_neurons[index];
            if (packet_neurons <= 0 || intermediate % packet_neurons != 0) {
                free(route->contribution_packet_neurons);
                free(route);
                return NULL;
            }
            route->contribution_packet_neurons[index] = packet_neurons;
            if (packet_neurons < route->contribution_finest_neurons) {
                route->contribution_finest_neurons = packet_neurons;
            }
            route->contribution_total_packets += intermediate / packet_neurons;
        }
        for (int32_t index = 0; index < contribution_size_count; ++index) {
            if (route->contribution_packet_neurons[index] %
                    route->contribution_finest_neurons != 0) {
                free(route->contribution_packet_neurons);
                free(route);
                return NULL;
            }
        }
        if (intermediate / route->contribution_finest_neurons > UINT16_MAX) {
            free(route->contribution_packet_neurons);
            free(route);
            return NULL;
        }
    }
    route->seen_layers = calloc((size_t) layers, 1);
    route->seen_activations = intermediate > 0 ? calloc((size_t) layers, 1) : NULL;
    route->seen_contributions = contribution_size_count > 0 ? calloc((size_t) layers, 1) : NULL;
    route->sample_input = sample_layer >= 0 ?
        malloc((size_t) hidden * sizeof(float)) : NULL;
    route->sample_activation = sample_layer >= 0 ?
        malloc((size_t) intermediate * sizeof(float)) : NULL;
    route->sample_down_output = sample_layer >= 0 ?
        malloc((size_t) hidden * sizeof(float)) : NULL;
    route->sample_all_down_outputs = sample_layer >= 0 && sample_expert < 0 ?
        malloc((size_t) top_k * hidden * sizeof(float)) : NULL;
    route->sample_down_weights = sample_layer >= 0 ?
        malloc((size_t) hidden * intermediate * sizeof(float)) : NULL;
    if (!route->seen_layers || (intermediate > 0 && !route->seen_activations) ||
        (contribution_size_count > 0 && !route->seen_contributions) ||
        (sample_layer >= 0 && (!route->sample_input || !route->sample_activation ||
            !route->sample_down_output ||
            !route->sample_down_weights ||
            (sample_expert < 0 && !route->sample_all_down_outputs)))) {
        free(route->seen_layers);
        free(route->seen_activations);
        free(route->seen_contributions);
        free(route->sample_input);
        free(route->sample_activation);
        free(route->sample_down_output);
        free(route->sample_all_down_outputs);
        free(route->sample_down_weights);
        free(route->contribution_packet_neurons);
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
        free(route->seen_contributions);
        free(route->sample_input);
        free(route->sample_activation);
        free(route->sample_down_output);
        free(route->sample_all_down_outputs);
        free(route->sample_down_weights);
        free(route->contribution_packet_neurons);
        free(route);
        return NULL;
    }
    if (llama_model_n_layer(route->model) != layers) {
        llama_model_free(route->model);
        free(route->seen_layers);
        free(route->seen_activations);
        free(route->seen_contributions);
        free(route->sample_input);
        free(route->sample_activation);
        free(route->sample_down_output);
        free(route->sample_all_down_outputs);
        free(route->sample_down_weights);
        free(route->contribution_packet_neurons);
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
        free(route->seen_contributions);
        free(route->sample_input);
        free(route->sample_activation);
        free(route->sample_down_output);
        free(route->sample_all_down_outputs);
        free(route->sample_down_weights);
        free(route->contribution_packet_neurons);
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
    free(route->seen_contributions);
    free(route->neuron_mask);
    free(route->sample_input);
    free(route->sample_activation);
    free(route->sample_down_output);
    free(route->sample_all_down_outputs);
    free(route->sample_down_weights);
    free(route->sample_student_down_weights);
    free(route->sample_joint_gate_weights);
    free(route->sample_joint_up_weights);
    free(route->sample_joint_down_weights);
    free(route->sample_joint_activation);
    for (int32_t index = 0; index < route->joint_count; ++index) {
        free(route->joint_gate_weights[index]);
        free(route->joint_up_weights[index]);
        free(route->joint_down_weights[index]);
        free(route->joint_activations[index]);
    }
    free(route->joint_layers);
    free(route->joint_experts);
    free(route->joint_widths);
    free(route->joint_gate_weights);
    free(route->joint_up_weights);
    free(route->joint_down_weights);
    free(route->joint_activations);
    free(route->joint_output);
    for (int32_t index = 0; index < route->student_count; ++index) {
        free(route->student_down_weights[index]);
    }
    free(route->student_layers);
    free(route->student_experts);
    free(route->student_down_weights);
    free(route->student_output);
    free(route->contribution_packet_neurons);
    free(route->scratch_quantized);
    free(route->scratch_row);
    free(route->scratch_output);
    free(route->scratch_base_vectors);
    free(route->scratch_packet_vectors);
    free(route->scratch_residual);
    free(route->scratch_norms);
    free(route->scratch_selected);
    free(route);
}

int qx_llama_route_take_expert_sample(
        qx_llama_route * route,
        float * input,
        int32_t input_count,
        float * activation,
        int32_t activation_count,
        float * down_output,
        int32_t down_count) {
    if (!route || !input || !activation || !down_output || input_count != route->hidden ||
        activation_count != route->intermediate ||
        down_count != route->hidden || route->sample_layer < 0) {
        return -1;
    }
    if (!route->sample_ready) {
        return 0;
    }
    memcpy(input, route->sample_input, (size_t) input_count * sizeof(float));
    memcpy(activation, route->sample_activation, (size_t) activation_count * sizeof(float));
    memcpy(down_output, route->sample_down_output, (size_t) down_count * sizeof(float));
    route->sample_ready = false;
    route->sample_input_ready = false;
    return 1;
}

int qx_llama_route_take_expert_samples(
        qx_llama_route * route,
        float * input,
        int32_t input_count,
        float * down_outputs,
        int32_t down_output_count) {
    const size_t expected_outputs = route ? (size_t) route->top_k * route->hidden : 0;
    if (!route || !input || !down_outputs || route->sample_expert >= 0 ||
        input_count != route->hidden || down_output_count < 0 ||
        (size_t) down_output_count != expected_outputs || route->sample_layer < 0) {
        return -1;
    }
    if (!route->sample_ready) {
        return 0;
    }
    memcpy(input, route->sample_input, (size_t) input_count * sizeof(float));
    memcpy(down_outputs, route->sample_all_down_outputs, expected_outputs * sizeof(float));
    route->sample_ready = false;
    route->sample_input_ready = false;
    return route->top_k;
}

int qx_llama_route_take_expert_weights(
        qx_llama_route * route,
        float * weights,
        int32_t weight_count) {
    const size_t expected = route && route->sample_layer >= 0 ?
        (size_t) route->hidden * route->intermediate : 0;
    if (!route || !weights || weight_count < 0 || (size_t) weight_count != expected) {
        return -1;
    }
    if (!route->sample_weights_ready || route->sample_weights_taken) {
        return 0;
    }
    memcpy(weights, route->sample_down_weights, expected * sizeof(float));
    route->sample_weights_taken = true;
    return 1;
}

int qx_llama_route_set_neuron_mask(
        qx_llama_route * route,
        const uint8_t * mask,
        int32_t mask_count) {
    const size_t expected = route && route->intermediate > 0 ?
        (size_t) route->layers * route->experts * route->intermediate : 0;
    if (!route || !mask || expected == 0 || mask_count < 0 || (size_t) mask_count != expected) {
        return -1;
    }
    uint8_t * copied = malloc(expected);
    if (!copied) {
        return -2;
    }
    for (size_t index = 0; index < expected; ++index) {
        if (mask[index] > 1) {
            free(copied);
            return -3;
        }
        copied[index] = mask[index];
    }
    free(route->neuron_mask);
    route->neuron_mask = copied;
    return 0;
}

int qx_llama_route_set_expert_student(
        qx_llama_route * route,
        const float * weights,
        int32_t weight_count) {
    const size_t expected = route && route->sample_layer >= 0 ?
        (size_t) route->hidden * route->intermediate : 0;
    if (!route || !weights || weight_count < 0 || (size_t) weight_count != expected) {
        return -1;
    }
    float * copied = malloc(expected * sizeof(float));
    if (!copied) {
        return -2;
    }
    memcpy(copied, weights, expected * sizeof(float));
    free(route->sample_student_down_weights);
    route->sample_student_down_weights = copied;
    return 0;
}

int qx_llama_route_set_joint_expert_student(
        qx_llama_route * route,
        int32_t width,
        const float * gate_weights,
        int32_t gate_count,
        const float * up_weights,
        int32_t up_count,
        const float * down_weights,
        int32_t down_count) {
    const size_t input_values = route && width > 0 ? (size_t) width * route->hidden : 0;
    const size_t down_values = input_values;
    if (!route || route->sample_layer < 0 || width <= 0 || width > route->intermediate ||
        !gate_weights || !up_weights || !down_weights || gate_count < 0 || up_count < 0 ||
        down_count < 0 || (size_t) gate_count != input_values ||
        (size_t) up_count != input_values || (size_t) down_count != down_values ||
        route->sample_student_down_weights) {
        return -1;
    }
    float * gate = malloc(input_values * sizeof(float));
    float * up = malloc(input_values * sizeof(float));
    float * down = malloc(down_values * sizeof(float));
    float * activation = malloc((size_t) width * sizeof(float));
    if (!gate || !up || !down || !activation) {
        free(gate);
        free(up);
        free(down);
        free(activation);
        return -2;
    }
    memcpy(gate, gate_weights, input_values * sizeof(float));
    memcpy(up, up_weights, input_values * sizeof(float));
    memcpy(down, down_weights, down_values * sizeof(float));
    free(route->sample_joint_gate_weights);
    free(route->sample_joint_up_weights);
    free(route->sample_joint_down_weights);
    free(route->sample_joint_activation);
    route->sample_joint_width = width;
    route->sample_joint_gate_weights = gate;
    route->sample_joint_up_weights = up;
    route->sample_joint_down_weights = down;
    route->sample_joint_activation = activation;
    return 0;
}

int qx_llama_route_add_joint_expert_student(
        qx_llama_route * route,
        int32_t layer,
        int32_t expert,
        int32_t width,
        const float * gate_weights,
        int32_t gate_count,
        const float * up_weights,
        int32_t up_count,
        const float * down_weights,
        int32_t down_count) {
    const size_t matrix_values = route && width > 0 ? (size_t) width * route->hidden : 0;
    if (!route || layer < 0 || layer >= route->layers || expert < 0 ||
        expert >= route->experts || width <= 0 || width > route->intermediate ||
        !gate_weights || !up_weights || !down_weights || gate_count < 0 || up_count < 0 ||
        down_count < 0 || (size_t) gate_count != matrix_values ||
        (size_t) up_count != matrix_values || (size_t) down_count != matrix_values) {
        return -1;
    }
    for (int32_t index = 0; index < route->joint_count; ++index) {
        if (route->joint_layers[index] == layer && route->joint_experts[index] == expert) {
            return -2;
        }
    }
    const size_t count = (size_t) route->joint_count + 1;
    int32_t * layers = malloc(count * sizeof(int32_t));
    int32_t * experts = malloc(count * sizeof(int32_t));
    int32_t * widths = malloc(count * sizeof(int32_t));
    float ** gates = malloc(count * sizeof(float *));
    float ** ups = malloc(count * sizeof(float *));
    float ** downs = malloc(count * sizeof(float *));
    float ** activations = malloc(count * sizeof(float *));
    float * gate = malloc(matrix_values * sizeof(float));
    float * up = malloc(matrix_values * sizeof(float));
    float * down = malloc(matrix_values * sizeof(float));
    float * activation = malloc((size_t) width * sizeof(float));
    float * input = route->sample_input ? route->sample_input :
        malloc((size_t) route->hidden * sizeof(float));
    float * output = route->joint_output ? route->joint_output :
        malloc((size_t) route->hidden * sizeof(float));
    if (!layers || !experts || !widths || !gates || !ups || !downs || !activations ||
        !gate || !up || !down || !activation || !input || !output) {
        free(layers);
        free(experts);
        free(widths);
        free(gates);
        free(ups);
        free(downs);
        free(activations);
        free(gate);
        free(up);
        free(down);
        free(activation);
        if (!route->sample_input) {
            free(input);
        }
        if (!route->joint_output) {
            free(output);
        }
        return -3;
    }
    if (route->joint_count > 0) {
        const size_t previous = (size_t) route->joint_count;
        memcpy(layers, route->joint_layers, previous * sizeof(int32_t));
        memcpy(experts, route->joint_experts, previous * sizeof(int32_t));
        memcpy(widths, route->joint_widths, previous * sizeof(int32_t));
        memcpy(gates, route->joint_gate_weights, previous * sizeof(float *));
        memcpy(ups, route->joint_up_weights, previous * sizeof(float *));
        memcpy(downs, route->joint_down_weights, previous * sizeof(float *));
        memcpy(activations, route->joint_activations, previous * sizeof(float *));
    }
    memcpy(gate, gate_weights, matrix_values * sizeof(float));
    memcpy(up, up_weights, matrix_values * sizeof(float));
    memcpy(down, down_weights, matrix_values * sizeof(float));
    layers[route->joint_count] = layer;
    experts[route->joint_count] = expert;
    widths[route->joint_count] = width;
    gates[route->joint_count] = gate;
    ups[route->joint_count] = up;
    downs[route->joint_count] = down;
    activations[route->joint_count] = activation;
    free(route->joint_layers);
    free(route->joint_experts);
    free(route->joint_widths);
    free(route->joint_gate_weights);
    free(route->joint_up_weights);
    free(route->joint_down_weights);
    free(route->joint_activations);
    route->joint_layers = layers;
    route->joint_experts = experts;
    route->joint_widths = widths;
    route->joint_gate_weights = gates;
    route->joint_up_weights = ups;
    route->joint_down_weights = downs;
    route->joint_activations = activations;
    route->sample_input = input;
    route->joint_output = output;
    route->joint_count += 1;
    route->joint_active_count = route->joint_count;
    return 0;
}

int qx_llama_route_set_joint_active_prefix(qx_llama_route * route, int32_t count) {
    if (!route || count < 0 || count > route->joint_count) {
        return -1;
    }
    route->joint_active_count = count;
    return 0;
}

int qx_llama_route_add_expert_student(
        qx_llama_route * route,
        int32_t layer,
        int32_t expert,
        const float * weights,
        int32_t weight_count) {
    const size_t expected = route ? (size_t) route->hidden * route->intermediate : 0;
    if (!route || layer < 0 || layer >= route->layers || expert < 0 ||
        expert >= route->experts || !weights || weight_count < 0 ||
        (size_t) weight_count != expected) {
        return -1;
    }
    for (int32_t index = 0; index < route->student_count; ++index) {
        if (route->student_layers[index] == layer && route->student_experts[index] == expert) {
            return -2;
        }
    }
    const size_t count = (size_t) route->student_count + 1;
    int32_t * layers = malloc(count * sizeof(int32_t));
    int32_t * experts = malloc(count * sizeof(int32_t));
    float ** matrices = malloc(count * sizeof(float *));
    float * copied = malloc(expected * sizeof(float));
    float * output = route->student_output ? route->student_output :
        malloc((size_t) route->hidden * sizeof(float));
    if (!layers || !experts || !matrices || !copied || !output) {
        free(layers);
        free(experts);
        free(matrices);
        free(copied);
        if (!route->student_output) {
            free(output);
        }
        return -3;
    }
    if (route->student_count > 0) {
        memcpy(layers, route->student_layers, (size_t) route->student_count * sizeof(int32_t));
        memcpy(experts, route->student_experts, (size_t) route->student_count * sizeof(int32_t));
        memcpy(matrices, route->student_down_weights,
            (size_t) route->student_count * sizeof(float *));
    }
    memcpy(copied, weights, expected * sizeof(float));
    layers[route->student_count] = layer;
    experts[route->student_count] = expert;
    matrices[route->student_count] = copied;
    free(route->student_layers);
    free(route->student_experts);
    free(route->student_down_weights);
    route->student_layers = layers;
    route->student_experts = experts;
    route->student_down_weights = matrices;
    route->student_output = output;
    route->student_count += 1;
    return 0;
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

int qx_llama_route_reset(qx_llama_route * route) {
    if (!route || !route->context) {
        return -1;
    }
    llama_memory_t memory = llama_get_memory(route->context);
    if (!memory) {
        return -2;
    }
    llama_synchronize(route->context);
    llama_memory_clear(memory, false);
    route->position = 0;
    route->sample_ready = false;
    route->sample_input_ready = false;
    return 0;
}

int qx_llama_route_step(
        qx_llama_route * route,
        int32_t input,
        int32_t target,
        float * target_nll,
        int32_t * output,
        int32_t * routes,
        int32_t route_count,
        float * activations,
        int32_t activation_count,
        float * contribution_norms,
        float * contribution_residuals,
        uint16_t * contribution_order,
        int32_t contribution_metric_count,
        float * contribution_reconstruction,
        int32_t contribution_reconstruction_count) {
    const int32_t expected_contribution_metrics =
        route ? route->layers * route->top_k * route->contribution_total_packets : 0;
    const int32_t expected_reconstruction =
        route ? route->layers * route->top_k : 0;
    if (!route || !route->context || !output || !routes ||
        ((target >= 0) != (target_nll != NULL)) || target >= route->n_vocab ||
        route_count != route->layers * route->top_k ||
        (route->intermediate > 0 && (!activations ||
            activation_count != route->layers * route->top_k * route->intermediate)) ||
        (route->intermediate == 0 && (activations || activation_count != 0)) ||
        (route->contribution_size_count > 0 &&
            (!contribution_norms || !contribution_residuals || !contribution_order ||
             !contribution_reconstruction ||
             contribution_metric_count != expected_contribution_metrics ||
             contribution_reconstruction_count != expected_reconstruction)) ||
        (route->contribution_size_count == 0 &&
            (contribution_norms || contribution_residuals || contribution_order ||
             contribution_reconstruction ||
             contribution_metric_count != 0 || contribution_reconstruction_count != 0))) {
        return -1;
    }
    memset(route->seen_layers, 0, (size_t) route->layers);
    if (route->intermediate > 0) {
        memset(route->seen_activations, 0, (size_t) route->layers);
        memset(activations, 0, (size_t) activation_count * sizeof(float));
    }
    if (route->contribution_size_count > 0) {
        memset(route->seen_contributions, 0, (size_t) route->layers);
        memset(contribution_norms, 0, (size_t) contribution_metric_count * sizeof(float));
        memset(contribution_residuals, 0, (size_t) contribution_metric_count * sizeof(float));
        memset(contribution_order, 0xff, (size_t) contribution_metric_count * sizeof(uint16_t));
        memset(
            contribution_reconstruction,
            0,
            (size_t) contribution_reconstruction_count * sizeof(float));
    }
    memset(routes, 0xff, (size_t) route_count * sizeof(int32_t));
    route->sample_ready = false;
    route->sample_input_ready = false;
    route->active_routes = routes;
    route->active_activations = activations;
    route->active_contribution_norms = contribution_norms;
    route->active_contribution_residuals = contribution_residuals;
    route->active_contribution_order = contribution_order;
    route->active_contribution_reconstruction = contribution_reconstruction;
    route->error[0] = '\0';

    llama_token token = input;
    struct llama_batch batch = llama_batch_get_one(&token, 1);
    int decode_status = llama_decode(route->context, batch);
    llama_synchronize(route->context);
    route->active_routes = NULL;
    route->active_activations = NULL;
    route->active_contribution_norms = NULL;
    route->active_contribution_residuals = NULL;
    route->active_contribution_order = NULL;
    route->active_contribution_reconstruction = NULL;
    if (decode_status != 0) {
        qx_set_error(route, "llama_decode failed");
        return -2;
    }
    if (route->error[0] != '\0') {
        return -7;
    }
    for (int32_t layer = 0; layer < route->layers; ++layer) {
        if (!route->seen_layers[layer] &&
            (route->capture_all_routes || layer == route->sample_layer)) {
            qx_set_error(route, "route callback missed one or more layers");
            return -3;
        }
        if (route->intermediate > 0 && !route->seen_activations[layer] &&
            (route->capture_all_activations || layer == route->sample_layer)) {
            qx_set_error(route, "activation callback missed one or more layers");
            return -5;
        }
        if (route->contribution_size_count > 0 && !route->seen_contributions[layer]) {
            qx_set_error(route, "contribution callback missed one or more layers");
            return -6;
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
    if (target >= 0) {
        double maximum = logits[0];
        for (int32_t index = 1; index < route->n_vocab; ++index) {
            if (logits[index] > maximum) {
                maximum = logits[index];
            }
        }
        double exponential_sum = 0.0;
        for (int32_t index = 0; index < route->n_vocab; ++index) {
            exponential_sum += exp((double) logits[index] - maximum);
        }
        *target_nll = (float) (log(exponential_sum) + maximum - logits[target]);
    }
    ++route->position;
    return 0;
}
