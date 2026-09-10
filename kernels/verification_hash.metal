#include <metal_stdlib>
using namespace metal;

struct VerificationHashParams {
    uint token;
    uint layers;
    uint residual_stride;
    uint checkpoint_interval;
    uint key_offsets[24];
    uint value_offsets[24];
    uint key_bytes[24];
    uint value_bytes[24];
};

struct Sha256Context {
    uint state[8];
    uchar block[64];
    uint used;
    ulong total;
};

constant uint SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};
constant uint B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};
constant uchar B3_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

constant uchar STATE_DOMAIN[11] = {'Q','X',':','S','T','A','T','E',':','V','1'};
constant uchar STATE_CHUNK_DOMAIN[17] = {
    'Q','X',':','S','T','A','T','E',':','C','H','U','N','K',':','V','1'
};
constant uchar KV_DOMAIN[8] = {'Q','X',':','K','V',':','V','1'};
constant uchar KV_KEY_CHUNK_DOMAIN[18] = {
    'Q','X',':','K','V',':','K','E','Y',':','C','H','U','N','K',':','V','1'
};
constant uchar KV_KEY_TREE_DOMAIN[17] = {
    'Q','X',':','K','V',':','K','E','Y',':','T','R','E','E',':','V','1'
};
constant uchar KV_VALUE_CHUNK_DOMAIN[20] = {
    'Q','X',':','K','V',':','V','A','L','U','E',':','C','H','U','N','K',':','V','1'
};
constant uchar KV_VALUE_TREE_DOMAIN[19] = {
    'Q','X',':','K','V',':','V','A','L','U','E',':','T','R','E','E',':','V','1'
};

inline uint rotr(uint value, uint shift) {
    return (value >> shift) | (value << (32u - shift));
}

struct Blake3Context {
    uint cv[8];
    uchar block[64];
    uint used;
    uint blocks;
};

inline void b3_g(thread uint* state, uint a, uint b, uint c, uint d, uint x, uint y) {
    state[a] = state[a] + state[b] + x;
    state[d] = rotr(state[d] ^ state[a], 16);
    state[c] += state[d];
    state[b] = rotr(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = rotr(state[d] ^ state[a], 8);
    state[c] += state[d];
    state[b] = rotr(state[b] ^ state[c], 7);
}

inline void b3_compress(
    thread const uint* cv,
    thread const uchar* block,
    uint block_len,
    uint flags,
    thread uint* output) {
    uint message[16];
    for (uint word = 0; word < 16; ++word) {
        uint offset = word * 4;
        message[word] = uint(block[offset]) | (uint(block[offset + 1]) << 8)
            | (uint(block[offset + 2]) << 16) | (uint(block[offset + 3]) << 24);
    }
    uint state[16];
    for (uint index = 0; index < 8; ++index) state[index] = cv[index];
    for (uint index = 0; index < 4; ++index) state[8 + index] = B3_IV[index];
    state[12] = 0;
    state[13] = 0;
    state[14] = block_len;
    state[15] = flags;
    for (uint round = 0; round < 7; ++round) {
        constant const uchar* s = B3_SCHEDULE[round];
        b3_g(state, 0, 4, 8, 12, message[s[0]], message[s[1]]);
        b3_g(state, 1, 5, 9, 13, message[s[2]], message[s[3]]);
        b3_g(state, 2, 6, 10, 14, message[s[4]], message[s[5]]);
        b3_g(state, 3, 7, 11, 15, message[s[6]], message[s[7]]);
        b3_g(state, 0, 5, 10, 15, message[s[8]], message[s[9]]);
        b3_g(state, 1, 6, 11, 12, message[s[10]], message[s[11]]);
        b3_g(state, 2, 7, 8, 13, message[s[12]], message[s[13]]);
        b3_g(state, 3, 4, 9, 14, message[s[14]], message[s[15]]);
    }
    for (uint index = 0; index < 8; ++index) output[index] = state[index] ^ state[index + 8];
}

inline void b3_init(thread Blake3Context& context) {
    for (uint index = 0; index < 8; ++index) context.cv[index] = B3_IV[index];
    for (uint index = 0; index < 64; ++index) context.block[index] = 0;
    context.used = 0;
    context.blocks = 0;
}

inline void b3_flush(thread Blake3Context& context) {
    uint next[8];
    b3_compress(context.cv, context.block, 64, context.blocks == 0 ? 1u : 0u, next);
    for (uint index = 0; index < 8; ++index) context.cv[index] = next[index];
    for (uint index = 0; index < 64; ++index) context.block[index] = 0;
    context.used = 0;
    context.blocks += 1;
}

inline void b3_byte(thread Blake3Context& context, uchar value) {
    if (context.used == 64) b3_flush(context);
    context.block[context.used++] = value;
}

inline void b3_u64_le(thread Blake3Context& context, ulong value) {
    for (uint shift = 0; shift < 64; shift += 8) b3_byte(context, uchar(value >> shift));
}
inline void b3_u32_le(thread Blake3Context& context, uint value) {
    for (uint shift = 0; shift < 32; shift += 8) b3_byte(context, uchar(value >> shift));
}
inline void b3_u16_le(thread Blake3Context& context, ushort value) {
    b3_byte(context, uchar(value));
    b3_byte(context, uchar(value >> 8));
}
inline void b3_constant_bytes(thread Blake3Context& context, constant const uchar* bytes, uint len) {
    for (uint index = 0; index < len; ++index) b3_byte(context, bytes[index]);
}
inline void b3_device_bytes(thread Blake3Context& context, device const uchar* bytes, uint len) {
    for (uint index = 0; index < len; ++index) b3_byte(context, bytes[index]);
}
inline void b3_device_chunk_256(thread Blake3Context& context, device const uchar* bytes) {
    uint consumed = 0;
    uint first = 64 - context.used;
    for (; consumed < first; ++consumed) context.block[context.used++] = bytes[consumed];
    while (consumed + 64 < 256) {
        b3_flush(context);
        for (uint index = 0; index < 64; ++index) context.block[index] = bytes[consumed + index];
        context.used = 64;
        consumed += 64;
    }
    if (context.used == 64) b3_flush(context);
    for (; consumed < 256; ++consumed) context.block[context.used++] = bytes[consumed];
}
inline void b3_digest_words(thread Blake3Context& context, threadgroup const uint* words) {
    for (uint word = 0; word < 8; ++word) {
        uint value = words[word];
        b3_byte(context, uchar(value));
        b3_byte(context, uchar(value >> 8));
        b3_byte(context, uchar(value >> 16));
        b3_byte(context, uchar(value >> 24));
    }
}
inline void b3_finish(thread Blake3Context& context, thread uint* output) {
    uint flags = (context.blocks == 0 ? 1u : 0u) | 2u | 8u;
    b3_compress(context.cv, context.block, context.used, flags, output);
}
inline void b3_store_threadgroup_device(threadgroup const uint* words, device uchar* output) {
    for (uint word = 0; word < 8; ++word) {
        uint value = words[word];
        output[word * 4] = uchar(value);
        output[word * 4 + 1] = uchar(value >> 8);
        output[word * 4 + 2] = uchar(value >> 16);
        output[word * 4 + 3] = uchar(value >> 24);
    }
}
inline void b3_store_device(thread const uint* words, device uchar* output) {
    for (uint word = 0; word < 8; ++word) {
        uint value = words[word];
        output[word * 4] = uchar(value);
        output[word * 4 + 1] = uchar(value >> 8);
        output[word * 4 + 2] = uchar(value >> 16);
        output[word * 4 + 3] = uchar(value >> 24);
    }
}

inline void sha256_compress(thread Sha256Context& context) {
    uint words[64];
    for (uint index = 0; index < 16; ++index) {
        uint offset = index * 4;
        words[index] = (uint(context.block[offset]) << 24)
            | (uint(context.block[offset + 1]) << 16)
            | (uint(context.block[offset + 2]) << 8)
            | uint(context.block[offset + 3]);
    }
    for (uint index = 16; index < 64; ++index) {
        uint left = words[index - 15];
        uint right = words[index - 2];
        uint s0 = rotr(left, 7) ^ rotr(left, 18) ^ (left >> 3);
        uint s1 = rotr(right, 17) ^ rotr(right, 19) ^ (right >> 10);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    uint a = context.state[0];
    uint b = context.state[1];
    uint c = context.state[2];
    uint d = context.state[3];
    uint e = context.state[4];
    uint f = context.state[5];
    uint g = context.state[6];
    uint h = context.state[7];
    for (uint index = 0; index < 64; ++index) {
        uint sum1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint choose = (e & f) ^ ((~e) & g);
        uint temp1 = h + sum1 + choose + SHA256_K[index] + words[index];
        uint sum0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint majority = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    context.state[0] += a;
    context.state[1] += b;
    context.state[2] += c;
    context.state[3] += d;
    context.state[4] += e;
    context.state[5] += f;
    context.state[6] += g;
    context.state[7] += h;
    context.used = 0;
}

inline void sha256_init(thread Sha256Context& context) {
    context.state[0] = 0x6a09e667u;
    context.state[1] = 0xbb67ae85u;
    context.state[2] = 0x3c6ef372u;
    context.state[3] = 0xa54ff53au;
    context.state[4] = 0x510e527fu;
    context.state[5] = 0x9b05688cu;
    context.state[6] = 0x1f83d9abu;
    context.state[7] = 0x5be0cd19u;
    context.used = 0;
    context.total = 0;
}

inline void sha256_byte(thread Sha256Context& context, uchar value) {
    context.block[context.used++] = value;
    context.total += 1;
    if (context.used == 64) sha256_compress(context);
}

inline void sha256_u64_le(thread Sha256Context& context, ulong value) {
    for (uint shift = 0; shift < 64; shift += 8) {
        sha256_byte(context, uchar((value >> shift) & 0xffu));
    }
}

inline void sha256_u32_le(thread Sha256Context& context, uint value) {
    for (uint shift = 0; shift < 32; shift += 8) {
        sha256_byte(context, uchar((value >> shift) & 0xffu));
    }
}

inline void sha256_u16_le(thread Sha256Context& context, ushort value) {
    sha256_byte(context, uchar(value & 0xffu));
    sha256_byte(context, uchar((value >> 8) & 0xffu));
}

inline void sha256_constant_bytes(
    thread Sha256Context& context,
    constant const uchar* bytes,
    uint length) {
    for (uint index = 0; index < length; ++index) sha256_byte(context, bytes[index]);
}

inline void sha256_device_bytes(
    thread Sha256Context& context,
    device const uchar* bytes,
    uint length) {
    for (uint index = 0; index < length; ++index) sha256_byte(context, bytes[index]);
}

inline void sha256_device_chunk_256(
    thread Sha256Context& context,
    device const uchar* bytes) {
    // The chunk-leaf prefix is 37/38/40 bytes depending on its domain. Fill
    // that partial block, then compress complete device blocks without the
    // per-byte fullness branch.
    uint first = 64 - context.used;
    uint consumed = 0;
    for (; consumed < first; ++consumed) sha256_byte(context, bytes[consumed]);
    while (consumed + 64 <= 256) {
        for (uint index = 0; index < 64; ++index) {
            context.block[index] = bytes[consumed + index];
        }
        context.used = 64;
        context.total += 64;
        sha256_compress(context);
        consumed += 64;
    }
    for (; consumed < 256; ++consumed) sha256_byte(context, bytes[consumed]);
}

inline void sha256_digest_words(
    thread Sha256Context& context,
    threadgroup const uint* words) {
    for (uint word = 0; word < 8; ++word) {
        uint value = words[word];
        sha256_byte(context, uchar(value >> 24));
        sha256_byte(context, uchar(value >> 16));
        sha256_byte(context, uchar(value >> 8));
        sha256_byte(context, uchar(value));
    }
}

inline void sha256_finalize(thread Sha256Context& context) {
    ulong bit_length = context.total * 8;
    sha256_byte(context, 0x80u);
    while (context.used != 56) sha256_byte(context, 0u);
    for (int shift = 56; shift >= 0; shift -= 8) {
        sha256_byte(context, uchar((bit_length >> uint(shift)) & 0xffu));
    }
}

inline void sha256_store_device(thread const uint* words, device uchar* output) {
    for (uint word = 0; word < 8; ++word) {
        uint value = words[word];
        output[word * 4] = uchar(value >> 24);
        output[word * 4 + 1] = uchar(value >> 16);
        output[word * 4 + 2] = uchar(value >> 8);
        output[word * 4 + 3] = uchar(value);
    }
}

inline void sha256_store_threadgroup_device(
    threadgroup const uint* words,
    device uchar* output) {
    for (uint word = 0; word < 8; ++word) {
        uint value = words[word];
        output[word * 4] = uchar(value >> 24);
        output[word * 4 + 1] = uchar(value >> 16);
        output[word * 4 + 2] = uchar(value >> 8);
        output[word * 4 + 3] = uchar(value);
    }
}

inline void hash_chunk(
    device const uchar* bytes,
    uint chunk_index,
    constant const uchar* domain,
    uint domain_length,
    thread uint* output) {
    Blake3Context context;
    b3_init(context);
    b3_constant_bytes(context, domain, domain_length);
    b3_u64_le(context, 4);
    b3_u32_le(context, chunk_index);
    b3_u64_le(context, 256);
    b3_device_chunk_256(context, bytes + chunk_index * 256);
    b3_finish(context, output);
}

inline void hash_parent(
    constant const uchar*,
    uint,
    threadgroup const uint* left,
    threadgroup const uint* right,
    thread uint* output) {
    Blake3Context context;
    b3_init(context);
    b3_digest_words(context, left);
    b3_digest_words(context, right);
    b3_finish(context, output);
}

inline void reduce_tree(
    threadgroup uint* nodes,
    uint count,
    constant const uchar* tree_domain,
    uint tree_domain_length,
    uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = count; width > 1; width >>= 1) {
        uint parent[8];
        if (tid < width / 2) {
            hash_parent(
                tree_domain,
                tree_domain_length,
                nodes + tid * 16,
                nodes + tid * 16 + 8,
                parent);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < width / 2) {
            for (uint word = 0; word < 8; ++word) nodes[tid * 8 + word] = parent[word];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qx_verification_hash_state_kv(
    device const uchar* residual_snapshots [[buffer(0)]],
    device const uchar* kv_arena [[buffer(1)]],
    device uchar* outputs [[buffer(2)]],
    constant VerificationHashParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint layer [[threadgroup_position_in_grid]]) {
    if (layer >= params.layers) return;
    threadgroup uint state_nodes[16 * 8];
    threadgroup uint key_nodes[8 * 8];
    threadgroup uint value_nodes[8 * 8];
    device const uchar* residual = residual_snapshots + layer * params.residual_stride;
    device const uchar* key = kv_arena + params.key_offsets[layer];
    device const uchar* value = kv_arena + params.value_offsets[layer];
    device uchar* output = outputs + layer * 64;
    bool checkpoint = ((layer + 1) % params.checkpoint_interval == 0)
        || layer + 1 == params.layers;
    uint state_chunks = checkpoint ? params.residual_stride / 256 : 1;
    uint key_chunks = params.key_bytes[layer] / 256;
    uint value_chunks = params.value_bytes[layer] / 256;
    uint group = tid / 32;
    uint lane = tid % 32;

    // Give residual, key, and value independent SIMD groups so their BLAKE3
    // compression streams can make progress concurrently.
    if (group == 0 && checkpoint && lane < state_chunks) {
        uint digest[8];
        hash_chunk(residual, lane, STATE_CHUNK_DOMAIN, 17, digest);
        for (uint word = 0; word < 8; ++word) state_nodes[lane * 8 + word] = digest[word];
    } else if (group == 0 && !checkpoint && lane == 0) {
        for (uint word = 0; word < 8; ++word) state_nodes[word] = 0;
    } else if (group == 1 && lane < key_chunks) {
        uint digest[8];
        hash_chunk(key, lane, KV_KEY_CHUNK_DOMAIN, 18, digest);
        for (uint word = 0; word < 8; ++word) key_nodes[lane * 8 + word] = digest[word];
    } else if (group == 2 && lane < value_chunks) {
        uint digest[8];
        hash_chunk(value, lane, KV_VALUE_CHUNK_DOMAIN, 20, digest);
        for (uint word = 0; word < 8; ++word) value_nodes[lane * 8 + word] = digest[word];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint level = 0; level < 4; ++level) {
        uint width = group == 0 ? state_chunks : (group == 1 ? key_chunks : value_chunks);
        width >>= level;
        uint parent[8];
        if (group < 3 && width > 1 && lane < width / 2) {
            if (group == 0) {
                hash_parent(STATE_DOMAIN, 11, state_nodes + lane * 16,
                    state_nodes + lane * 16 + 8, parent);
            } else if (group == 1) {
                hash_parent(KV_KEY_TREE_DOMAIN, 17, key_nodes + lane * 16,
                    key_nodes + lane * 16 + 8, parent);
            } else {
                hash_parent(KV_VALUE_TREE_DOMAIN, 19, value_nodes + lane * 16,
                    value_nodes + lane * 16 + 8, parent);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group < 3 && width > 1 && lane < width / 2) {
            if (group == 0) {
                for (uint word = 0; word < 8; ++word) state_nodes[lane * 8 + word] = parent[word];
            } else if (group == 1) {
                for (uint word = 0; word < 8; ++word) key_nodes[lane * 8 + word] = parent[word];
            } else {
                for (uint word = 0; word < 8; ++word) value_nodes[lane * 8 + word] = parent[word];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        b3_store_threadgroup_device(state_nodes, output);
        Blake3Context kv;
        b3_init(kv);
        b3_constant_bytes(kv, KV_DOMAIN, 8);
        b3_u64_le(kv, 2);
        b3_u16_le(kv, ushort(layer));
        b3_u64_le(kv, 4);
        b3_u32_le(kv, params.token);
        b3_u64_le(kv, 32);
        b3_digest_words(kv, key_nodes);
        b3_u64_le(kv, 32);
        b3_digest_words(kv, value_nodes);
        uint kv_output[8];
        b3_finish(kv, kv_output);
        b3_store_device(kv_output, output + 32);
    }
}
