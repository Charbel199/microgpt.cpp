#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <random>
#include <fstream>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <deque>
#include <cstring>

#ifdef DEBUG
#define LOG(msg) std::cerr << "[LOG] " << msg << std::endl
#else
#define LOG(msg)
#endif

std::mt19937 rng(42); //random seed
using data_T = double;
using grad_T = double;

// defining our model params
constexpr int N_LAYER = 1;
constexpr int N_EMBD = 16;
constexpr int BLOCK_SIZE = 16;
constexpr int N_HEAD = 4;
constexpr int HEAD_DIM = N_EMBD / N_HEAD;
const data_T INV_SQRT_HEAD_DIM = 1.0 / std::sqrt((data_T)HEAD_DIM);
constexpr int NO_CHILD = -1; // since children point to indices -> no child index = -1
constexpr int MAX_VOCAB_SIZE = 27;
constexpr int NUM_STEPS = 1000;

// we want to avoid storing 2 grad_T (with doubles = 2*8 bytes = 16 bytes) for every element, 
// instead we will store one uint8 op per element (only 1 byte) and compute the local grad on the fly in backward() based on the op type and the values of the children
// this should lead to better performance because we are currently memory bound
// Note: Also added a Fused Multiply Add op (FMA)
enum Op : uint8_t { OP_CONST, OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_LOG, OP_EXP, OP_RELU, OP_INV_SQRT, OP_INV_LOG, OP_SUB_CONST, OP_MUL_CONST, OP_DIV_CONST, OP_FMA };



int weights_end = 0;
struct Arena{
    data_T* data; // pointer to data array
    grad_T* grad; // pointer to grad array
    int* i_child0; // pointer to first child index array
    int* i_child1; // pointer to second child index array
    int* i_child2; // pointer to third child index array (for fused ops, currently only FMA)
    Op* op; // pointer to the operation type

    int size = 0; // current number of elements in the arena per array (num of data = num of grad = ...)
    int cap = 0; // maximum size (number of elements) our arena can handle at the moment PER ARRAY

    void init(int n){
        cap = n;
        data = (data_T*)std::malloc(n * sizeof(data_T));
        grad = (grad_T*)std::malloc(n * sizeof(grad_T));
        i_child0 = (int*)std::malloc(n * sizeof(int));
        i_child1 = (int*)std::malloc(n * sizeof(int));
        i_child2 = (int*)std::malloc(n * sizeof(int));
        op = (Op*)std::malloc(n * sizeof(Op));
    }

    void grow(){ // double memory allocation for all arrays (Since they grow in parallel)
        int new_cap = cap * 2;
        data = (data_T*)std::realloc(data, new_cap * sizeof(data_T));
        grad = (grad_T*)std::realloc(grad, new_cap * sizeof(grad_T));
        i_child0 = (int*)std::realloc(i_child0, new_cap * sizeof(int));
        i_child1 = (int*)std::realloc(i_child1, new_cap * sizeof(int));
        i_child2 = (int*)std::realloc(i_child2, new_cap * sizeof(int));
        op = (Op*)std::realloc(op, new_cap * sizeof(Op));
        cap = new_cap;
    }

    int get_size() const { return size; } // current arena next pointer
    inline void ensure() { if (size == cap) grow(); }

    void truncate(int n) { size = n; } // remove elements (ignore them) until size n
    
    void zero_grad(int n) { std::memset(grad, 0, n * sizeof(grad_T)); }
    
    inline int push_no_op(data_T d){
        ensure();
        int i = size++;
        data[i] = d;
        op[i] = OP_CONST;
        return i;
    }

    inline int push_unary_op(data_T d, int i_c, Op o){
        ensure();
        int i = size++;
        data[i] = d;
        i_child0[i] = i_c;
        op[i] = o;
        return i;
    }
    
    inline int push_binary_op(data_T d, int i_c0, int i_c1, Op o){
        ensure();
        int i = size++;
        data[i] = d;
        i_child0[i] = i_c0; i_child1[i] = i_c1;
        op[i] = o;
        return i;
    }

    void cleanup(){
        std::free(data); std::free(grad);
        std::free(i_child0); std::free(i_child1); std::free(i_child2);
        std::free(op);
    }
};

Arena arena{};// memory management for all of our values

void backward(int i_loss){
    std::memset(arena.grad + weights_end, 0, (i_loss + 1 - weights_end) * sizeof(grad_T));
    arena.grad[i_loss] = 1.0f;

    // cache base pointers in registers
    data_T* __restrict__ p_data = arena.data;
    grad_T* __restrict__ p_grad = arena.grad;
    const int* __restrict__ p_c0 = arena.i_child0;
    const int* __restrict__ p_c1 = arena.i_child1;
    const int* __restrict__ p_c2 = arena.i_child2;
    const Op* __restrict__ p_op = arena.op;

    for (int i = i_loss; i >= 0; i--){
        Op o = p_op[i];
        if (o == OP_CONST) continue; // skip if constant
        grad_T g = p_grad[i];
        if (g == 0.0f) continue; // skip if grad is 0
        int c0 = p_c0[i]; // compiler would probably do that automatically but it's okay
        switch (o) {
            case OP_ADD:
                p_grad[c0] += g;
                p_grad[p_c1[i]] += g;
                break;
            case OP_MUL: {
                int c1 = p_c1[i];
                p_grad[c0] += g * p_data[c1];
                p_grad[c1] += g * p_data[c0];
                break;
            }
            case OP_DIV: {
                int c1 = p_c1[i]; // compiler would probably do that automatically but it's okay
                data_T d1 = p_data[c1];
                p_grad[c0] += g / d1;
                p_grad[c1] -= g * p_data[c0] / (d1 * d1);
                break;
            }
            case OP_NEG:
                p_grad[c0] -= g;
                break;
            case OP_LOG:
                p_grad[c0] += g / p_data[c0];
                break;
            case OP_EXP:
                p_grad[c0] += g * p_data[i];
                break;
            case OP_RELU:
                if (p_data[c0] > 0.0f) p_grad[c0] += g;
                break;
            case OP_INV_SQRT:
                p_grad[c0] -= 0.5f * g * p_data[i] / (p_data[c0] + 1e-5f);
                break;
            case OP_INV_LOG:
                p_grad[c0] -= g / p_data[c0];
                break;
            case OP_SUB_CONST:
                p_grad[c0] += g;
                break;
            case OP_MUL_CONST:
                p_grad[c0] += g * p_data[p_c1[i]];
                break;
            case OP_DIV_CONST:
                p_grad[c0] += g / p_data[p_c1[i]];
                break;
            case OP_FMA: {
                // result = a*b + c -> grad[a] += g*b, grad[b] += g*a, grad[c] += g
                int c1 = p_c1[i];
                p_grad[c0] += g * p_data[c1];
                p_grad[c1] += g * p_data[c0];
                p_grad[p_c2[i]] += g;
                break;
            }
            default: break;
        }
    }
}

// fused (TODO: add a push_fused_op at some point)
inline int vmul_add(int a, int b, int c) {
    arena.ensure();
    int i = arena.size++;
    arena.data[i] = arena.data[a] * arena.data[b] + arena.data[c];
    arena.i_child0[i] = a;
    arena.i_child1[i] = b;
    arena.i_child2[i] = c;
    arena.op[i] = OP_FMA;
    return i;
}

// operations (binary)
inline int vadd(int a, int b) { return arena.push_binary_op(arena.data[a] + arena.data[b], a, b, OP_ADD); }
inline int vmul(int a, int b) { return arena.push_binary_op(arena.data[a] * arena.data[b], a, b, OP_MUL); }
inline int vdiv(int a, int b) { return arena.push_binary_op(arena.data[a] / arena.data[b], a, b, OP_DIV); }

// operations (unary)
inline int vneg(int a) { return arena.push_unary_op(-arena.data[a], a, OP_NEG); }
inline int vlog(int a) { return arena.push_unary_op(std::log(arena.data[a]), a, OP_LOG); }
inline int vinv_log(int a) { return arena.push_unary_op(-std::log(arena.data[a]), a, OP_INV_LOG); }
inline int vexp(int a) { return arena.push_unary_op(std::exp(arena.data[a]), a, OP_EXP); }
inline int vrelu(int a) { return arena.push_unary_op(std::max(data_T{0.0}, arena.data[a]), a, OP_RELU); }
inline int vinv_sqrt(int a) { return arena.push_unary_op(std::pow(arena.data[a] + 1e-5f, -0.5f), a, OP_INV_SQRT); }


// operations with consts (1 node instead of 2)
inline int mul_const(int a, data_T c) { int ic = arena.push_no_op(c); return arena.push_binary_op(arena.data[a] * c, a, ic, OP_MUL_CONST); }
inline int div_const(int a, data_T c) { int ic = arena.push_no_op(c); return arena.push_binary_op(arena.data[a] / c, a, ic, OP_DIV_CONST); }
inline int sub_const(int a, data_T c) { return arena.push_unary_op(arena.data[a] - c, a, OP_SUB_CONST); }

struct Matrix {
    int data_start;
    int rows, cols;

    Matrix(int rows, int cols, float std=0.08) : rows(rows), cols(cols) {
        data_start = arena.get_size(); // start at the current arena pointer
        std::normal_distribution<data_T> dist(0.0, std);
        for (int i = 0; i < rows*cols; i++) arena.push_no_op(dist(rng));
    }

    int at(int i, int j) const { return data_start + i * cols + j;}
};
struct Layer {
    Matrix attn_wq, attn_wk, attn_wv, attn_wo;
    Matrix mlp_fc1, mlp_fc2;

    Layer(int n_embd) : 
        attn_wq(n_embd, n_embd),
        attn_wk(n_embd, n_embd),
        attn_wv(n_embd, n_embd),
        attn_wo(n_embd, n_embd),
        mlp_fc1(4* n_embd, n_embd),
        mlp_fc2(n_embd, 4 * n_embd)
    {}
};
struct Model {
    Matrix wte, wpe, lm_head;
    std::vector<Layer> layers;

    Model(int vocab_size, int n_embd, int block_size, int n_layer):
        wte(vocab_size, n_embd),
        wpe(block_size, n_embd),
        lm_head(vocab_size, n_embd)
    {
        for(int i =0;i<n_layer;i++){
            layers.emplace_back(n_embd);
        }
    }

    std::vector<int> params() {
        std::vector<int> p;
        // helper to add all elements of a matrix
        auto add = [&](Matrix& m) {
            for (int i = 0; i < m.rows*m.cols; i++) p.push_back(m.data_start+i); // add indices to matrix values in arena
        };
        add(wte);
        add(wpe);
        add(lm_head);
        for (auto& layer : layers) {
            add(layer.attn_wq);
            add(layer.attn_wk);
            add(layer.attn_wv);
            add(layer.attn_wo);
            add(layer.mlp_fc1);
            add(layer.mlp_fc2);
        }
        return p;
    }
};

/*
keys[layer][timestep][dimension]
       |        |         |
       |        |         -- which of the 16 numbers in the key vector (0..N_EMBD-1)
       |        -- which token position was processed (grows: 0, 1, 2, ...)
       -- which transformer layer (0..N_LAYER-1). Each layer has its own Q/K/V weights,
          so each layer produces different keys and values

key = [k0, k1, k2, k3,   k4, k5, k6, k7,   k8, k9, k10, k11,   k12, k13, k14, k15]
       --- head 0 -----  ---- head 1 ----  ---- head 2 -----   ----- head 3 -----
*/
using KVCache = int[N_LAYER][BLOCK_SIZE][N_EMBD]; // fully stack allocated fixed-size 3D array


void linear(int* out, const int* x, Matrix& w){ // matrix * vector 
    for(int i=0; i<w.rows;i++){
        int sum = vmul(w.at(i,0), x[0]);
        for(int j=1; j<w.cols;j++){
            sum = vmul_add(w.at(i,j), x[j], sum);
        }
        out[i] = sum;
    }
}

void softmax(int* out, const int* logits, int logits_len){
    data_T max_val = arena.data[logits[0]];
    for (int i = 0; i < logits_len; i++) if (arena.data[logits[i]] > max_val) max_val = arena.data[logits[i]];
    int exps[MAX_VOCAB_SIZE]; // indices of exps
    for (int i = 0; i < logits_len; i++) exps[i] = vexp(sub_const(logits[i], max_val));
    int total = exps[0];
    for (int i = 1; i< logits_len; i++) total = vadd(total, exps[i]);
    for (int i = 0; i < logits_len; i++) out[i] = vdiv(exps[i], total);
}

void rmsnorm(int* out, const int* x, int x_len){
    int total = vmul(x[0], x[0]);
    for (int i = 1; i < x_len; i++) total = vmul_add(x[i], x[i], total);
    total = div_const(total, x_len);
    int scale = vinv_sqrt(total);
    for (int i = 0; i < x_len; i++) out[i] = vmul(x[i], scale);
}


void gpt(
    int* logits_out,
    const int token_id, 
    const int pos_id,
    KVCache& keys,
    KVCache& values,
    Model& state_dict){
        int x[N_EMBD]; // joint token and position embedding
        int tmp[N_EMBD]; // tmp array for rmsnorm, since we can't do it in place
        for (int j=0;j< N_EMBD;j++)
            x[j] = vadd(state_dict.wte.at(token_id, j), state_dict.wpe.at(pos_id, j));
        rmsnorm(tmp, x, N_EMBD);
        std::memcpy(x, tmp, N_EMBD * sizeof(int));

        for (int i_layer=0; i_layer<N_LAYER;i_layer++){
            // save residual
            int x_residual[N_EMBD];
            std::memcpy(x_residual, x, N_EMBD * sizeof(int));
            // rmsnorm
            rmsnorm(tmp, x, N_EMBD);
            std::memcpy(x, tmp, N_EMBD * sizeof(int));
            // Q, K, V
            int q[N_EMBD], k[N_EMBD], v[N_EMBD];
            linear(q, x, state_dict.layers[i_layer].attn_wq);
            linear(k, x, state_dict.layers[i_layer].attn_wk);
            linear(v, x, state_dict.layers[i_layer].attn_wv);
            std::memcpy(keys[i_layer][pos_id], k, N_EMBD * sizeof(int)); // copying this 'keys' chunk into our kvcache on the stack array
            std::memcpy(values[i_layer][pos_id], v, N_EMBD * sizeof(int)); // copying this 'values' chunk into our kvcache on the stack array
            int num_timesteps = pos_id + 1;
            // multi-head attention
            int x_attn[N_EMBD];
            for(int h=0;h<N_HEAD;h++){
                int hs = h*HEAD_DIM; // starting index of the full N_EMBD vector for head

                // computing attention dot(q_h, k_h[t]) / sqrt(head_dim)
                int attention_logits[BLOCK_SIZE];
                for (int t = 0; t < num_timesteps; t++) {
                    int sum = vmul(q[hs], keys[i_layer][t][hs]);
                    for (int j=1;j<HEAD_DIM;j++){
                        sum = vmul_add(q[hs+j], keys[i_layer][t][hs+j], sum);
                    }
                    attention_logits[t] = mul_const(sum, INV_SQRT_HEAD_DIM);
                }
                // softmax
                int attn_weights[BLOCK_SIZE];
                softmax(attn_weights, attention_logits, num_timesteps);
                
                // weighted sum of values
                for (int j=0;j<HEAD_DIM;j++){
                    int sum = vmul(attn_weights[0], values[i_layer][0][hs+j]);
                    for (int t=1;t<num_timesteps;t++){
                        sum = vmul_add(attn_weights[t], values[i_layer][t][hs+j], sum);
                    }
                    x_attn[hs + j] = sum;
                }
            }

            // output projection
            linear(x, x_attn, state_dict.layers[i_layer].attn_wo);

            // residual connection
            for (int i = 0; i < N_EMBD; i++) {
                x[i] = vadd(x[i], x_residual[i]);
            }
            // MLP block
            std::memcpy(x_residual, x, N_EMBD*sizeof(int));
            rmsnorm(tmp, x, N_EMBD);
            std::memcpy(x, tmp, N_EMBD * sizeof(int));

            int mlp_hidden[4 * N_EMBD]; // since shape of mlp_fc1 is (4*N_EMBD, N_EMBD)
            linear(mlp_hidden, x, state_dict.layers[i_layer].mlp_fc1);
            for (int i = 0; i < 4 * N_EMBD; i++) mlp_hidden[i] = vrelu(mlp_hidden[i]);
            linear(x, mlp_hidden,  state_dict.layers[i_layer].mlp_fc2);
            for (int i = 0; i < N_EMBD; i++) x[i] = vadd(x[i], x_residual[i]);
        }
        linear(logits_out, x, state_dict.lm_head); 
}

int main() {
    arena.init(MAX_VOCAB_SIZE * N_EMBD * 3 // accounting for wte, wpe, lm_head
        + N_LAYER * (4 * N_EMBD * N_EMBD + 4 * N_EMBD * N_EMBD) // accounting for attention heads and linear fc layers
    ); // it will grow automatically, this is just a hint to avoid a couple of realloc at the start

    if (!std::filesystem::exists("input.txt")){
        LOG("Downloading input.txt ...");
         if (system("wget -q -O input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt") != 0) LOG("Download failed");
    }
    std::vector<std::string> docs;
    std::ifstream file("input.txt");
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) docs.push_back(line);
    } 
    std::shuffle(docs.begin(), docs.end(), rng);
    LOG("We have "<<docs.size()<<" names.");

    std::set<char> uchars{};
    for (auto& name: docs) uchars.insert(name.begin(), name.end());
    int BOS = uchars.size(); // token id for a special Beginning of Sequence (BOS) token
    int vocab_size = uchars.size()+1;
    LOG("Vocab size is: "<<vocab_size);
    if (vocab_size > MAX_VOCAB_SIZE) {
        throw std::runtime_error("vocab_size (" + std::to_string(vocab_size) + ") exceeds MAX_VOCAB_SIZE (" + std::to_string(MAX_VOCAB_SIZE) + ")");
    }

    // build char lookup
    std::vector<char> idx_to_char(uchars.begin(), uchars.end());
    int char_to_idx[256] = {}; // to cover all ASCII
    { int idx = 0; for (char c : uchars) char_to_idx[(unsigned char)c] = idx++; } // reverse idx_to_char to char_to_idx

    
    Model state_dict(vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER);
    weights_end = arena.get_size();
    std::vector<int> params = state_dict.params();
    LOG("Number of params: "<<params.size());

    float learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    std::vector<float> m(params.size(), 0.0f);
    std::vector<float> v(params.size(), 0.0f);

    // training loop
    for (int step=0; step< NUM_STEPS; step++){
        // Take a document, tokenize it, surround it by BOS tokens
        std::string doc = docs[step%docs.size()];
        int tokens[BLOCK_SIZE+2]; // context + 2 BOS
        int token_len = 0;
        tokens[token_len++] = BOS;
        for (char ch:doc){ tokens[token_len++] = char_to_idx[(unsigned char)ch]; }
        tokens[token_len++] = BOS;
        int n = std::min(BLOCK_SIZE, token_len - 1);

        //forward tokens through the model
        KVCache keys = {}, values = {};
        int losses[BLOCK_SIZE];
        int n_losses = 0;
        for (int pos_id=0; pos_id<n;pos_id++){
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id+1];
            int logits[MAX_VOCAB_SIZE];
            gpt(logits, token_id, pos_id, keys, values, state_dict);
            int probs[MAX_VOCAB_SIZE];
            softmax(probs, logits, vocab_size);
            losses[n_losses++] = vinv_log(probs[target_id]);
        }

        int total_losses = losses[0]; 
        for (int i = 1; i<n_losses;i++) total_losses = vadd(total_losses, losses[i]);
        int loss = mul_const(total_losses, 1.0/n);

        // backward pass
        backward(loss);

        // adam optimizer
        float lr_t = learning_rate*(1-(double)step/NUM_STEPS);
        data_T beta1_pow = std::pow(beta1,(step + 1));
        data_T beta2_pow = std::pow(beta2,(step + 1));
        for (int i = 0;i<params.size();i++){
            int i_p = params[i]; // parameter index
            grad_T p_grad = arena.grad[i_p]; // parameter gradient
            m[i] = beta1 * m[i] + (1 - beta1) * p_grad;
            v[i] = beta2 * v[i] + (1 - beta2) * p_grad * p_grad;
            grad_T m_hat = m[i] / (1 - beta1_pow);
            grad_T v_hat = v[i] / (1 - beta2_pow);
            arena.data[i_p] -= lr_t*m_hat / (std::sqrt(v_hat)+eps_adam);
        }
        LOG("Step "<<(step+1)<<" / "<<NUM_STEPS<<" | loss "<< arena.data[loss]);
        LOG("Arena size: " << arena.get_size());
        arena.truncate(weights_end); // clean until end of weights values
        arena.zero_grad(weights_end);
    }

    float temperature = 0.5;
    LOG("\n\nTime for inference---------------");
    for (int sample_idx = 0; sample_idx<20;sample_idx++){
        KVCache keys = {}, values = {};
        int token_id = BOS;
        std::vector<char> samples;
        for (int pos_id = 0;pos_id<BLOCK_SIZE;pos_id++){
            int logits[MAX_VOCAB_SIZE];
            gpt(logits, token_id, pos_id, keys, values, state_dict);
            for (int i = 0; i < vocab_size; i++)
                logits[i] = mul_const(logits[i],1.0/temperature);
            int probs[MAX_VOCAB_SIZE];
            softmax(probs, logits, vocab_size);

            data_T weights[MAX_VOCAB_SIZE];
            for (int i = 0; i < vocab_size; i++) weights[i] = arena.data[probs[i]];
            std::discrete_distribution<int> dist(weights, weights + vocab_size);
            token_id = dist(rng);
            if (token_id == BOS) break;
            samples.push_back(idx_to_char[token_id]);
        }
        std::string result(samples.begin(), samples.end());
        LOG("Sample: "<< sample_idx<<": "<<result);
        arena.truncate(weights_end); // clean until end of weights values
    }
    return 0;
}