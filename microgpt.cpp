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

struct Arena{
    data_T* data; // pointer to data array
    grad_T* grad; // pointer to grad array
    int* i_child0; // pointer to first child index array
    int* i_child1; // pointer to second child index array
    grad_T* local_grad0; // pointer to first local grad array
    grad_T* local_grad1; // pointer to second local grad array

    int size = 0; // current number of elements in the arena per array (num of data = num of grad = ...)
    int cap = 0; // maximum size (number of elements) our arena can handle at the moment PER ARRAY

    void init(int n){
        cap = n;
        data = (data_T*)std::malloc(n * sizeof(data_T));
        grad = (grad_T*)std::malloc(n * sizeof(grad_T));
        i_child0 = (int*)std::malloc(n * sizeof(int));
        i_child1 = (int*)std::malloc(n * sizeof(int));
        local_grad0 = (grad_T*)std::malloc(n * sizeof(grad_T));
        local_grad1 = (grad_T*)std::malloc(n * sizeof(grad_T));
    }

    void grow(){ // double memory allocation for all arrays (Since they grow in parallel)
        int new_cap = cap * 2;
        data = (data_T*)std::realloc(data, new_cap * sizeof(data_T));
        grad = (grad_T*)std::realloc(grad, new_cap * sizeof(grad_T));
        i_child0 = (int*)std::realloc(i_child0, new_cap * sizeof(int));
        i_child1 = (int*)std::realloc(i_child1, new_cap * sizeof(int));
        local_grad0 = (grad_T*)std::realloc(local_grad0, new_cap * sizeof(grad_T));
        local_grad1 = (grad_T*)std::realloc(local_grad1, new_cap * sizeof(grad_T));
        cap = new_cap;
    }

    int size() const { return size; } // current arena next pointer
    inline void ensure() { if (size == cap) grow(); }

    void truncate(int n) { size = n; } // remove elements (ignore them) until size n
    
    void zero_grad(int n) { std::memset(grad, 0, n * sizeof(grad_T)); }
    
    inline int push_no_op(data_T d){
        ensure();
        int i = size++;
        data[i] = d;
        grad[i] = 0;
        i_child0[i] = NO_CHILD; i_child1[i] = NO_CHILD;
        local_grad0[i] = 0; local_grad1[i] = 0;
        return i;
    }

    inline int push_unary_op(data_T d, int i_c, grad_T g){
        ensure();
        int i = size++;
        data[i] = d;
        grad[i] = 0;
        i_child0[i] = i_c; i_child1[i] = NO_CHILD;
        local_grad0[i] = g; local_grad1[i] = 0;
        return i;
    }
    
    inline int push_binary_op(data_T d, int i_c0, grad_T g0, int i_c1, grad_T g1){
        ensure();
        int i = size++;
        data[i] = d;
        grad[i] = 0;
        i_child0[i] = i_c0; i_child1[i] = i_c1;
        local_grad0[i] = g0; local_grad1[i] = g1;
        return i;
    }

    void cleanup(){
        std::free(data); std::free(grad);
        std::free(i_child0); std::free(i_child1);
        std::free(local_grad0); std::free(local_grad1);
    }
};

Arena arena{};// memory management for all of our values

void backward(int i_loss){
    arena.grad[i_loss] = 1;
    for (int i = i_loss; i>=0 ; i--){
        grad_T g = arena.grad[i];
        if (g == 0.0f){continue;} // skip node when  grad is 0
        int i_c0 = arena.i_child0[i];
        int i_c1 = arena.i_child1[i];
        if (i_c0 != NO_CHILD){
            arena.grad[i_c0] += arena.local_grad0[i] * g;
            if (i_c1 != NO_CHILD){
                arena.grad[i_c1] += arena.local_grad1[i] * g;
            }
        }
    }
}

// operations (binary)
inline int vadd(int a, int b) { return arena.push_binary_op(arena.data[a] + arena.data[b], a, 1.0, b, 1.0); }
inline int vsub(int a, int b) { return arena.push_binary_op(arena.data[a] - arena.data[b], a, 1.0, b, -1.0); }
inline int vmul(int a, int b) { return arena.push_binary_op(arena.data[a] * arena.data[b], a, arena.data[b], b, arena.data[a]); }
inline int vdiv(int a, int b) { return arena.push_binary_op(arena.data[a] / arena.data[b], a, 1.0/arena.data[b], b, -arena.data[a]/(arena.data[b]*arena.data[b])); }

// operations (unary)
inline int vneg(int a) { return arena.push_unary_op(-arena.data[a], a, -1.0); }
inline int vlog(int a) { return arena.push_unary_op(std::log(arena.data[a]), a, 1.0 / arena.data[a]); }
inline int vexp(int a) { data_T e = std::exp(arena.data[a]); return arena.push_unary_op(e, a, e); }
inline int vrelu(int a) { return arena.push_unary_op(std::max(0.0, arena.data[a]), a, arena.data[a] > 0 ? 1.0 : 0.0); }
inline int vpow(int a, data_T n) { return arena.push_unary_op(std::pow(arena.data[a], n), a, n * std::pow(arena.data[a], n - 1)); }

// operations with consts (1 node instead of 2)
inline int mul_const(int a, data_T c) { return arena.push_unary_op(arena.data[a] * c, a, c); }
inline int div_const(int a, data_T c) { return arena.push_unary_op(arena.data[a] / c, a, 1.0 / c); }
inline int add_const(int a, data_T c) { return arena.push_unary_op(arena.data[a] + c, a, 1.0); }
inline int sub_const(int a, data_T c) { return arena.push_unary_op(arena.data[a] - c, a, 1.0); }


struct Matrix {
    int data_start;
    int rows, cols;

    Matrix(int rows, int cols, float std=0.08) : rows(rows), cols(cols) {
        data_start = arena.size(); // start at the current arena pointer
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
struct FlatKVCache{
    std::vector<int> data; // indices to find the values we need
    int n_layer, dim;
    int counts [N_LAYER] = {}; // timesteps per layer (max N_LAYER layers)

    FlatKVCache(int n_layer, int d) : n_layer(n_layer), dim(dim) {
        data.reserve(n_layer * BLOCK_SIZE * dim); // pre-alloc for up to BLOCK_SIZE timesteps (context length)
    }

    void push(int i_layer, const int* vals){
        int base = 0;
        for (int l = 0; l < i_layer; l++) base += counts[l] * dim; // skip past all time steps for all previous layers
        base += counts[i_layer] * dim; // skip past this existing time steps for this layer
        // insert at the right position
        data.insert(data.begin() + base, vals, vals + dim);
        counts[i_layer]++; // we are now at the next time step for layer: i_layer
    }

    int get(int i_layer, int t, int d){
        int base = 0;
        for (int l = 0; l < i_layer; l++) base += counts[l] * dim;
        return data[base + t * dim + d];
    }

    int num_timesteps(int i_layer) const { return counts[i_layer]; }

    void clear() {
        data.clear();
        std::memset(counts, 0, sizeof(counts));
    }
};


void linear(int* out, const int* x, Matrix& w){ // matrix * vector 
    for(int i=0; i<w.rows;i++){
        int sum = vmul(w.at(i,0), x[0]);
        for(int j=1; j<w.cols;j++){
            sum = vadd(sum, vmul(w.at(i,j), x[j]));
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
    for (int i = 1; i < x_len; i++) total = vadd(total, vmul(x[i], x[i]));
    total = div_const(total, x_len);
    int scale = vpow(add_const(total,1e-5), data_T{-0.5});
    for (int i = 0; i < x_len; i++) out[i] = vmul(x[i], scale);
}


void gpt(
    int* logits_out,
    const int token_id, 
    const int pos_id,
    FlatKVCache& keys,
    FlatKVCache& values,
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
            keys.push(i_layer, k);
            values.push(i_layer, v);
            // multi-head attention
            int x_attn[N_EMBD];
            int num_timesteps = keys.num_timesteps(i_layer);
            for(int h=0;h<N_HEAD;h++){
                int hs = h*HEAD_DIM; // starting index of the full N_EMBD vector for head

                // computing attention dot(q_h, k_h[t]) / sqrt(head_dim)
                int attention_logits[BLOCK_SIZE];
                for (int t = 0; t < num_timesteps; t++) {
                    int sum = vmul(q[hs],keys.get(i_layer, t, hs));
                    for (int j=1;j<HEAD_DIM;j++){
                        sum = vadd(sum, vmul(q[hs+j],keys.get(i_layer, t, hs + j)));
                    }
                    attention_logits[t] = mul_const(sum, INV_SQRT_HEAD_DIM)
                }
                // softmax
                int attn_weights[BLOCK_SIZE];
                softmax(attn_weights, attention_logits, num_timesteps);
                
                // weighted sum of values
                for (int j=0;j<HEAD_DIM;j++){
                    int sum = vmul(attn_weights[0], values.get(i_layer, 0, hs+ j));
                    for (int t=1;t<num_timesteps;t++){
                        sum = vadd(sum,vmul(attn_weights[t],values.get(i_layer, t, hs + j)));
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
            for (int i = 0; i < 4 * N_EMBD; i++) x[i] = vrelu(x[i]);
            linear(x, mlp_hidden,  state_dict.layers[i_layer].mlp_fc2);
            for (int i = 0; i < N_EMBD; i++) x[i] = vadd(x[i], x_residual[i]);
        }
        linear(logits_out, x, state_dict.lm_head); 
}

int main() {
    // reserve the number of values in arena, so that data is stored contiguously without moving (faster backward pass using Wengert tape)
    // alternative: use deque for arena and remove the following reserve line 
    arena.reserve(200000);
    if (!std::filesystem::exists("input.txt")){
        LOG("Downloading input.txt ...");
        system("wget -q -O input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt");
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
    for (auto& name: docs){
        uchars.insert(name.begin(), name.end());
    }
    int BOS = uchars.size(); // token id for a special Beginning of Sequence (BOS) token
    int vocab_size = uchars.size()+1;
    LOG("Vocab size is: "<<vocab_size);
    
    Model state_dict(vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER);
    std::vector<Value*> params = state_dict.params();
    LOG("Number of params: "<<params.size());

    float learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    std::vector<double> m(params.size(), 0.0);
    std::vector<double> v(params.size(), 0.0);
    int num_steps = 1000;

    // training loop
    for (int step=0; step< num_steps; step++){
        // Take a document, tokenize it, surround it by BOS tokens
        std::string doc = docs[step%docs.size()];
        std::vector<int> tokens;
        tokens.push_back(BOS);
        for (char ch:doc){
            tokens.push_back(std::distance(uchars.begin(), uchars.find(ch)));
        }
        tokens.push_back(BOS);
        int n = std::min(BLOCK_SIZE, (int)tokens.size() - 1);

        // Forward tokens through the model
        KVCache keys(N_LAYER), values(N_LAYER);
        std::vector<Value*> losses{};
        for (int pos_id=0; pos_id<n;pos_id++){
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id+1];
            std::vector<Value*> logits = gpt(token_id, pos_id, keys, values, state_dict);
            std::vector<Value*> probs = softmax(logits);
            Value* loss_t = neg(log(probs[target_id]));
            losses.push_back(loss_t);
        }
        Value* total_losses = losses[0]; 
        for (int i = 1; i<losses.size();i++) total_losses = add(total_losses, losses[i]);
        Value* loss = mul(total_losses, make_value(1.0 / n));

        // backward pass
        backward(loss);

        // adam optimizer
        float lr_t = learning_rate*(1-(double)step/num_steps);
        double beta1_pow = std::pow(beta1,(step + 1));
        double beta2_pow = std::pow(beta2,(step + 1));
        for (int i = 0;i<params.size();i++){
            Value* p = params[i];
            m[i] = beta1 * m[i] + (1 - beta1) * p->grad;
            v[i] = beta2 * v[i] + (1 - beta2) * p->grad * p->grad;
            grad_T m_hat = m[i] / (1 - beta1_pow);
            grad_T v_hat = v[i] / (1 - beta2_pow);
            p->data -= lr_t*m_hat / (std::sqrt(v_hat)+eps_adam);
            p->grad = 0;
        }
        LOG("Step "<<(step+1)<<" / "<<num_steps<<" | loss "<< loss->data);
        LOG("Arena size: " << arena.size());

        arena.clear(); // Clear memory after the end of backprop
    }

    float temperature = 0.5;
    LOG("\n\nTime for inference---------------");
    std::vector<char> idx_to_char(uchars.begin(), uchars.end());
    for (int sample_idx = 0; sample_idx<20;sample_idx++){
        KVCache keys(N_LAYER), values(N_LAYER);
        int token_id = BOS;
        std::vector<char> samples;
        for (int pos_id = 0;pos_id<BLOCK_SIZE;pos_id++){
            std::vector<Value*> logits = gpt(token_id, pos_id, keys, values, state_dict);
            for (int i = 0; i < logits.size(); i++)
                logits[i] = div(logits[i],make_value(temperature));
            std::vector<Value*> probs = softmax(logits);
            std::vector<double> weights;
            for (auto& p : probs) weights.push_back(p->data);
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            token_id = dist(rng);
            if (token_id == BOS){
                break;
            }
            samples.push_back(idx_to_char[token_id]);
        }
        std::string result(samples.begin(), samples.end());
        LOG("Sample: "<< sample_idx<<": "<<result);
        arena.clear(); // Clear memory after the end of inference pass
    }
    return 0;
}