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

    int size() const { return size; }
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
    std::vector<Value> data;
    int rows, cols;

    Matrix(int rows, int cols, float std=0.08) : data(rows * cols), rows(rows), cols(cols) {
        std::normal_distribution<data_T> dist(0.0, std);
        for (auto&v : data) v = Value(dist(rng));
    }

    std::vector<Value> row(int i) {
        auto start = data.begin() + i * cols;
        return std::vector<Value>(start, start + cols);
    }

    Value* operator()(int i, int j) { return &data[i * cols + j]; }
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

    std::vector<Value*> params() {
        std::vector<Value*> p;
        // helper to add all elements of a matrix
        auto add = [&](Matrix& m) {
            for (auto& v : m.data) p.push_back(&v);
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
using KVCache = std::vector<std::vector<std::vector<Value*>>>;

std::vector<Value*> linear(const std::vector<Value*>& x, Matrix& w){ // matrix * vector 
    std::vector<Value*> out;
    out.reserve(w.rows); //optimization
    for(int i=0; i<w.rows;i++){
        Value* sum = mul(w(i,0), x[0]); // to avoid creating a noop const 0
        for(int j=1; j<w.cols;j++){
            sum = add(sum, mul(w(i,j), x[j]));
        }
        out.push_back(sum);
    }
    return out;
}

std::vector<Value*> softmax(const std::vector<Value*>& logits){
    data_T max_val = logits[0]->data;
    for (auto* v: logits) if (v->data > max_val) max_val = v->data;
    std::vector<Value*> exps;
    exps.reserve(logits.size()); //optimization
    for (auto* v : logits) exps.push_back(exp(sub(v, make_value(max_val))));
    Value* total = exps[0];
    for (int i = 1; i<exps.size(); i++) total = add(total, exps[i]);
    std::vector<Value*> out;
    out.reserve(logits.size()); //optimization
    for (auto* v : exps) out.push_back(div(v, total));
    return out;
}

std::vector<Value*> rmsnorm(const std::vector<Value*>& x){
    Value* total= mul(x[0], x[0]);
    for (int i = 1; i < x.size(); i++) total = add(total, mul(x[i], x[i]));
    total = div(total,make_value((data_T)x.size()));
    Value* scale = pow(add(total,make_value(1e-5)), data_T{-0.5});
    std::vector<Value*> out;
    out.reserve(x.size()); //optimization
    for (auto* xi : x) out.push_back(mul(xi, scale));
    return out;
}

std::vector<Value*> gpt(
    const int token_id, 
    const int pos_id,
    KVCache& keys,
    KVCache& values,
    Model& state_dict){
        std::vector<Value* > x; // joint token and position embedding
        for (int j=0;j<state_dict.wte.cols;j++){
            x.push_back(add(state_dict.wte(token_id, j), state_dict.wpe(pos_id, j)));
        }
        x = rmsnorm(x);
        for (int li=0; li<N_LAYER;li++){
            // 1. Multi-Head attention
            std::vector<Value*> x_residual = x;
            x = rmsnorm(x);

            std::vector<Value*> q = linear(x, state_dict.layers[li].attn_wq);
            std::vector<Value*> k = linear(x, state_dict.layers[li].attn_wk);
            std::vector<Value*> v = linear(x, state_dict.layers[li].attn_wv);
            keys[li].push_back(k);
            values[li].push_back(v);

            std::vector<Value*> x_attn;
            for(int h=0;h<N_HEAD;h++){
                int hs = h*HEAD_DIM;
                std::vector<Value*> q_h(q.begin() + hs, q.begin() + hs + HEAD_DIM);
                std::vector<std::vector<Value*>> k_h, v_h;
                for (int t = 0; t < keys[li].size(); t++) {
                    k_h.emplace_back(keys[li][t].begin() + hs, keys[li][t].begin() + hs + HEAD_DIM);
                    v_h.emplace_back(values[li][t].begin() + hs, values[li][t].begin() + hs + HEAD_DIM);
                }
                std::vector<Value*> attn_logits;
                for (int t=0;t<k_h.size();t++){
                    Value* sum = mul(q_h[0],k_h[t][0]);
                    for (int j=1;j<HEAD_DIM;j++){
                        sum = add(sum, mul(q_h[j],k_h[t][j]));
                    }
                    attn_logits.push_back(div(sum, make_value(SQRT_HEAD_DIM)));
                }
                std::vector<Value*> attn_weights = softmax(attn_logits);
                std::vector<Value*> head_out;
                for (int j=0;j<HEAD_DIM;j++){
                    Value* sum = mul(attn_weights[0],v_h[0][j]);
                    for (int t=1;t<v_h.size();t++){
                        sum = add(sum,mul(attn_weights[t],v_h[t][j]));
                    }
                    head_out.push_back(sum);
                }
                x_attn.insert(x_attn.end(), head_out.begin(), head_out.end()); //extend
            }

            x = linear(x_attn, state_dict.layers[li].attn_wo);
            for (int i = 0; i < x.size(); i++) {
                x[i] = add(x[i], x_residual[i]);
            }
            // 2. MLP block
            x_residual = x;
            x = rmsnorm(x);
            x = linear(x, state_dict.layers[li].mlp_fc1);
            for (int i = 0; i < x.size(); i++) x[i] = relu(x[i]);
            x = linear(x, state_dict.layers[li].mlp_fc2);
            for (int i = 0; i < x.size(); i++) x[i] = add(x[i], x_residual[i]);
        }
        std::vector<Value*> logits = linear(x,state_dict.lm_head); 
        return logits;
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