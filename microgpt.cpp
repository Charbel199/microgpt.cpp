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
const data_T SQRT_HEAD_DIM = std::sqrt(HEAD_DIM);

struct Value {
    data_T data;
    mutable grad_T grad = 0;
    int num_children = 0;
    const Value* _children[2] = {}; // we only support noop, unary op and binary op 
    grad_T _local_grads[2] = {};

    Value(data_T data) : data(data) {}
    Value() : data(data_T{0}) {}
};

std::vector<Value> arena; // memory management for Values in a pass

void backward(Value* loss){
    loss->grad = 1;
    for (auto it = arena.rbegin(); it != arena.rend(); ++it) {
        for(int i = 0; i< it->num_children;i++){
            it->_children[i]->grad += it->_local_grads[i] * it->grad;
        }
    }
}

Value* make_value(data_T data, const Value* c1, grad_T g1, const Value* c2, grad_T g2) { // binary ops middleware
    arena.emplace_back(data);
    Value* v = &arena.back();
    v->num_children = 2;
    v->_children[0] = c1; v->_children[1] = c2;
    v->_local_grads[0] = g1; v->_local_grads[1] = g2;
    return v;
}
Value* make_value(data_T data, const Value* c1, grad_T g1) { // unary ops middleware
    arena.emplace_back(data);
    Value* v = &arena.back();
    v->num_children = 1;
    v->_children[0] = c1;
    v->_local_grads[0] = g1;
    return v;
}
Value* make_value(data_T data) { // noop middlware (constant, no children)
    arena.emplace_back(data);
    return &arena.back();
}

// operations
Value* add(Value* a, Value* b) {
    return make_value(a->data + b->data, a, 1.0, b, 1.0);
}
Value* mul(Value* a, Value* b) {
    return make_value(a->data * b->data, a, static_cast<grad_T>(b->data), b, static_cast<grad_T>(a->data));
}
Value* pow(Value* a, data_T other) {
    return make_value(std::pow(a->data, other), a, other*std::pow(a->data,(other - 1)));
}
Value* log(Value* a) {
    return make_value(std::log(a->data),a, 1/a->data);
}
Value* exp(Value* a) {
    return make_value(std::exp(a->data), a, std::exp(a->data));
}
Value* relu(Value* a) {
    return make_value(std::max(data_T{0}, a->data), a, a->data>0?data_T{1}:data_T{0});
}
Value* div(Value* a, Value* b) {
    return make_value(a->data / b->data, a, 1.0/b->data, b, -a->data/(b->data*b->data));
}
Value* neg(Value* a) {
    return make_value(-a->data, a, -1.0);
}
Value* sub(Value* a, Value* b) {
    return make_value(a->data - b->data, a, 1.0, b, -1.0);
}

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