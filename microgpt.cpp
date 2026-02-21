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
#define LOG(msg) std::cerr << "[LOG] " << msg << std::endl

// defining our model params
constexpr int N_LAYER = 1;
constexpr int N_EMBD = 16;
constexpr int BLOCK_SIZE = 4;
constexpr int N_HEAD = 4;
constexpr int HEAD_DIM = N_EMBD / N_HEAD;

std::mt19937 rng(42); //random seed
using data_T = double;
using grad_T = double;
struct Value {

    data_T data;
    mutable grad_T grad = 0;

    // TODO: for our operations we can only have a max of 2 children,
    // might be worth using regular arrays `const Value* _children[2] = {};` for better performance 
    std::vector<const Value*> _children;
    std::vector<grad_T> _local_grads;

    Value(
        data_T data, 
        std::vector<const Value*> children, 
        std::vector<grad_T> local_grads):
        data(data),
        grad(0),
        _children(std::move(children)),
        _local_grads(std::move(local_grads))
    {}

    // to support scenarios like 1.0+Val()
    Value(
        data_T data):
        data(data)
    {}

    Value() : data(data_T{0}) {}

    Value operator+(const Value& other) const {
        return Value(this->data + other.data, {this, &other}, {1,1});
    }
    Value operator*(const Value& other) const {
        return Value(this->data * other.data, {this, &other}, {static_cast<grad_T>(other.data), static_cast<grad_T>(this->data)});
    }
    Value pow(data_T other) const {
        return Value(std::pow(this->data, other), {this}, {other*std::pow(this->data,(other - 1))});
    }
    Value log() const {
        return Value(std::log(this->data), {this}, {1/this->data});
    }
    Value exp() const {
        return Value(std::exp(this->data), {this}, {std::exp(this->data)});
    }
    Value relu() const {
        return Value(std::max(data_T{0}, this->data), {this}, {this->data>0?data_T{1}:data_T{0}});
    }


    Value operator-() const {
        return *this * -1;
    }
    Value operator-(const Value& other) const {
        return *this + (-other);
    }
    Value operator/(const Value& other) const {
        return *this * other.pow(-1);
    }
    Value& operator+=(const Value& other) {
        *this = *this + other;
        return *this;
    }
    bool operator<(const Value& other) const { return data < other.data; }
    bool operator>(const Value& other) const { return data > other.data; }

    /**
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
     */
    void backward(){
        std::vector<const Value*> topo = {};
        std::unordered_set<const Value*> visited = {};

        std::function<void(const Value&)>  build_topo = [&](const Value& v){
            if (!visited.count(&v)){
                visited.insert(&v);
                for (const Value* child : v._children){
                    build_topo(*child);
                }
                topo.push_back(&v);
            }
        };
        build_topo(*this);
        this->grad = 1;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const Value* v = *it;
            
            for(int i = 0; i< v->_children.size();i++){
                v->_children[i]->grad += v->_local_grads[i] * v->grad;
            }

        }
        

    }
};
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

    Value& operator()(int i, int j) { return data[i * cols + j]; }
    const Value& operator()(int i, int j) const { return data[i * cols + j]; }
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
            layers.emplace_back(n_embd); // equivalent to layers.push_back(Layer(n_embd));
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
using KVCache = std::vector<std::vector<std::vector<Value>>>;


// matrix * vector 
std::vector<Value> linear(const std::vector<Value>& x, Matrix& w){
    std::vector<Value> out;
    for(int i=0; i<w.rows;i++){
        Value sum;
        for(int j=0; j<w.cols;j++){
            sum+=w(i,j)*x[j];
        }
        out.push_back(sum);
    }
    return out;
}

std::vector<Value> softmax(const std::vector<Value>& logits){
    auto max_it = std::max_element(logits.begin(), logits.end());
    data_T max_val = max_it->data;
    std::vector<Value> exps;
    for (auto& val : logits){
        exps.push_back((val-max_val).exp());
    }
    Value total;
    for (auto& v : exps) total += v;
    std::vector<Value> out;
    for (auto& v : exps) out.push_back(v/total);
    return out;
}


std::vector<Value> rmsnorm(const std::vector<Value>& x){
    Value total;
    for (auto& xi : x) total += xi*xi;
    total = total/x.size();
    Value scale = (total+1e-5).pow(-0.5);
    std::vector<Value> out;
    for (auto& xi : x) out.push_back(xi * scale);
    return out;
}


std::vector<Value> gpt(
    const int token_id, 
    const int pos_id,
    KVCache& keys,
    KVCache& values,
    Model& state_dict){
        std::vector<Value> x; // joint token and position embedding
        for (int j=0;j<state_dict.wte.cols;j++){
            x.push_back(state_dict.wte(token_id, j)+state_dict.wpe(pos_id, j));
        }
        x = rmsnorm(x);

        for (int li=0; li<N_LAYER;li++){
            // 1. Multi-Head attention
            std::vector<Value> x_residual = x;
            x = rmsnorm(x);
            std::vector<Value> q = linear(x, state_dict.layers[li].attn_wq);
            std::vector<Value> k = linear(x, state_dict.layers[li].attn_wk);
            std::vector<Value> v = linear(x, state_dict.layers[li].attn_wv);
            keys[li].push_back(k);
            values[li].push_back(v);
            

            std::vector<Value> x_attn;
            for(int h=0;h<N_HEAD;h++){
                int hs = h*HEAD_DIM;
                std::vector<Value> q_h(q.begin() + hs, q.begin() + hs + HEAD_DIM);
                std::vector<std::vector<Value>> k_h, v_h;
                for (int t = 0; t < keys[li].size(); t++) {
                    k_h.emplace_back(keys[li][t].begin() + hs, keys[li][t].begin() + hs + HEAD_DIM);
                    v_h.emplace_back(values[li][t].begin() + hs, values[li][t].begin() + hs + HEAD_DIM);
                }
                std::vector<Value> attn_logits;
                for (int t=0;t<k_h.size();t++){
                    Value sum;
                    for (int j=0;j<HEAD_DIM;j++){
                        sum+=q_h[j]*k_h[t][j];
                    }
                    attn_logits.push_back(sum/std::pow(HEAD_DIM,0.5));
                }
                
                std::vector<Value> attn_weights = softmax(attn_logits);
                
                std::vector<Value> head_out;
                for (int j=0;j<HEAD_DIM;j++){
                    Value sum;
                    for (int t=0;t<v_h.size();t++){
                        sum+=attn_weights[t]*v_h[t][j];
                    }
                    head_out.push_back(sum);
                }

                x_attn.insert(x_attn.end(), head_out.begin(), head_out.end()); //extend
            }

            x = linear(x_attn, state_dict.layers[li].attn_wo);
            
            for (int i = 0; i < x.size(); i++) {
                x[i] = x[i] + x_residual[i];
            }
            // 2. MLP block
            x_residual = x;
            x = rmsnorm(x);
            x = linear(x, state_dict.layers[li].mlp_fc1);
            for (int i = 0; i < x.size(); i++) {
                x[i] = x[i].relu();
            }
            x = linear(x, state_dict.layers[li].mlp_fc2);
                        for (int i = 0; i < x.size(); i++) {
                x[i] = x[i] + x_residual[i];
            }
        }
        std::vector<Value> logits = linear(x,state_dict.lm_head); 
        return logits;
}



int main() {
    std::cout << "Hello, MicroGPT!" << std::endl;

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
        std::vector<Value> losses{};

        for (int pos_id=0; pos_id<n;pos_id++){
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id+1];
            std::vector<Value> logits = gpt(token_id, pos_id, keys, values, state_dict);
            std::vector<Value> probs = softmax(logits);
            Value loss_t = -probs[target_id].log();
            losses.push_back(loss_t);
        }
        Value total_losses{};
        for (auto& x : losses) total_losses += x;
        Value loss = total_losses * (1.0 / n) ;

        // backward pass
        loss.backward();

        // adam optimizer
        float lr_t = learning_rate*(1-(double)step/num_steps);
        for (int i = 0;i<params.size();i++){
            Value* p = params[i];
            m[i] = beta1 * m[i] + (1 - beta1) * p->grad;
            v[i] = beta2 * v[i] + (1 - beta2) * std::pow(p->grad, 2);
            grad_T m_hat = m[i] / (1 - std::pow(beta1,(step + 1)));
            grad_T v_hat = v[i] / (1 - std::pow(beta2,(step + 1)));
            p->data -= lr_t*m_hat / (std::pow(v_hat,0.5)+eps_adam);
            p->grad = 0;
        }
        LOG("Step "<<(step+1)<<" / "<<num_steps<<" | loss "<< loss.data);

    }

    float temperature = 0.5;

    LOG("\n\nTime for inference---------------");
    std::vector<char> idx_to_char(uchars.begin(), uchars.end());
    for (int sample_idx = 0; sample_idx<20;sample_idx++){
        KVCache keys(N_LAYER), values(N_LAYER);
        int token_id = BOS;
        std::vector<char> samples;

        for (int pos_id = 0;pos_id<BLOCK_SIZE;pos_id++){
            std::vector<Value> logits = gpt(token_id, pos_id, keys, values, state_dict);
            for (int i = 0; i < logits.size(); i++)
                logits[i] = logits[i] / Value(temperature);

            std::vector<Value> probs = softmax(logits);
            
            std::vector<double> weights;
            for (auto& p : probs) weights.push_back(p.data);
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            token_id = dist(rng);
            if (token_id == BOS){
                break;
            }
            samples.push_back(idx_to_char[token_id]);
        }
        
        std::string result(samples.begin(), samples.end());
        LOG("Sample: "<< sample_idx<<": "<<result);
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    Value a(2.0);
    Value b(3.0);

    Value c = a*b;
    Value d = c.pow(3);

    for (auto& v : d._local_grads) std::cout << v << " ";
    
    std::cout<<d.grad<<std::endl;

    d.backward();


    std::cout<<a.grad<<std::endl;
    std::cout<<b.grad<<std::endl;
    std::cout<<c.grad<<std::endl;
    std::cout<<d.grad<<std::endl;
    return 0;
}