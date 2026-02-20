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
    std::vector<data_T> data;
    int rows, cols;

    Matrix(int rows, int cols, float std=0.08) : data(rows * cols), rows(rows), cols(cols) {
        std::normal_distribution<data_T> dist(0.0, std);
        for (auto&v : data) v = dist(rng);
    }

    data_T& operator()(int i, int j) { return data[i * cols + j]; }
    const data_T& operator()(int i, int j) const { return data[i * cols + j]; }
};

struct Layer {
    Matrix attn_wq, attn_wk, attn_wv, attn_wo;
    Matrix mlp_fc1, mlp_fc2;
};

struct Model {
    Matrix wte, wpe, lm_head;
    std::vector<Layer> layers;
};

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