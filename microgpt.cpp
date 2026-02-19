#include <iostream>
#include <vector>

template <typename data_T, typename grad_T>
struct Value {

    data_T data;
    grad_T grad = 0;
    std::vector<Value*> _children;
    std::vector<grad_T> _local_grads;

    Value(
        data_T data, 
        std::vector<Value*> children, 
        std::vector<grad_T> local_grads):
        data(data),
        grad(0),
        _children(std::move(children)),
        _local_grads(std::move(local_grads))
    {}

    Value operator+(const Value& other) const {
        return Value(this->data + other.data, {this, &other}, {1,1});
    }
    Value operator*(const Value& other) const {
        return Value(this->data * other.data, {this, &other}, {static_cast<grad_T>(other.data), static_cast<grad_T>(this->data)};
    }


};


int main() {
    std::cout << "Hello, MicroGPT!" << std::endl;
    return 0;
}