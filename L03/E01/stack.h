#ifndef STACK_H
#define STACK_H

#include <vector>
#include <cstddef> //for size_t

template <typename T> //define a template class to use the stack with any type
class Stack {
    private:
        std::vector<T> vec;
        size_t size;

    public:
        Stack();
        void push(T a);
        T pop();
        void visit();
        int getSize();
        bool empty();

};

#endif