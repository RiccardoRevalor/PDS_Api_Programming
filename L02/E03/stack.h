#ifndef STACK_H
#define STACK_H

#include <vector>

class Stack {
    private:
        std::vector<int> vec;
        int size;

    public:
        Stack();
        void push(int a);
        void pop();
        void visit();
        int getSize();
        bool empty();

};

#endif