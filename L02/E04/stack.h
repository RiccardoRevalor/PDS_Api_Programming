#ifndef STACK_H
#define STACK_H

#include <list>
#include <string>

class Stack {
    private:
        //use a list to make a LIFO
        std::list<std::string> l;
        int size;

    public:
        Stack();
        void push(std::string a);
        void pop();
        void visit();
        int getSize();
        bool empty();

};

#endif