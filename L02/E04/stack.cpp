#include "stack.h"
#include <iostream>
#include <list>
#include <string>


Stack::Stack(){
    this->size = 0;
}

void Stack::push(std::string a){
   //add a to the end of the vector
   //and increase the size
   this->l.push_back(a);
   this->size++;
}

void Stack::pop(){
    //pop the first element of the stack
    //and decrease the size
    if (this->size > 0){
        this->l.pop_front(); //use pop_front to pop the first element since it is a FIFO
        this->size--;
    }
    else{
        std::cerr << "Stack is empty" << std::endl;
    }
}

void Stack::visit(){
    for (std::string v: this->l){
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

int Stack::getSize(){
    return this->size;
}

bool Stack::empty(){
    return this->size == 0;
}
