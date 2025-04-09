#include "stack.h"
#include <iostream>
#include <vector>


Stack::Stack(){
    this->size = 0;
}

void Stack::push(int a){
   //add a to the end of the vector
   //and increase the size
   this->vec.push_back(a);
   this->size++;
}

void Stack::pop(){
    //pop the last element of the stack
    //and decrease the size
    if (this->size > 0){
        this->vec.pop_back();
        this->size--;
    }
    else{
        std::cerr << "Stack is empty" << std::endl;
    }
}

void Stack::visit(){
    for (int v: this->vec){
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
