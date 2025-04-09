#ifndef STACK_HPP
#define STACK_HPP
//#include "stack.h" -> do not include the header file here to avoid circular dependency
#include <iostream>
#include <vector>
#include <cstddef> //for size_t

template <typename T> //define a template class to use the stack with any type
Stack<T>::Stack(){
    this->size = 0;
}

template <typename T>
void Stack<T>::push(T a){
   //add a to the end of the vector
   //and increase the size
   this->vec.push_back(a);
   this->size++;
}

template <typename T>
T Stack<T>::pop(){
    //pop the last element of the stack
    //and decrease the size
    if (this->size > 0){
        T res = this->vec.back(); //get the last element
        this->vec.pop_back();
        this->size--;

        return res;
    }
    else{
        std::cerr << "Stack is empty" << std::endl;

        return T(); //return default value of T
    }
}

template <typename T>
void Stack<T>::visit(){
    for (T  v: this->vec){
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

template <typename T>
int Stack<T>::getSize(){
    return this->size;
}

template <typename T>
bool Stack<T>::empty(){
    return this->size == 0;
}

#endif
