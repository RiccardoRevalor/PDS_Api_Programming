#include "stack.h"
#include <iostream>

int main(void){
    Stack s;
    s.push(1);
    s.push(2);
    s.push(3);
    s.push(4);
    s.push(5);

    s.visit();

    int stackSize = s.getSize();
    std::cout << "Stack size: " << stackSize << std::endl;

    bool isEmpty = s.empty();
    std::cout << "Is stack empty? " << isEmpty << std::endl;
}