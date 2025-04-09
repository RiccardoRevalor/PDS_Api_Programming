#include "stack.h"
#include <iostream>
#include <string>

int main(void){
    Stack s;
    s.push("Hello");
    s.push("World");
    s.push("This");
    s.push("Is");
    s.push("A");
    s.push("Stack");


    s.visit();

    s.pop();

    s.visit();

    int stackSize = s.getSize();
    std::cout << "Stack size: " << stackSize << std::endl;

    bool isEmpty = s.empty();
    std::cout << "Is stack empty? " << isEmpty << std::endl;
}