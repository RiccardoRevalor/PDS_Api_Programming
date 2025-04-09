#include "stack.h"
#include "stack.hpp"
#include "queue.hpp"
#include <iostream>
#include <string>

using namespace std;

int main(void){
    Stack<int> s; //create a stack of integers

    //use an integer stack
    s.push(1);
    s.push(2);
    s.push(3);
    s.push(4);
    s.push(5);

    s.visit();

    int stackSize = s.getSize();
    cout << "Stack size: " << stackSize << std::endl;


    int poppedValue = s.pop();
    cout << "Popped value: " << poppedValue << std::endl;

    bool isEmpty = s.empty();
    cout << "Is stack empty? " << isEmpty << std::endl;



    //use a string stack
    Stack<string> s1; //create a stack of strings

    s1.push("Hello");
    s1.push("World");

    s1.visit();


    stackSize = s1.getSize();
    cout << "Stack size: " << stackSize << std::endl;


    string poppedValue1 = s1.pop();
    cout << "Popped value: " << poppedValue1 << std::endl;

    isEmpty = s1.empty();
    cout << "Is stack empty? " << isEmpty << std::endl;


    //use a int queue
    Queue<int> q;

    q.push(1);
    q.push(2);
    q.push(3);

    q.visit();

    int queueSize = q.getSize();
    cout << "Queue size: " << queueSize << std::endl;

    int poppedValue2 = q.pop();
    cout << "Popped value: " << poppedValue2 << std::endl;

    isEmpty = q.empty();
    cout << "Is queue empty? " << isEmpty << std::endl;


    //use a string queue
    Queue<string> q1;

    q1.push("Hello");
    q1.push("World");


    q1.visit();

    queueSize = q1.getSize();
    cout << "Queue size: " << queueSize << std::endl;

    string poppedValue3 = q1.pop();
    cout << "Popped value: " << poppedValue3 << std::endl;


    isEmpty = q1.empty();
    cout << "Is queue empty? " << isEmpty << std::endl;



    return 0;
}