#ifndef QUEUE_HPP
#define QUEUE_HPP
#include <iostream>
#include <list>
#include <string>
#include <cstddef> //for size_t

using namespace std;

template <typename T>
class Queue {
    private:
        //use a list to make a LIFO
        list<T> l;
        size_t size;

    public:
        Queue();
        void push(T a);
        T pop();
        void visit();
        int getSize();
        bool empty();

};


template <typename T>
Queue<T>::Queue(){
    this->size = 0;
}

template <typename T>
void Queue<T>::push(T a){
   //add a to the end of the vector
   //and increase the size
   this->l.push_back(a);
   this->size++;
}

template <typename T>
T Queue<T>::pop(){
    //pop the first element of the stack
    //and decrease the size
    if (this->size > 0){
        T res = this->l.front(); //get the first element
        this->l.pop_front(); //use pop_front to pop the first element since it is a FIFO
        this->size--;

        return res;
    }
    else{
        cerr << "Stack is empty" << std::endl;

        return T(); //return default value of T
    }
}


template <typename T>
void Queue<T>::visit(){
    for (T v: this->l){
        cout << v << " ";
    }
    cout << std::endl;
}



template <typename T>
int Queue<T>::getSize(){
    return this->size;
}

template <typename T>
bool Queue<T>::empty(){
    return this->size == 0;
}

#endif
