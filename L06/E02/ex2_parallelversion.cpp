#include <vector>
#include <future>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
using namespace std;

#define VECTOR_SIZE 1000000
vector<double> input(VECTOR_SIZE);
vector<double> output(VECTOR_SIZE);

void threadFunc(int idx){
    output[idx] = sin(input[idx]) + cos(input[idx]);
}


int main(void){
    //randomize the input vector
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 100.0);
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        input[i] = dis(gen);
    }

    //start a bunch of tasks
    vector<future<void>> futures;
    cout << "Starting tasks..." << endl;
    auto startTime = chrono::high_resolution_clock::now();
    for (int i = 0; i <VECTOR_SIZE; i++){
        futures.emplace_back(async(launch::async, threadFunc, i)); 
    }

    //wait for all tasks to finish
    for (auto& fut : futures) {
        fut.get(); //wait for each task to finish
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();


    cout << "Time passed: " << duration << " ms" << endl;
    
    for (int i = 0; i < 10; i++) {
        cout << "output[" << i << "] = " << output[i] << endl; //print first 10 results
    }

    return 0;


}