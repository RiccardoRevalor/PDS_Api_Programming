#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

#define ARRAY_SIZE 1000000
vector<float> input(ARRAY_SIZE);
vector<float> output(ARRAY_SIZE);


void threadFunc(int startChunk, int endChunk){
    for (int i = startChunk; i < endChunk; i++){
        float lower;
        if (i == 0) {
            lower = input[i];
        } else {
            lower = input[i - 1];
        }

        float upper;
        if (i == ARRAY_SIZE - 1) {
            upper = input[i];
        } else {
            upper = input[i + 1];
        }
        output[i] = (lower + input[i] + upper) / 3.0;
    }
}


int main(void){
    // Initialize the input array with random values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 100.0);
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        input[i] = dis(gen);
    }

    int sizeChunks = 100; //size of each chunk in terms of elements
    int numThreads = ARRAY_SIZE / sizeChunks;

    vector<thread> workers;
    auto startTime = chrono::high_resolution_clock::now();
    for (int i = 0; i < numThreads; i++){
        workers.emplace_back(threadFunc, i * sizeChunks, i * sizeChunks + sizeChunks);
    }


    for (auto &worker: workers){
        worker.join();
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto difference = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "Time taken: " << difference << " ms" << endl;

    //Output the first 10 results
    for (int i = 0; i < 10; ++i) {
        cout << "output[" << i << "] = " << output[i] << endl;
    }
    cout << "..." << endl;


    return 0;
}
