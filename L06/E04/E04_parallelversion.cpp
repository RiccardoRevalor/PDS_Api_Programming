#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <string>
#include <future>

using namespace std;

#define ARRAY_SIZE 1000000
#define NUM_BINS 10
const unsigned int NUM_THREADS = thread::hardware_concurrency(); //deciced by the HW

//bins
map<string, int> histogram = {
    {"0-9", 0},
    {"10-19", 0},
    {"20-29", 0},
    {"30-39", 0},
    {"40-49", 0},
    {"50-59", 0},
    {"60-69", 0},
    {"70-79", 0},
    {"80-89", 0},
    {"90-100", 0}
};

//input data
vector<int> input;


void computeLocalHistogram(int startChunk, int endChunk, promise<map<string, int>> &setLocalHistogram){
    //count how many data points fall into each bin
    //build a local histogram just for the processed chunk of points
    map<string, int> binCounts = {
        {"0-9", 0},
        {"10-19", 0},
        {"20-29", 0},
        {"30-39", 0},
        {"40-49", 0},
        {"50-59", 0},
        {"60-69", 0},
        {"70-79", 0},
        {"80-89", 0},
        {"90-100", 0}
    }; 

    for (int i = startChunk; i < endChunk; ++i){
        int inputElement = input[i];

        for (const auto &bin : binCounts){
            string range = bin.first; //take the key of the bin, which is the bin range
            int rangeStart = stoi(range.substr(0, range.find('-')));
            int rangeEnd = stoi(range.substr(range.find('-') + 1));

            if (inputElement >= rangeStart && inputElement <= rangeEnd){
                //bin found
                ++binCounts[range];
                break;
            }
        }
        
    }


    //set promise
    setLocalHistogram.set_value(binCounts);
}


int main(void){

    //randomize input data
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, 100);

    for (int i = 0; i < ARRAY_SIZE; i++){
        input.emplace_back(dis(gen));
    }

    vector<promise<map<string, int>>> promises(NUM_THREADS);
    vector<future<map<string, int>>> futures;
    vector<thread> workers;

    int chunkSize = ARRAY_SIZE / NUM_THREADS; 

    cout << "Starting threads. Number of threads: " << NUM_THREADS << endl;
    auto startTime = chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS; i++){
        futures.emplace_back(promises[i].get_future());

        int startChunk = i * chunkSize;
        //since we use a number of threads equal to thread::hardware_concurrency(), we have to handle cases of variable size chunks for the last thread and cap them at ARRAY_SIZE
        int endChunk;
        if (i == NUM_THREADS - 1){
            endChunk = ARRAY_SIZE;
        } else {
            endChunk = (i+1) * chunkSize;
        }

        workers.emplace_back(thread(computeLocalHistogram, startChunk, endChunk, ref(promises[i])));
    }

    for (int i = 0; i < NUM_THREADS; i++){
        workers[i].join();

        //get future and update global histogram
        map<string, int> localHist = futures[i].get();

        //update global histogram
        for (auto &keyvaluepair: localHist){
            histogram[keyvaluepair.first] += keyvaluepair.second;
        }

    }

    auto endTime = chrono::high_resolution_clock::now();
    auto difference = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "Time taken: " << difference << " ms" << endl;
    cout << "Global Histogram:" << endl;

    //show global histogram
    for (auto &keyvaluepair: histogram){
        cout << "[" << keyvaluepair.first + "] -> " << keyvaluepair.second << endl;
    }
    






    return 0;
}