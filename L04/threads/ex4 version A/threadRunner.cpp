#include <iostream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <thread>
#include <algorithm>
#include <mutex> //in this version just used for errors
#include <vector>
#include "convertToBinary.h"

#define DEBUG 1

using namespace std;

struct ArrayRead {
    int *numbers;
    int len; //len of numbers

};

ArrayRead readNumbers(string fileBin, mutex &errMtx, int threadId){
    ifstream input(fileBin);

    if (!input || !input.is_open()) return {nullptr, -1}; //size -1 indicates error;

    //read the first integer -> vector size
    int firstNumber;
    input.read(reinterpret_cast<char *>(&firstNumber), sizeof(firstNumber));

    #if DEBUG
    errMtx.lock();
    cout << "[T" << threadId << "]" << "Read from file " << fileBin << " firstNumber: " << firstNumber << endl;
    errMtx.unlock();
    #endif

    //check if indeed you've read a number
    if (input.gcount() != sizeof(firstNumber)){
        errMtx.lock();
        cerr << "[T" << threadId << "] Cannot read firstNumber to determine the arra size!" << endl;
        errMtx.unlock();
        input.close();
        return {nullptr, -1}; 
    }

    //alloc the array to store the other numbers
    int *numbers = new (nothrow) int[firstNumber];  //do not throw an exception, return a nullptr in case of alloc failure
    if (numbers == nullptr){
        errMtx.lock();
        cerr << "[T" << threadId << "] Cannot dinamically allocate array for file " << fileBin << endl;
        errMtx.unlock();
        input.close();
        return {nullptr, -1};
    }

    /*
    //this is a useful thing but in this case I already have the len of the array
    //populate the array by reading the other integers from the input file
    streamsize startingPos = input.tellg(); //starting pos to read
    //got to the end of file to calculate readLen
    input.seekg(0, ios_base::end);
    int readLen = static_cast<int>(input.tellg()) - static_cast<int>(startingPos);
    */
    //So I can just do:
    int readLen = firstNumber * sizeof(int);


    #if DEBUG
    errMtx.lock();
    cout << "[T" << threadId << "]" << "size of array of number is: " << readLen << " bytes" << endl;
    errMtx.unlock();
    #endif

    //numbers is an array (int *) and it's already a pointer! Passing &number is wrong
    input.read(reinterpret_cast<char *>(numbers), readLen);

    #if DEBUG
    errMtx.lock();
    cout << "[T" << threadId << "]";
    for (int i = 0; i < firstNumber; i++){
        cout << numbers[i] << " ";
    }
    cout << endl;
    errMtx.unlock();
    #endif

    //check reading of array was OK
    if (input.gcount() != readLen) {
        errMtx.lock();
        cerr << "[T" << threadId << "] Didn't read the full number of numbers, which is: " << firstNumber << endl;
        errMtx.unlock();
        input.close();
        delete[] numbers;
        return {nullptr, -1};
    }

    input.close();
    return {numbers, firstNumber};
}

void threadFunc(string fileBin, vector<ArrayRead> &sortedNumbers, int threadId, mutex &errMtx){

    //read numbers
    ArrayRead data = readNumbers(fileBin, errMtx, threadId);

    //check errors
    if (data.len <= 0){
        //dealloc
        errMtx.lock();
        cerr << "[T" << threadId << "]" << "ArrayData null, exiting..." << endl;
        errMtx.unlock();
        if (data.numbers != nullptr) delete[] data.numbers;
        return;
    }

    //sort the numbers
    sort(data.numbers, data.numbers + data.len);

    //save the array at the specific threadId
    //since here every thread is accessing a DIFFERENT memory location (i.e. sortedNumbers[0] has different mem location from sortedNumbers[1], I can simply do like this without mutexes etc...)
    sortedNumbers[threadId] = data;

    #if DEBUG
    errMtx.lock();
    cout << "[T" << threadId << "] Sorted Array of numbers:";
    for (int i = 0; i < data.len; i++){
        cout << data.numbers[i] << " ";
    }
    cout << endl;
    errMtx.unlock();
    #endif


    return;
}



int main(int argc, char **argv){

    string outputFile;
    int i;

    //argv[0] is the program name
    //argv[argc - 1] is the outputFile
    //so I have to create argc - 2 threads

    if (argc == 1){
        cerr << "Bad Args!" << endl;
        return -1;
    }

    vector<thread> workers;
    vector<ArrayRead> sortedNumbers(argc - 1); 
    //mutex for outputting errors on cerr
    mutex errMtx;

    for (i = 1; i < argc - 1; i++){
       string fileToRead = argv[i];
       workers.emplace_back(thread(threadFunc, fileToRead, ref(sortedNumbers), i, ref(errMtx)));
    }

    outputFile = argv[argc - 1];

    #if DEBUG
    errMtx.lock();
    cout << "[MAIN] Waiting for workers..." << endl;
    errMtx.unlock();
    #endif

    for (auto &worker : workers){
        worker.join();
        errMtx.lock();
        cout << "[MAIN] Waiting..." << endl;
        errMtx.unlock();
    }

    #if DEBUG
    cout << "[MAIN] Waited for the workers done." << endl;
    #endif

    //now merge all the arrays into one unique array, and store in binary in the output file
    vector<int> merged;

    for (auto &arr: sortedNumbers){
        for (int j = 0; j < arr.len; j++){
            if (arr.numbers != nullptr && arr.len > 0){ //only merge valid arrays
                merged.emplace_back(arr.numbers[j]);
            }
        }
    }

    //write to output file
    writeBinaryFile(merged, outputFile);



    //free dinamic arrays
    for (auto &arr : sortedNumbers) {
        if (arr.numbers != nullptr) {
            delete[] arr.numbers;
        }
    }



    return 0;
}