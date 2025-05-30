#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <random>
#include <stdio.h>
#define MAX_LEN 10
#define THREAD_CHUNK_SIZE 10 //elements processed per thread
#define ARRAY_SIZE 1000000
#define NUM_BINS 10

using namespace std;


#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

typedef struct bin_s{
    char key[MAX_LEN];
    int value;
} bin_t;

__device__ int extractIntegerFromString(const char *str, int start, int len){
    //Device (aka GPU) function to extract the start and end of bins
    //extract the integer from start to len
    int value = 0;
    for (int i = start; i < len; i++){
        //check if character at position i is a digit
        if (str[i] >= '0' && str[i] <= '9'){
            value = value * 10 + (str[i] - '0');
        } else {
            break; //no need to continue if we encounter the first non-digit character
        }
    }

    return value;
}

__global__ void computeLocalHistogram(bin_t *globalHistogram, int *input, int numBins, int chunkSize, int arraySize){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int startChunk = idx * chunkSize;
    //handle case where startChunk goes after arraySize
    if (startChunk >= arraySize) return;

    int endChunk = (idx + 1) * chunkSize;
    //handle case where endChunk goes after arraySize
    if (endChunk > arraySize){
        endChunk = arraySize; //probably used for last feasible chunk to be processed
    } 

    for (int i = startChunk; i < endChunk; i++){
        int inputElement = input[i];

        for (int kbin = 0; kbin < numBins; kbin++){
            //check if element is inside key interval
            //search dash position
            const char *binKey = globalHistogram[kbin].key;

            int dashIndex = -1;
            for (int j = 0; j < MAX_LEN; j++){
                if (binKey[j] == '-'){
                    dashIndex = j;
                    break;
                }
            }

            if (dashIndex > 0){
                int startBin = extractIntegerFromString(binKey, 0, dashIndex);
                int endBin = extractIntegerFromString(binKey, dashIndex + 1, MAX_LEN);

                //check if input elements falls into the bin range
                if (inputElement >= startBin && inputElement <= endBin){
                    //atomically update the globalHistogram
                    atomicAdd(&(globalHistogram[kbin].value), 1); //globalHistogram[kbin].value++ but atomically
                    break; //go to next element
                }
            }
        }
    }

}


int main(void){

    //host input and global Histogram
    int input[ARRAY_SIZE];
    //randomize input
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, 100);
    for (int i = 0; i < ARRAY_SIZE; i++){
        input[i] = dis(gen);
    }
    bin_t globalHistogram[NUM_BINS] = {
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


    //copy to GPU
    int *d_input;
    bin_t *d_globalHistogram;
    auto startTime = chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc((void **) &d_input, ARRAY_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &d_globalHistogram, NUM_BINS * sizeof(bin_t)));

    CUDA_CHECK(cudaMemcpy(d_input, input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_globalHistogram, globalHistogram, NUM_BINS * sizeof(bin_t), cudaMemcpyHostToDevice));

    int blockSize = 256; //threads per block
    
    //total number of chunks, considering the chunkSize and total array size
    int numChunks = (ARRAY_SIZE + THREAD_CHUNK_SIZE - 1) / THREAD_CHUNK_SIZE;
    
    int gridSize = (numChunks + blockSize - 1) / blockSize;


    computeLocalHistogram<<<gridSize, blockSize>>>(d_globalHistogram, d_input, NUM_BINS, THREAD_CHUNK_SIZE, ARRAY_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime0 = chrono::high_resolution_clock::now();
    auto difference = chrono::duration_cast<chrono::milliseconds>(endTime0 - startTime).count();
    printf("Time taken for sending data to GPU and computing on GPU: %d ms\n", difference);
    
    //copy back to CPU
    CUDA_CHECK(cudaMemcpy(globalHistogram, d_globalHistogram, NUM_BINS * sizeof(bin_t), cudaMemcpyDeviceToHost));

    auto endTime1 = chrono::high_resolution_clock::now();
    difference = chrono::duration_cast<chrono::milliseconds>(endTime1 - endTime0).count();
    printf("Time taken for sending data back from GPU to CPU on PCIe bus: %d ms\n", difference);

    //free GPU arrays
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_globalHistogram));

    CUDA_CHECK(cudaDeviceReset());

    //print global histogram
    printf("Global Histogram:\n");
    for (int i = 0; i < NUM_BINS; i++){
        printf("%s -> %d\n", globalHistogram[i].key, globalHistogram[i].value);
    }


    return 0;
}