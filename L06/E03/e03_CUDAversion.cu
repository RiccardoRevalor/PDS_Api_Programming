#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <random>

using namespace std;

#define ARRAY_SIZE 1000000

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

__global__ void kernelFunc(float *input, float *output, int n){
    //compute index of thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        float lower;
        if (idx == 0){
            lower = input[0];
        } else {
            lower = input[idx - 1];
        }

        float upper;
        if (idx == n - 1){
            upper = input[n - 1];
        } else {
            upper = input[idx + 1];
        }


        output[idx] = (lower + input[idx] + upper) / 3.0;
    }
}



int main(void){
    float input[ARRAY_SIZE];
    float output[ARRAY_SIZE];


    //randomize values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 100.0);
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        input[i] = dis(gen);
    }

    float *d_input, *d_output; //for gpu

    //alloc device arrays and copy input from host to device via PCIe bus
    CUDA_CHECK(cudaMalloc((void **) &d_input, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_output, ARRAY_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = ARRAY_SIZE; //1 thread per element of the array
    int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize; //number of blocks needed

    auto start = std::chrono::high_resolution_clock::now();
    kernelFunc<<<gridSize, blockSize>>>(d_input, d_output, ARRAY_SIZE);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto difference = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Time taken for seding to GPU + computing on GPU: %d ms\n", difference);


    //copy from device to host
    CUDA_CHECK(cudaMemcpy(output, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    auto end1 = std::chrono::high_resolution_clock::now();
    auto difference1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - end).count();
    printf("Time taken for sending output back to host via PCIe bus: %d ms\n", difference1);


    //free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));


    //print first 10 elements of output
    for (int i = 0; i < 10; ++i) {
        printf("output[%d] = %f\n", i, output[i]);
    }

    CUDA_CHECK(cudaDeviceReset());





    return 0;
}