#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

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

#define BLOCK_SIZE 256
#define ARRAY_SIZE 16777216

#define USE_WINDOWS 1

typedef struct timeval tval;

/**
 * Helper method to generate a very naive "hash".
 */
float generate_hash(int n, float *y)
{
    float hash = 0.0f;
    
    for (int i = 0; i < n; i++)
    {
        hash += y[i];
    }
    
    return hash;
}

#if USE_WINDOWS
/*
* get_elapsed, Windows version
*/
double get_elapsed(std::chrono::high_resolution_clock::time_point t0, std::chrono::high_resolution_clock::time_point t1)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}
#else
/**
 * Helper method that calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}
#endif


__global__ void addMathFunction(float *input, float *output, int n){
    //compute index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        output[idx] =  sin(input[idx]) + cos(input[idx]);
    }

}

int main(int argc, char **argv)
{
    float a     = 0.0f;
    float *x    = NULL;
    float *y    = NULL;
    float error = 0.0f;
    
    ///////////
    // TO-DO ////////////////////////////////////
    // Introduce the grid and block definition //
    /////////////////////////////////////////////
    int grid_size = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE; //grid size = number of blocks in the grid
    
    ///////////
    // TO-DO ////////////////////////////////////
    // Declare the device pointers d_x and d_y //
    /////////////////////////////////////////////
    //C like declarations
    float *d_x, *d_y;
    
    // Make sure the constant is provided
    
    // Retrieve the constant and allocate the arrays on the CPU
    a = 2.0;
    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    
    // Initialize them with fixed values
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = 0.1f;
        y[i] = 0.0f;
    }
    
    ///////////
    // TO-DO ///////////////////////////////////////////////////////////////
    // Allocate d_x and d_y on the GPU, and copy the content from the CPU //
    ////////////////////////////////////////////////////////////////////////
    //first allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void **) &d_x, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_y, ARRAY_SIZE * sizeof(float)));
    //then copy the content from the CPU to the GPU using the PCIe bus
    CUDA_CHECK(cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    addMathFunction<<<grid_size, BLOCK_SIZE>>>(d_x, d_y, ARRAY_SIZE);
    
    printf("Executing the GPU code...\n");
    auto tgpu_start = std::chrono::high_resolution_clock::now();

    //make cpu wait for the GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    auto tgpu_end = std::chrono::high_resolution_clock::now();
    printf("GPU execution time: %.3f ms\n", get_elapsed(tgpu_start, tgpu_end));

    //copy content from GPU to CPU
    CUDA_CHECK(cudaMemcpy(y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    //print first 10 values
    for (int i = 0; i < 10; i++){
        printf("%f ", d_y[i]);
    }

    
    // Release all the allocations
    free(x);
    free(y);
    
    ////////////
    // TO-DO ////////////////
    // Release d_x and d_y //
    /////////////////////////
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Reset the device
    CUDA_CHECK(cudaDeviceReset());
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

