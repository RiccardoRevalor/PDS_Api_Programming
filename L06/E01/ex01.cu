#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

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


/**
 * SAXPY reference implementation using the CPU.
 */
void cpu_saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

///////////
// TO-DO //////////////////////////////////////////////////////////////////
// Declare the kernel gpu_saxpy() with the same interface as cpu_saxpy() //
///////////////////////////////////////////////////////////////////////////
__global__ void gpu_saxpy(int n, float a, float *x, float *y){
    //compute index of thread, this corresponds to the index of the element in the array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //check idx is feasible
    if (idx < n){
        //perform the operation
        y[idx] = a * x[idx] + y[idx];
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
    if (argc != 2)
    {
        fprintf(stderr, "Error: The constant is missing!\n");
        return -1;
    }
    
    // Retrieve the constant and allocate the arrays on the CPU
    a = atof(argv[1]);
    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    
    // Initialize them with fixed values
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = 0.1f;
        y[i] = 0.2f;
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
    
    // Call the CPU code
    printf("Executing the CPU code...\n");
    //declare timestamps related to the CPU execution
    auto tcpu_start = std::chrono::high_resolution_clock::now();
    //CPU execution of the SAXPY operation
    cpu_saxpy(ARRAY_SIZE, a, x, y);
    
    auto tcpu_end = std::chrono::high_resolution_clock::now();
    //print the elapsed time
    printf("CPU execution time: %.3f ms\n", get_elapsed(tcpu_start, tcpu_end));
    
    // Calculate the "hash" of the result from the CPU
    error = generate_hash(ARRAY_SIZE, y);
    
    ///////////
    // TO-DO /////////////////////////////////////////////
    // Call the GPU kernel gpu_saxpy() with d_x and d_y //
    //////////////////////////////////////////////////////
    gpu_saxpy<<<grid_size, BLOCK_SIZE>>>(ARRAY_SIZE, a, d_x, d_y);
    
    ///////////
    // TO-DO ///////////////////////////////////////////////////////////
    // Copy the content of d_y from the GPU to the array y on the CPU //
    ////////////////////////////////////////////////////////////////////
    printf("Executing the GPU code...\n");
    auto tgpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    //make cpu wait for the GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    auto tgpu_end = std::chrono::high_resolution_clock::now();
    printf("GPU execution time: %.3f ms\n", get_elapsed(tgpu_start, tgpu_end));

    
    // Calculate the "hash" of the result from the GPU
    error = fabsf(error - generate_hash(ARRAY_SIZE, y));
    
    // Confirm that the execution has finished
    printf("Execution finished (error=%.6f).\n", error);
    
    if (error > 0.0001f)
    {
        fprintf(stderr, "Error: The solution is incorrect!\n");
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

