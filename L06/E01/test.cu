// test.cu
#include <iostream>
__global__ void testKernel() {
    printf("Hello from kernel\n");
}
int main() {
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
