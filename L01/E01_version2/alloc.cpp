#include <stdlib.h>
#include <iostream>


//allocate an array of n elements
float * allocateArray(int n){
    float *vet = (float *) malloc(n * sizeof(float));
    if (vet == nullptr){
        return nullptr;
    }

    return vet;
}