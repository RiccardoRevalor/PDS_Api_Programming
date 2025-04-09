#include <stdlib.h>
#include <iostream>

using namespace std;

//sort the array in increasing order
void sort(float *vet, int n){
    //bubble sort algortithm is used
    for (int i = 0; i < n; i++){
        for (int j = i+1; j < n; j++){
            if (vet[i] > vet[j]){
                float temp = vet[i];
                vet[i] = vet[j];
                vet[j] = temp;
            }
        }
    }
}