#include <stdlib.h>
#include <iostream>

using namespace std;

float *vet;


//prototypes
float * allocateArray(int n);
void fillArray(float *vet, int n);
void sort(float *vet, int n);


int main(void){
    int n;
    cout << "Insert n: ";
    cin >> n;

    //allocate array
    vet = allocateArray(n);

    //fill array reading from stdin
    fillArray(vet, n);

    //sort the array in increasing order
    sort(vet, n);

    //print the sorted array
    for (size_t i = 0; i < n; i++){
        cout << vet[i] << " ";
    }


    cin.get();
    cout << "\nPress a key to exit...";
    cin.get();



    return 0;
}


//allocate an array of n elements
float * allocateArray(int n){
    float *vet = (float *) malloc(n * sizeof(float));
    if (vet == nullptr){
        cout << "Memory allocation error" << endl;
        exit(1);
    }

    return vet;
}


//fill the array with n elements
void fillArray(float *vet, int n){
    for (size_t i = 0; i < n; i++){
        cout << "Insert element " << i << ": ";
        cin >> vet[i];
    }
}

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