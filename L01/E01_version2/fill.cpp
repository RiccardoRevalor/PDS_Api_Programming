#include <stdlib.h>
#include <iostream>

using namespace std;


//fill the array with n elements
void fillArray(float *vet, int n){
    for (int i = 0; i < n; i++){
        cout << "Insert element " << i << ": ";
        cin >> vet[i];
    }
}