#include <stdlib.h>
#include <iostream>

float * allocateArray(int n);
void fillArray(float *vet, int n);
void sort(float *vet, int n);

using namespace std;

float *vet;


int main(int argc, char *argv[]){

    /*
    int n;
    cout << "Insert n: ";
    cin >> n;
    */

    //take n from command line
    if (argc != 2){
        cout << "Usage: " << argv[0] << " n" << endl;
        exit(1);
    }

    int n = atoi(argv[1]);

    //allocate array
    vet = allocateArray(n);
    if (vet == nullptr){
        cout << "Memory allocation error" << endl;
        exit(1);
    }


    //fill array reading from stdin
    fillArray(vet, n);


    //sort the array in increasing order
    sort(vet, n);

    //print the sorted array
    for (int i = 0; i < n; i++){
        cout << vet[i] << " ";
    }


    cin.get();
    cout << "\nPress a key to exit...";
    cin.get();


    return 0;
}