#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "convertToBinary.h"

using namespace std;


void writeBinaryFile(const string &ASCII_filename, const string &binary_filename){
    ifstream input(ASCII_filename); //inpit file
    ofstream output(binary_filename, ios_base::binary); //output in binary

    if (!input || !output){
        cerr << "Error opening files" << endl;
        return;
    }

    if (!input.is_open() || !output.is_open()){
        cerr << "Error opening files" << endl;
        return;
    }


    int number;

    while (input >> number){
        //use syscall write to write in binary, convert to a byte, so a char
        output.write(reinterpret_cast<char*>(&number), sizeof(number));
    }

    //close files
    input.close(); output.close();
}


void writeBinaryFile(vector<int> &inputData,  const string &binary_filename){
    ofstream output(binary_filename); //output in binary: ios_base::binary disabled for testing purposes

    if (!output){
        cerr << "Error opening files" << endl;
        return;
    }

    for (auto &number : inputData){
        output << number << " ";    //ASCII output for testing purposes
        //output.write(reinterpret_cast<char*>(&number), sizeof(number)); //for binary
    }

    output.close();
}

/*
not needed anymore, input files already converted to binary
int main(int argc, char **argv){
    if (argc != 3){
        cerr << "Bad args!" << endl;
    }

    string fileIn = argv[1];
    string fileOut = argv[2];

    writeBinaryFile(fileIn, fileOut);

}*/