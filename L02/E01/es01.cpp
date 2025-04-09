#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>

#define MAX_STRING 30

//struct used to write into the binary file
typedef struct {
    int id;
    long registerNumber;
    char surname[MAX_STRING];
    char name[MAX_STRING];
    int examinationMark;
} Line_Binary;



void readFile_1(const char *file_1, const char *file_2){
    std::ifstream input(file_1); //input is ASCII file
    std::ofstream output(file_2, std::ios::binary); //output is binary file

    if (!input || !output){
        std::cerr << "Error opening files" << std::endl;
        return;
    }


    Line_Binary line;

    //read from file_1 and write to file_2
    while (input >> line.id >> line.registerNumber){
        //beware to not consider the spaces where parsing the strings
        input >> std::ws;
        input.getline(line.surname, MAX_STRING, ' ');;
        input.getline(line.name, MAX_STRING, ' ');
        input >> line.examinationMark;

        //save in binary, so use the syscall write
        output.write(reinterpret_cast<char*>(&line), sizeof(Line_Binary));
    }




}


void readFile_2(const char *file_2, const char *file_3){
   std::ifstream input(file_2, std::ios::binary);
   std::ofstream output(file_3);

   if (!input || !output){
    std::cerr << "Error opening files" << std::endl;
    return;
   }
   
   //use read to read from binary file
   Line_Binary line;
   while (input.read(reinterpret_cast<char*>(&line), sizeof(Line_Binary))){
       //write the struct to file3
         output << line.id << " " << line.registerNumber << " " << line.surname << " " << line.name << " " << line.examinationMark << std::endl;
   }

}



int main(int argc, char **argv){

    if (argc != 4){
        std::cerr << "Usage: " << argv[0] << " <file_1> <file_2> <file_3>" << std::endl;
        return 1;
    }

    //step1: read from text file file_1 and write to binary file file_2
    readFile_1(argv[1], argv[2]);

    //step2: read frpom binary file file_2 and write to text file file_3
    readFile_2(argv[2], argv[3]);

    return 0;
}