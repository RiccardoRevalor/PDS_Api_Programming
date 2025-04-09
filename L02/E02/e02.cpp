#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>

#define MAX_STRING 10

//struct used to write into the binary file
typedef struct {
    int id;
    long registerNumber;
    char surname[MAX_STRING];
    char name[MAX_STRING];
    int examinationMark;
} Line_Binary;


void readBinaryFile(const char *filePath, const int pos){
    std::ifstream input(filePath, std::ios::binary);

    if (!input){
        std::cerr << "Error opening file" << std::endl;
        return;
    }


    Line_Binary line;



    
    //set pointer at pos-1
    input.seekg(sizeof(Line_Binary) * (pos -1), std::ios::beg);
    input.read(reinterpret_cast<char*>(&line), sizeof(Line_Binary));
    //print line read
    std::cout << line.id << " " << line.registerNumber << " " << line.surname << " " << line.name << " " << line.examinationMark << std::endl;
    



    /*
    while (input >> line.id >> line.registerNumber){
        //beware to not consider the spaces where parsing the strings
        input >> std::ws;
        input.getline(line.surname, MAX_STRING, ' ');;
        input.getline(line.name, MAX_STRING, ' ');
        input >> line.examinationMark;

        if (pos == line.id){
            std::cout << line.id << " " << line.registerNumber << " " << line.surname << " " << line.name << " " << line.examinationMark << std::endl;
        }
    }
        */
    
}


void writeBinaryFile(const char *filePath, const int pos, const Line_Binary lineToWrite){
    //write lineToWrite in filePath at position n
    std::ofstream output(filePath, std::ios::binary);

    if (!output){
        std::cerr << "Error opening file" << std::endl;
        return;
    }


    Line_Binary line;


    //set pointer at position pos -1
    output.seekp(sizeof(Line_Binary) * (pos -1), std::ios::beg);
    output.write(reinterpret_cast<char*>(&line), sizeof(Line_Binary));


}


int main(int argc, char **argv){
    if (argc != 2){
        std::cerr << "Error, invalid args" << std::endl;
        return 1;
    }

    char cmd[2];
    int n;

    while (1){
        std::cout << "Insert cmd: ";
        std::cin >> cmd;
        if (strcmp(cmd, "E") == 0) break;
        std::cin >> n;
        std::cout << cmd << std::endl;

        if (strcmp(cmd, "R") == 0){
            readBinaryFile(argv[1], n);
        } else if (strcmp(cmd, "W") == 0){
            Line_Binary line;
            std::cin >> line.id >> line.registerNumber >> line.surname >> line.name >> line.examinationMark;
            writeBinaryFile(argv[1], n, line);
        } else {
            std::cerr << "Error, invalid cmd" << std::endl;
        }
    }
}