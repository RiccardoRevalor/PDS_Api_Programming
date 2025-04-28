#ifndef CONVERT_TO_BINARY
#define CONVERT_TO_BINARY

#include <string>
#include <vector>

void writeBinaryFile(const std::string &ASCII_filename, const std::string &binary_filename);
void writeBinaryFile(std::vector<int> &inputData, const std::string &binary_filename);

#endif