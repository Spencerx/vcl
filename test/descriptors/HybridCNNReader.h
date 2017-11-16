#pragma once

#include <string>
#include <fstream>

class HybridCNNReader {



    std::ifstream _ifs;

    uint32_t _dim;

public:
    HybridCNNReader();
    HybridCNNReader(std::string filename);
    void open(std::string filename);
    void read(int vectors, float* buff, size_t buf_size);
};