#include <sstream>
#include <vector>
#include <iostream>

#include "HybridCNNReader.h"

HybridCNNReader::HybridCNNReader()
{}

HybridCNNReader::HybridCNNReader(std::string filename)
{
    open(filename);
}

void HybridCNNReader::open(std::string filename)
{
    _ifs.open(filename);

    if (!_ifs)
        printf("problem opening file?\n");

    std::string str;

    std::getline(_ifs, str); // First line of file is empty
    std::getline(_ifs, str);

    std::stringstream ss(str);
    std::string str_aux;
    std::getline(ss, str_aux, '\t');
    // std::cout << str_aux << std::endl;
    std::getline(ss, str_aux, '\t');
    // std::cout << str_aux << std::endl;

    float aux;
    int counter = 0;
    while (ss >> aux) {

        ++counter;
    }

    _dim = counter;
    // printf("%d dim\n", _dim);

    _ifs.close();
    _ifs.open(filename);
    std::getline(_ifs, str); // First line of file is empty
}

void HybridCNNReader::read(int vectors, float* buff, size_t buf_size)
{
    if (buf_size < vectors * sizeof(float) * _dim) {
        printf("not enough space in buffer\n");
    }

    char str[256];

    for (int i = 0; i < vectors; ++i) {

        _ifs.getline(str, 256, '\t');
        _ifs.getline(str, 256, '\t');
        // printf("%s\n",str );

        for (int j = 0; j < _dim; ++j) {
            _ifs >> *(buff+j*i);
            // std::cout << *(buff+j) << " ";
        }
        // std::cout << std::endl;
    }

}

