#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>

#include "HybridCNNReader.h"
#include "VCL.h"

// #include "Chrono.h"

int main() {

    int d = 4096;
    int nb = 10000;

    float *xb = new float[d * nb];

    HybridCNNReader yfcc("hybridCNN_gmean_test.txt");
    yfcc.read(nb, xb, nb*4*d);

    VCL::Descriptors index(std::string("faiss_index.faiss"), unsigned(d));

    index.add(xb, nb);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    std::cout << "Descriptors: " << std::endl;

    for (auto& desc : desc_ids) {
        std::cout << desc << " ";
    }
    std::cout << std::endl;

    std::cout << "Distances: " << std::endl;

    for (auto& dist: distances) {
        std::cout << dist <<  " ";
    }
    std::cout << std::endl;

    std::cout << "Storing index..." << std::endl;
    index.store();

    delete [] xb;

    return 0;
}
