#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>

#include "VCL.h"
#include "helpers.h"
#include "gtest/gtest.h"

TEST(DESCRIPTORS, add_ivfflatl2_100d_2add_file)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_ivfflatl2_100d_2add_file.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissIVFFlatL2);

    index.add(xb, nb);
    index.store();

    generate_desc_linear_increase(d, nb, xb, .6);

    VCL::Descriptors index_f(index_filename);
    index_f.add(xb, nb);

    generate_desc_linear_increase(d, 4, xb, 0);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index_f.search(xb, 1, 4, desc_ids, distances);

    float results[] = {0,36,100,256};
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(int(distances[i]), int(results[i]));
    }

    index_f.store();

    delete [] xb;
}

// TileDB Dense Tests

TEST(DESCRIPTORS, add_tiledbdense_100d_file)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_tiledbdense_100d_tdb_file";
    VCL::Descriptors index_f(index_filename, unsigned(d), VCL::TileDBDense);

    index_f.add(xb, nb);
    index_f.store();

    VCL::Descriptors index(index_filename, VCL::TileDBDense);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    int exp = 0;
    // std::cout << "Descriptors: " << std::endl;
    for (auto& desc : desc_ids) {
        // std::cout << desc << " ";
        EXPECT_EQ(desc, exp++);
    }

    // std::cout << "Distances: " << std::endl;
    int results[] = {0,100,400,900};
    for (int i = 0; i < 4; ++i) {
        // std::cout << distances[i] <<  " ";
        EXPECT_EQ(distances[i], results[i]);
    }
    // std::cout << std::endl;

    index.store();

    delete [] xb;
}

TEST(DESCRIPTORS, add_tiledbdense_100d_2add_file)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_tiledbdense_100d_2add_file";
    VCL::Descriptors index_f(index_filename, unsigned(d), VCL::TileDBDense);

    index_f.add(xb, nb);

    generate_desc_linear_increase(d, nb, xb, .6);

    index_f.add(xb, nb);

    generate_desc_linear_increase(d, 4, xb, 0);

    VCL::Descriptors index(index_filename, VCL::TileDBDense);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    float results[] = {0,36,100,256};
    // This is:
    //  (0)  ^2 * 100 = 0
    //  (0.6)^2 * 100 = 36
    //  (1  )^2 * 100 = 100
    //  (1.6)^2 * 100 = 256

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(std::round(distances[i]), std::round(results[i]));
        // printf(" %f, %f \n", float(distances[i]), float(results[i]));
    }

    index.store();
    delete [] xb;
}

// TileDB Sparse
