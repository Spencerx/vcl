#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>

#include "VCL.h"
#include "helpers.h"
#include "gtest/gtest.h"

TEST(DESCRIPTORS, add_flatl2_100d)
{
    int d = 100;
    int nb = 10000;

    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_flatl2_100d.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissFlatL2);

    index.add(xb, nb);

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
    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(2, 2)*d),
                       float(std::pow(3, 2)*d) };
    for (int i = 0; i < 4; ++i) {
        // std::cout << distances[i] <<  " ";
        EXPECT_EQ(distances[i], results[i]);
    }
    // std::cout << std::endl;

    index.store();

    delete [] xb;
}

TEST(DESCRIPTORS, add_ivfflatl2_100d)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_ivfflatl2_100d.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissIVFFlatL2);

    std::vector<long> classes(nb);

    for (auto& str : classes) {
        str = 1;
    }

    index.add(xb, nb, classes);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    int exp = 0;
    // std::cout << "Descriptors: " << std::endl;
    for (auto& desc : desc_ids) {
        // std::cout << desc << " ";
        EXPECT_EQ(desc, exp++);
    }
    // std::cout << std::endl;

    // std::cout << "Distances: " << std::endl;
    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(2, 2)*d),
                       float(std::pow(3, 2)*d) };
    for (int i = 0; i < 4; ++i) {
        // std::cout << distances[i] <<  " ";
        EXPECT_EQ(distances[i], results[i]);
    }
    // std::cout << std::endl;

    index.store();

    delete [] xb;
}

TEST(DESCRIPTORS, add_recons_flatl2_100d)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_recons_flatl2_100d.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissFlatL2);

    std::vector<long> classes(nb);

    for (auto& cl : classes) {
        cl = 1;
    }

    index.add(xb, nb, classes);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);
    desc_ids.clear();

    float *recons = new float[d * nb];
    for (int i = 0; i < nb; ++i) {
        desc_ids.push_back(i);
    }

    index.get_descriptors(desc_ids, recons);

    for (int i = 0; i < nb*d; ++i) {
        EXPECT_EQ(xb[i], recons[i]);
    }

    index.store();

    delete [] xb;
}

TEST(DESCRIPTORS, add_flatl2_100d_2add)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_flatl2_100d_2add.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissFlatL2);

    index.add(xb, nb);

    generate_desc_linear_increase(d, nb, xb, .6);

    index.add(xb, nb);

    generate_desc_linear_increase(d, 4, xb, 0);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(.6, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(1.6, 2)*d) };
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(int(distances[i]), int(results[i]));
    }

    index.store();
    delete [] xb;
}

// TileDB Dense Tests

TEST(DESCRIPTORS, add_tiledbdense_100d)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_tiledbdense_100d_tdb";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::TileDBDense);

    index.add(xb, nb);

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
    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(2, 2)*d),
                       float(std::pow(3, 2)*d) };
    for (int i = 0; i < 4; ++i) {
        // std::cout << distances[i] <<  " ";
        EXPECT_EQ(distances[i], results[i]);
    }
    // std::cout << std::endl;

    index.store();

    delete [] xb;
}

TEST(DESCRIPTORS, add_tiledbdense_100d_2add)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);

    std::string index_filename = "dbs/add_tiledbdense_100d_2add";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::TileDBDense);

    index.add(xb, nb);

    generate_desc_linear_increase(d, nb, xb, .6);

    index.add(xb, nb);

    generate_desc_linear_increase(d, 4, xb, 0);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(.6, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(1.6, 2)*d) };
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

TEST(DESCRIPTORS, add_tiledbsparse_100d_2add)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);
    // generate_desc_linear_increase(d, nb, xb, .1);

    std::string index_filename = "dbs/add_tiledbsparse_100d_2add";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::TileDBSparse);

    index.add(xb, nb);

    generate_desc_linear_increase(d, nb, xb, .6);

    index.add(xb, nb);

    generate_desc_linear_increase(d, 4, xb, 0);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 2, 4, desc_ids, distances);

    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(.6, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(1.6, 2)*d) };

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(std::round(distances[i]), std::round(results[i]));
    }

    index.store();
    delete [] xb;
}

// TileDB Sparse

TEST(DESCRIPTORS, add_tiledbsparse_100d)
{
    int d = 100;
    int nb = 10000;
    float *xb = generate_desc_linear_increase(d, nb);
    // generate_desc_linear_increase(d, nb, xb, .1);

    std::string index_filename = "dbs/add_tiledbsparse_100d";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::TileDBSparse);

    index.add(xb, nb);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    float results[] = {float(std::pow(0, 2)*d),
                       float(std::pow(1, 2)*d),
                       float(std::pow(2, 2)*d),
                       float(std::pow(3, 2)*d) };

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(std::round(distances[i]), std::round(results[i]));
    }

    index.store();
    delete [] xb;
}
