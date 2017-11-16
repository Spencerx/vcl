#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>

#include "VCL.h"
#include "gtest/gtest.h"
#include "helpers.h"

TEST(DESCRIPTORS, classify_flatl2_4d)
{
    int d = 4;
    int nb = 10000;

    float *xb = generate_desc_linear_increase(d, nb);

    auto class_map = animals_map();

    std::string index_filename = "dbs/classify_flatl2_4d.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissFlatL2);

    index.set_labels(class_map);

    int offset = 10;
    std::vector<long> classes = classes_increasing_offset(nb, offset);

    index.add(xb, nb, classes);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index.search(xb, 1, 4, desc_ids, distances);

    int exp = 0;
    for (auto& desc : desc_ids) {
        EXPECT_EQ(desc, exp++);
    }

    int results[] = {0,4,16,36};
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(distances[i], results[i]);
    }

    // index.search(xb, 60, 7, desc_ids, distances);
    // for (int i = 0; i < 60; ++i)
    // {
    //     for (int j = 0; j < 7; ++j)
    //     {
    //         std::cout << desc_ids[i*7 + j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<long> ret_ids = index.classify(xb, 60);

    exp = 1;
    int i = 0;
    for (auto& id : ret_ids) {
        // EXPECT_EQ(id, (++i)%offset == 0 ? ++exp : exp);
        printf("%ld ", id);
    }
    printf("\n");

    std::vector<std::string> ret = index.label_id_to_string(ret_ids);
    // std::cout << ret.size() << std::endl;

    // for (auto& label : ret) {
    //     std::cout << label << std::endl;
    // }

    for (int i = 0; i < offset; ++i) {
        EXPECT_EQ(ret[i], "dog");
        EXPECT_EQ(ret[i+offset], "cat");
        EXPECT_EQ(ret[i+2*offset], "messi");
        EXPECT_EQ(ret[i+3*offset], "bird");
        EXPECT_EQ(ret[i+4*offset], "condor");
        EXPECT_EQ(ret[i+5*offset], "panda");
    }

    desc_ids.clear();
    distances.clear();

    index.search(xb, 1, offset, desc_ids, distances);
    ret = index.get_labels(desc_ids);

    for (auto& label : ret) {
        // std::cout << label << std::endl;
        EXPECT_EQ(label, "dog");
    }

    index.store();

    delete [] xb;
}

TEST(DESCRIPTORS, classify_ivfflatl2_4d_file)
{
    int d = 4;
    int nb = 10000;

    float *xb = generate_desc_linear_increase(d, nb);

    auto class_map = animals_map();

    std::string index_filename = "dbs/classify_ivfflatl2_4d_file.faiss";
    VCL::Descriptors index(index_filename, unsigned(d), VCL::FaissIVFFlatL2);

    int offset = 10;
    std::vector<long> classes = classes_increasing_offset(nb, offset);

    index.set_labels(class_map);

    index.add(xb, nb, classes);
    index.store();

    VCL::Descriptors index_f(index_filename);

    std::vector<float> distances;
    std::vector<long> desc_ids;
    index_f.search(xb, 1, 4, desc_ids, distances);

    int exp = 0;
    for (auto& desc : desc_ids) {
        EXPECT_EQ(desc, exp++);
    }

    int results[] = {0,4,16,36};
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(distances[i], results[i]);
    }

    std::vector<long> ret_ids = index.classify(xb, 60);
    std::vector<std::string> ret = index.label_id_to_string(ret_ids);

    for (int i = 0; i < offset; ++i) {
        EXPECT_EQ(ret[i], "dog");
        EXPECT_EQ(ret[i+offset], "cat");
        EXPECT_EQ(ret[i+2*offset], "messi");
        EXPECT_EQ(ret[i+3*offset], "bird");
        EXPECT_EQ(ret[i+4*offset], "condor");
        EXPECT_EQ(ret[i+5*offset], "panda");
    }

    index_f.search(xb, 1, offset, desc_ids, distances);
    ret = index_f.get_labels(desc_ids);

    for (auto& label : ret){
        EXPECT_EQ(label, "dog");
    }

    delete [] xb;
}


// TILEDBDense tests

// TEST(DESCRIPTORS, classify_tdbdense_4d)
// {
//     int d = 4;
//     int nb = 10000;

//     float *xb = generate_desc_linear_increase(d, nb);

//     auto class_map = animals_map();

//     std::string index_filename = "dbs/classify_tdbdense_4d";
//     VCL::Descriptors index(index_filename, unsigned(d), VCL::TileDBDense);

//     index.set_labels(class_map);

//     int offset = 10;
//     std::vector<long> classes = classes_increasing_offset(nb, offset);

//     index.add(xb, nb, classes);

//     std::vector<float> distances;
//     std::vector<long> desc_ids;
//     index.search(xb, 1, 4, desc_ids, distances);

//     int exp = 0;
//     for (auto& desc : desc_ids) {
//         EXPECT_EQ(desc, exp++);
//     }

//     int results[] = {0,4,16,36};
//     for (int i = 0; i < 4; ++i) {
//         EXPECT_EQ(distances[i], results[i]);
//     }

//     std::vector<long> ret_ids = index.classify(xb, 60);

//     for (auto& desc : ret_ids) {
//         printf("%ld ", desc);
//     }
//     printf("\n");

//     std::vector<std::string> ret = index.label_id_to_string(ret_ids);
//     // std::cout << ret.size() << std::endl;

//     // for (auto& label : ret) {
//     //     std::cout << label << std::endl;
//     // }

//     for (int i = 0; i < offset; ++i) {
//         EXPECT_EQ(ret[i], "dog");
//         EXPECT_EQ(ret[i+offset], "cat");
//         EXPECT_EQ(ret[i+2*offset], "messi");
//         EXPECT_EQ(ret[i+3*offset], "bird");
//         EXPECT_EQ(ret[i+4*offset], "condor");
//         EXPECT_EQ(ret[i+5*offset], "panda");
//     }

//     desc_ids.clear();
//     distances.clear();

//     // index.search(xb, 1, offset, desc_ids, distances);
//     // ret = index.get_labels(desc_ids);

//     // for (auto& label : ret) {
//     //     // std::cout << label << std::endl;
//     //     EXPECT_EQ(label, "dog");
//     // }

//     index.store();

//     delete [] xb;
// }
