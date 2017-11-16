#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>

#include "VCL.h"
#include "chrono/Chrono.h"
#include "helpers.h"

void compare_all_search(int d, int nb)
{
    float *xb = generate_desc_linear_increase(d, nb);
    std::vector<float> distances;
    std::vector<long> desc_ids;

    std::string index_filename;

    // FAISS FLAT L2

    index_filename = "dbs/search_faiss_flatl2_" + std::to_string(d) + "_" + std::to_string(nb) + ".faiss";
    VCL::Descriptors
        i_faiss_flatl2(index_filename, unsigned(d), VCL::FaissFlatL2);

    i_faiss_flatl2.add(xb, nb);

    ChronoCpu ch_i_faiss_flatl2("faiss_flatl2");

    ch_i_faiss_flatl2.tic();
    i_faiss_flatl2.search(xb, 1, 4, desc_ids, distances);
    ch_i_faiss_flatl2.tac();


    // FIASS IVFFLAT L2

    index_filename = "dbs/search_faiss_ivfflatl2_" + std::to_string(d) + "_" + std::to_string(nb) + ".faiss";
    VCL::Descriptors
        i_faiss_ivfflatl2(index_filename, unsigned(d), VCL::FaissIVFFlatL2);

    i_faiss_ivfflatl2.add(xb, nb);

    ChronoCpu ch_i_faiss_ivfflatl2("faiss_ivfflatl2");

    ch_i_faiss_ivfflatl2.tic();
    i_faiss_ivfflatl2.search(xb, 1, 4, desc_ids, distances);
    ch_i_faiss_ivfflatl2.tac();


    // TILEDB DENSE

    index_filename = "dbs/search_tiledbdense_" + std::to_string(d) + "_" + std::to_string(nb);
    VCL::Descriptors i_tiledb_dense(index_filename, unsigned(d), VCL::TileDBDense);

    i_tiledb_dense.add(xb, nb);

    ChronoCpu ch_i_tiledb_dense("tiledb_dense");

    ch_i_tiledb_dense.tic();
    i_tiledb_dense.search(xb, 1, 4, desc_ids, distances);
    ch_i_tiledb_dense.tac();

    // TILEDB Sparse

    index_filename = "dbs/search_tiledbsparse_" + std::to_string(d) + "_" + std::to_string(nb);
    VCL::Descriptors i_tiledb_sparse(index_filename, unsigned(d), VCL::TileDBSparse);

    i_tiledb_sparse.add(xb, nb);

    ChronoCpu ch_i_tiledb_sparse("tiledb_sparse");

    ch_i_tiledb_sparse.tic();
    i_tiledb_sparse.search(xb, 1, 4, desc_ids, distances);
    ch_i_tiledb_sparse.tac();

    std::cout << d << "\t" << nb << "\t";
    std::cout << ch_i_faiss_flatl2.getTotalTime_ms() << "\t";
    std::cout << ch_i_faiss_ivfflatl2.getTotalTime_ms() << "\t";
    std::cout << ch_i_tiledb_dense.getTotalTime_ms() << "\t";
    std::cout << ch_i_tiledb_sparse.getTotalTime_ms() << "\t";
    std::cout << std::endl;

    delete[] xb;
}

void compare_all_add(int d, int nb)
{
    float *xb = generate_desc_linear_increase(d, nb);
    std::vector<float> distances;
    std::vector<long> desc_ids;

    std::string index_filename;

    // FAISS FLAT L2

    index_filename = "dbs/add_faiss_flatl2_" + std::to_string(d) + "_" + std::to_string(nb) + ".faiss";
    VCL::Descriptors
        i_faiss_flatl2(index_filename, unsigned(d), VCL::FaissFlatL2);

    ChronoCpu ch_i_faiss_flatl2("faiss_flatl2");

    ch_i_faiss_flatl2.tic();
    i_faiss_flatl2.add(xb, nb);
    ch_i_faiss_flatl2.tac();

    // FIASS IVFFLAT L2

    index_filename = "dbs/add_faiss_ivfflatl2_" + std::to_string(d) + "_" + std::to_string(nb) + ".faiss";
    VCL::Descriptors
        i_faiss_ivfflatl2(index_filename, unsigned(d), VCL::FaissIVFFlatL2);

    ChronoCpu ch_i_faiss_ivfflatl2("faiss_ivfflatl2");

    ch_i_faiss_ivfflatl2.tic();
    i_faiss_ivfflatl2.add(xb, nb);
    ch_i_faiss_ivfflatl2.tac();

    // TILEDB DENSE

    index_filename = "dbs/add_tiledbdense_" + std::to_string(d) + "_" + std::to_string(nb);
    VCL::Descriptors i_tiledb_dense(index_filename, unsigned(d), VCL::TileDBDense);

    ChronoCpu ch_i_tiledb_dense("tiledb_dense");

    ch_i_tiledb_dense.tic();
    i_tiledb_dense.add(xb, nb);
    ch_i_tiledb_dense.tac();

    // TILEDB Sparse

    index_filename = "dbs/add_tiledbsparse_" + std::to_string(d) + "_" + std::to_string(nb);
    VCL::Descriptors i_tiledb_sparse(index_filename, unsigned(d), VCL::TileDBSparse);

    ChronoCpu ch_i_tiledb_sparse("tiledb_sparse");

    ch_i_tiledb_sparse.tic();
    i_tiledb_sparse.add(xb, nb);
    ch_i_tiledb_sparse.tac();

    std::cout << d << "\t" << nb << "\t";
    std::cout << ch_i_faiss_flatl2.getTotalTime_ms() << "\t";
    std::cout << ch_i_faiss_ivfflatl2.getTotalTime_ms() << "\t";
    std::cout << ch_i_tiledb_dense.getTotalTime_ms() << "\t";
    std::cout << ch_i_tiledb_sparse.getTotalTime_ms() << "\t";
    std::cout << std::endl;

    delete[] xb;
}

int main()
{
    std::cout << "Search..." << std::endl;
    compare_all_search(100,  10000);
    compare_all_search(1000, 10000);
    compare_all_search(10,   25000);
    compare_all_search(100,  25000);
    compare_all_search(1000, 25000);
    compare_all_search(4000, 25000);
    // compare_all_search(100,  10000);
    // compare_all_search(1000, 10000);

    std::cout << "Add..." << std::endl;
    compare_all_add(10,   10000);
    compare_all_add(100,  10000);
    compare_all_add(100,  25000);
    compare_all_add(1000, 25000);
    compare_all_add(4000, 25000);

    return 0;
}
