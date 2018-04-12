#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#define USE_COMPUTE_OMP

#ifndef USE_COMPUTE_OMP
#define USE_COMPUTE_MKL
#endif

#ifdef USE_COMPUTE_MKL
#include "mkl.h" // INTEL MKL
#endif

#include "DescriptorsTileDB.h"

#define ATTRIBUTE_NAME "val"

using namespace VCL;

DescriptorsTileDB::DescriptorsTileDB(const std::string &filename):
    DescriptorsData(filename)
{
}

DescriptorsTileDB::DescriptorsTileDB(const std::string &filename,
                                     uint32_t dim):
    DescriptorsData(filename, dim),
    _n_total(0)
{
}

DescriptorsTileDB::~DescriptorsTileDB()
{
}

void DescriptorsTileDB::train()
{
}

void DescriptorsTileDB::compute_distances(float* q,
                                          std::vector<float>& d,
                                          std::vector<float>& data)
{
    size_t n = data.size() / _dimensions;

    float* sub = new float[_dimensions * n];

#ifdef USE_COMPUTE_MKL

    // Intel MKL
    for (int i = 0; i < n; ++i) {
        size_t idx = i * _dimensions;
        vsSub(_dimensions, q, data.data() + idx, sub + idx);
        d[i] = std::pow(cblas_snrm2(_dimensions, sub + idx, 1),2);
    }
#endif

#ifdef USE_COMPUTE_OMP
    // Using RAW OpenMP / This can be optimized
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        size_t idx = i * _dimensions;

        float sum = 0;
        // #pragma omp parallel for // has to be a reduction
        for (int j = 0; j < _dimensions; ++j) {
            sum += std::pow(data[idx + j] - q[j], 2);
        }

        d[i] = sum; // std::sqrt(sum);
    }
#endif
}

void DescriptorsTileDB::classify(float* descriptors, unsigned n, long* labels)
{
    //TODO how we add the voting number
    int quorum = 7;
    float* distances = new float[n * quorum];
    long*  ids_aux   = new long [n * quorum];

    search(descriptors, n, quorum, ids_aux, distances);

    for (int j = 0; j < n; ++j) {
        std::map<long, int> map_voting;
        long winner = 0;
        unsigned max = 0;
        for (int i = 0; i < quorum; ++i) {
            long idx = ids_aux[quorum*j + i];
            if (idx < 0) continue; // Means not found

            long label_id = _ids_vec.at(idx);
            map_voting[label_id] += 1;
            if (max < map_voting[label_id]) {
                max = map_voting[label_id];
                winner = label_id;
            }
        }
        labels[j] = winner;
    }
}

std::vector<std::string> DescriptorsTileDB::get_labels(long* ids, unsigned n)
{
    return std::vector<std::string>(n);
}

void DescriptorsTileDB::get_descriptors(long* ids, unsigned n,
                                        float* descriptors)
{
}

void DescriptorsTileDB::store()
{
    // Consolidate array
    // tiledb::Array::consolidate(_tiledb_ctx, _set_path);
}

void DescriptorsTileDB::store(std::string filename)
{
}