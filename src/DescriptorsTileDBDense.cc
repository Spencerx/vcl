#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

#include "DescriptorsTileDB.h"

#define ATTRIBUTE_NAME "val"

using namespace VCL;

DescriptorsTileDBDense::DescriptorsTileDBDense(const std::string &filename):
    DescriptorsTileDB(filename),
    _flag_buffer_updated(false)
{
    std::vector<float> metadata(2);

    tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_READ);
    query.set_layout(TILEDB_ROW_MAJOR);
    query.set_subarray<uint64_t>( {0, 1, METADATA_OFFSET, METADATA_OFFSET} );
    query.set_buffer(ATTRIBUTE_NAME, metadata);

    query.submit();

    _dimensions = (unsigned)   std::round(metadata[0]);
    _n_total = (uint64_t) std::round(metadata[1]);

    // std::cout << metadata.size() << std::endl;
    // std::cout << metadata[0] << std::endl;
    // std::cout << metadata[1] << std::endl;
    // printf("dim: %d\n", _dimensions);
    // std::cout << _n_total << std::endl;
    // printf("n_total: %ld\n", _n_total);
}

DescriptorsTileDBDense::DescriptorsTileDBDense(const std::string &filename,
                                     uint32_t dim):
    DescriptorsTileDB(filename, dim),
    _flag_buffer_updated(true)
{
    auto d1 = tiledb::Dimension::create<uint64_t>(
                            _tiledb_ctx, "d1", {{0, _dimensions + 1}}, 1);
    auto d2 = tiledb::Dimension::create<uint64_t>(
                            _tiledb_ctx, "d2", {{0, MAX_DESC-1}}, 10);

    tiledb::Domain domain(_tiledb_ctx);
    domain.add_dimension(d1).add_dimension(d2);

    // for (int i = 0; i < _dimensions; ++i) {
    //     std::string dim_name = "dim" + std::to_string(i);
    //     auto d = tiledb::Dimension::create<int>(_tiledb_ctx, dim_name.c_str(),
    //                                               {{-100, 100}}, 2);
    //     domain.add_dimension(d);
    // }

    tiledb::Attribute a = tiledb::Attribute::create<float>(_tiledb_ctx, ATTRIBUTE_NAME);
    a.set_compressor({TILEDB_BLOSC_LZ, -1});

    tiledb::ArraySchema schema(_tiledb_ctx, TILEDB_DENSE);
    schema.set_tile_order(TILEDB_ROW_MAJOR).set_cell_order(TILEDB_ROW_MAJOR);
    schema.set_domain(domain);
    schema.add_attribute(a);

    try {
        schema.check();
    } catch (tiledb::TileDBError &e) {
        std::cout << e.what() << "\n";
    }

    // Create array
    tiledb::Array::create(_set_path, schema);

    std::vector<float> metadata;
    metadata.push_back(_dimensions);
    metadata.push_back(_n_total);
    metadata.push_back(MAX_DESC);

    // Write metadata
    tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_WRITE);
    query.set_layout(TILEDB_ROW_MAJOR);
    query.set_subarray<uint64_t>({ 0, 2, METADATA_OFFSET, METADATA_OFFSET});
    query.set_buffer(ATTRIBUTE_NAME, metadata);
    query.submit();

    // const char* attributes[] = {ATTRIBUTE_NAME};
    // int64_t subarray[] = { 0, 1, -2, -2};

    // float* data[] = {&metadata[0], &metadata[1]};

    // void* buffers[] = {data};
    // uint64_t buffer_sizes[] = {2};
    // tiledb_ctx_t* ctx;
    // tiledb_ctx_create(&ctx, NULL);

    // tiledb_query_t* query;
    // tiledb_query_create(ctx, &query, _set_path.c_str(), TILEDB_WRITE);

    // tiledb_query_set_layout(ctx, query, TILEDB_ROW_MAJOR);
    // tiledb_query_set_subarray(ctx, query, subarray);
    // tiledb_query_set_buffers(ctx, query, attributes, 1, buffers, buffer_sizes);

    // // Submit query
    // tiledb_query_submit(ctx, query);

    // // Clean up
    // tiledb_query_free(ctx, &query);
    // tiledb_ctx_free(&ctx);
}

void DescriptorsTileDBDense::load_buffer()
{
    try {
        {
            _buffer.resize(_dimensions * _n_total);
            tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_READ);
            query.set_layout(TILEDB_ROW_MAJOR);
            query.set_subarray<uint64_t>(
                        { 0, _dimensions - 1, 0, _n_total - 1});
            query.set_buffer(ATTRIBUTE_NAME, _buffer);

            query.submit();
        }

        {
            _ids_vec.resize(_n_total);

            tiledb_ctx_t* ctx = _tiledb_ctx;
            const char* attributes_l[] = {ATTRIBUTE_NAME};
            uint64_t subarray_l[] = { _dimensions, _dimensions + 1,
                                      METADATA_OFFSET, METADATA_OFFSET};

            void* buffers_l[] = { _ids_vec.data()};
            uint64_t buffer_s_l[] = { _n_total * sizeof(long) };

            tiledb_query_t* q_lab;
            tiledb_query_create(ctx, &q_lab, _set_path.c_str(), TILEDB_READ);

            tiledb_query_set_layout(ctx, q_lab, TILEDB_ROW_MAJOR);
            tiledb_query_set_subarray(ctx, q_lab, subarray_l);
            tiledb_query_set_buffers(ctx, q_lab, attributes_l, 1, buffers_l, buffer_s_l);

            // Submit query
            tiledb_query_submit(ctx, q_lab);

            // Clean up
            tiledb_query_free(ctx, &q_lab);
        }

    } catch (tiledb::TileDBError &e) {
        std::cout << e.what() << " DescriptorsTileDB::load_buffer \n";
    }

    _flag_buffer_updated = true;
}

void DescriptorsTileDBDense::add(float* descriptors, unsigned n, long* labels)
{
    try {
        {
            std::vector<float> a_data;
            a_data.resize(n * _dimensions);
            std::memcpy(a_data.data(),
                        descriptors, sizeof(float) * n * _dimensions);

            tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_WRITE);
            query.set_layout(TILEDB_ROW_MAJOR);
            query.set_subarray<uint64_t>(
                        { 0, _dimensions-1,
                          _n_total, _n_total + n-1});
            query.set_buffer(ATTRIBUTE_NAME, a_data);
            query.submit();
        }

        // {
        //     const char* attributes[] = {ATTRIBUTE_NAME};
        //     uint64_t subarray[] = { 0, _dimensions - 1,
        //                             _n_total,
        //                             _n_total + n-1};

        //     void* buffers[] = {descriptors};
        //     uint64_t buffer_sizes[] = { n * _dimensions * sizeof(float) };
        //     // tiledb_ctx_t* ctx;
        //     // tiledb_ctx_create(&ctx, NULL);
        //     tiledb_ctx_t* ctx = _tiledb_ctx;

        //     tiledb_query_t* query;
        //     tiledb_query_create(ctx, &query, _set_path.c_str(), TILEDB_WRITE);

        //     tiledb_query_set_layout(ctx, query, TILEDB_ROW_MAJOR);
        //     tiledb_query_set_subarray(ctx, query, subarray);
        //     tiledb_query_set_buffers(ctx, query, attributes, 1, buffers, buffer_sizes);

        //     // Submit query
        //     tiledb_query_submit(ctx, query);

        //     // Clean up
        //     tiledb_query_free(ctx, &query);
        //     // tiledb_ctx_free(&ctx);
        // }

        if (labels != NULL) {
            const char* attributes_l[] = {ATTRIBUTE_NAME};
            uint64_t subarray_l[] = { _dimensions, _dimensions + 1,
                                      _n_total, _n_total + n-1};

            void* buffers_l[] = {labels};
            uint64_t buffer_s_l[] = { n * sizeof(long) };

            tiledb_query_t* q_lab;
            tiledb_ctx_t* ctx = _tiledb_ctx;
            tiledb_query_create(ctx, &q_lab, _set_path.c_str(), TILEDB_WRITE);

            tiledb_query_set_layout(ctx, q_lab, TILEDB_ROW_MAJOR);
            tiledb_query_set_subarray(ctx, q_lab, subarray_l);
            tiledb_query_set_buffers(ctx, q_lab, attributes_l, 1, buffers_l, buffer_s_l);

            // Submit query
            tiledb_query_submit(ctx, q_lab);

            // Clean up
            tiledb_query_free(ctx, &q_lab);
        }

        {
            tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_WRITE);
            std::vector<float> metadata;
            metadata.push_back(_n_total + n);
            query.set_subarray<uint64_t>({ 1, 1, METADATA_OFFSET, METADATA_OFFSET});
            query.set_buffer(ATTRIBUTE_NAME, metadata);

            query.submit();
        }

        // if (labels != NULL) {
        //     for (int i = 0; i < n; ++i) {
        //         _ids_vec.push_back(labels[i]);
        //     }
        // }
    } catch (tiledb::TileDBError &e) {
        std::cout << e.what() << " DescriptorsTileDB::add \n";
        _flag_buffer_updated = false;
        return;
    }

    _buffer.resize((_n_total + n) * _dimensions);
    std::memcpy(&_buffer[_n_total*_dimensions],
                descriptors, n * _dimensions * sizeof(float));

    if (labels != NULL) {
        _ids_vec.resize(_n_total + n);
        std::memcpy(&_ids_vec[_n_total],
                    labels, n * sizeof(long));
    }

    _n_total += n;

    _flag_buffer_updated = true;
}

void DescriptorsTileDBDense::search(float* query,
                                    unsigned n_queries, unsigned k,
                                    long* descriptors, float* distances)
{
    if (!_flag_buffer_updated) {
        load_buffer();
    }

    std::vector<float> d;
    d.resize(_n_total);
    std::vector<long> idxs(d.size());

    for (int i = 0; i < n_queries; ++i) {

        compute_distances(query+i, d, _buffer);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(),
                [&d](size_t i1, size_t i2) { return d[i1] < d[i2]; });

        for (int j = 0; j < k; ++j) {
            descriptors[i * k + j] = idxs[j];
            distances  [i * k + j] = d[idxs[j]];
        }
    }
}
