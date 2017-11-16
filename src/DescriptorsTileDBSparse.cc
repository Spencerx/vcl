#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "DescriptorsTileDB.h"

#define ATTRIBUTE_SPARSE_ID "id"
#define ATTRIBUTE_SPARSE_LABEL "label"

using namespace VCL;

// TileDB Sparse

DescriptorsTileDBSparse::DescriptorsTileDBSparse(const std::string &filename):
    DescriptorsTileDB(filename)
{
    // TODO
}

DescriptorsTileDBSparse::DescriptorsTileDBSparse(const std::string &filename,
                                     uint32_t dim):
    DescriptorsTileDB(filename, dim)
{
    tiledb::Domain domain(_tiledb_ctx);

    for (int i = 0; i < _dimensions; ++i) {
        std::string dim_name = "dim" + std::to_string(i);
        auto d = tiledb::Dimension::create<float>(
                        _tiledb_ctx, dim_name.c_str(), {{-10000, 10000}}, 250);
        // TODO, THE DOMAINS MUST BE BASED ON SOME TRAINING
        domain.add_dimension(d);
    }

    tiledb::Attribute att_id = tiledb::Attribute::create<long>(_tiledb_ctx, ATTRIBUTE_SPARSE_ID);
    // att_id.set_compressor({TILEDB_BLOSC_LZ, -1});

    tiledb::Attribute att_label = tiledb::Attribute::create<long>(_tiledb_ctx, ATTRIBUTE_SPARSE_LABEL);
    // att_label.set_compressor({TILEDB_BLOSC_LZ, -1});

    tiledb::ArraySchema schema(_tiledb_ctx, TILEDB_SPARSE);
    schema.set_tile_order(TILEDB_ROW_MAJOR).set_cell_order(TILEDB_ROW_MAJOR);
    schema.set_capacity(100);
    schema.set_domain(domain);
    schema.add_attribute(att_id);
    schema.add_attribute(att_label);

    try {
        schema.check();
    } catch (tiledb::TileDBError &e) {
        std::cout << e.what() << "\n";
    }

    // Create array
    tiledb::Array::create(_set_path, schema);
}

void DescriptorsTileDBSparse::add(float* descriptors, unsigned n, long* labels)
{
    try {

        std::vector<float> idxs(_dimensions * n);
        // for (int i = 0; i < n * _dimensions; ++i)
        // {
        //     idxs[i] = std::round(descriptors[i]);
        // }
        std::memcpy(idxs.data(), descriptors, n * _dimensions * sizeof(float));

        std::vector<long> att_id(n);
        std::iota(att_id.begin(), att_id.end(), _n_total);

        std::vector<long> att_label(n, -1); // By default, labels is -1
        if (labels != NULL)
            std::memcpy(att_label.data(), labels, n * sizeof(long));

        tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_WRITE);
        query.set_layout(TILEDB_UNORDERED);
        query.set_buffer(ATTRIBUTE_SPARSE_ID, att_id);
        query.set_buffer(ATTRIBUTE_SPARSE_LABEL, att_label);
        query.set_coordinates(idxs);

        // Submit query
        query.submit();

    } catch (tiledb::TileDBError &e) {
        std::cout << e.what() << " DescriptorsTileDB::add \n";
        return;
    }

    _n_total += n;
}

void DescriptorsTileDBSparse::load_neighbors(float* q, unsigned k,
                                        std::vector<float>& descriptors,
                                        std::vector<long>& desc_ids,
                                        std::vector<long>& desc_labels)
{
    // Print non-empty domain
    auto domain =
      tiledb::Array::non_empty_domain<float>(_tiledb_ctx, _set_path);
    // std::cout << "Non empty domain:\n";
    // for (const auto& d : domain) {
    //     std::cout << d.first << ": (" << d.second.first << ", " << d.second.second
    //           << ")\n";
    // }

    // Calculate maximum buffer elements for the query results per attribute

    std::vector<float> subarray(_dimensions * 2);

    float space = 10;

    #pragma omp parallel for
    for (int i = 0; i < _dimensions; ++i) {
        subarray[2*i]   = q[i] - space;
        subarray[2*i+1] = q[i] + space;
    }

    auto max_sizes =
      tiledb::Array::max_buffer_sizes(_tiledb_ctx, _set_path, subarray);

    // std::cout << "\nMaximum buffer elements:\n";
    // for (const auto& e : max_sizes) {
    //     std::cout << e.first << ": (" << e.second.first
    //               << ", " << e.second.second << ")\n";
    // }
    // std::cout << std::endl;

    // std::vector<long> att_id_buff(
    //                 max_sizes[ATTRIBUTE_SPARSE_ID].second / sizeof(long));
    // std::vector<long> att_label_buff(
    //                 max_sizes[ATTRIBUTE_SPARSE_LABEL].second / sizeof(long));
    // std::vector<float> coords_buff(max_sizes[TILEDB_COORDS].first);

    // Prepare cell buffers
    descriptors.resize(max_sizes[TILEDB_COORDS].first);
    desc_ids.resize(max_sizes[ATTRIBUTE_SPARSE_ID].second / sizeof(long));
    desc_labels.resize(max_sizes[ATTRIBUTE_SPARSE_LABEL].second / sizeof(long));

    // std::cout << "coordbuff size: " << coords_buff.size() << std::endl;
    // std::cout << "idbuff size: " << att_id_buff.size() << std::endl;
    // std::cout << "labelbuff size: " << att_label_buff.size() << std::endl;

    // Create query
    tiledb::Query query(_tiledb_ctx, _set_path, TILEDB_READ);
    query.set_layout(TILEDB_ROW_MAJOR).set_subarray(subarray);
    query.set_buffer(ATTRIBUTE_SPARSE_ID, desc_ids);
    query.set_buffer(ATTRIBUTE_SPARSE_LABEL, desc_labels);
    query.set_coordinates(descriptors);

    // Submit query
    query.submit();

    // Print cell values (assumes all attributes are read)
    auto result_el = query.result_buffer_elements();
    // auto a2_buff =
    //   std::pair<std::vector<uint64_t>, std::vector<char>>(a2_offsets, a2_data);
    // auto a2 = tiledb::group_by_cell(
    //   a2_buff, result_el["a2"].first, result_el["a2"].second);
    // auto a3 = tiledb::group_by_cell<2>(a3_buff);
    // auto coords =
    //   tiledb::group_by_cell<2>(coords_buff, result_el[TILEDB_COORDS].first);

    unsigned n_close = result_el[ATTRIBUTE_SPARSE_ID].second;

    // std::cout << "Result num: " << result_el[ATTRIBUTE_SPARSE_ID].second << "\n\n";
    // std::cout << std::setw(8) << TILEDB_COORDS << std::setw(9) << "id"
    //         << std::setw(9) << "\n";
    // std::cout << "------------------------------------------------\n";

    // for (int i = 0; i < n_close; ++i) {
    //     std::cout << _buffer[_dimensions*i] << ",";
    //     std::cout << _buffer[_dimensions*i+1] << ",";
    //     std::cout << _buffer[_dimensions*i+2] << " - ID: ";
    //     std::cout << _desc_ids[i] << " - Label: ";
    //     std::cout << _ids_vec[i] << "\n";
    // }

    descriptors.resize(n_close * _dimensions);
    desc_ids.resize(n_close);
    desc_labels.resize(n_close);

    _n_total = n_close;

    // for (unsigned i = 0; i < result_el[ATTRIBUTE_SPARSE_ID].first; ++i) {
    // std::cout << "(" << coords[i][0] << ", " << coords[i][1] << ")"
    //           << std::setw(10) << att_id_buff[i] << std::setw(10)
    //           << att_label_buff[i] << '\n';
    // }
}

void DescriptorsTileDBSparse::search(float* query, unsigned n, unsigned k,
                                    long* descriptors, float* distances)
{
    std::vector<float> descs; // we use _buffer for this
    std::vector<long> desc_ids;
    std::vector<long> desc_labels;

    load_neighbors(query, k, descs, desc_ids, desc_labels);
    unsigned found = desc_ids.size();

    std::vector<float> d;
    d.resize(found);
    std::vector<long> idxs(found);

    for (int i = 0; i < n; ++i) {

        if ( i >= found) {
               for (int j = 0; j < k; ++j) {
                descriptors[i * k + j] = -1;
                distances  [i * k + j] = -1;
            }
            break;
        }

        compute_distances(query+i, d, descs);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(),
                [&d](size_t i1, size_t i2) { return d[i1] < d[i2]; });

        for (int j = 0; j < k; ++j) {
            descriptors[i * k + j] = desc_ids[idxs[j]];
            distances  [i * k + j] = d[idxs[j]];
        }
    }
}
