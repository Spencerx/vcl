/**
 * @file   Descriptors.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * This file declares the C++ API for Descriptors.
 */

#pragma once

#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>
#include <map>

#include <tiledb/tiledb>

#include "DescriptorsData.h"
#include "Exception.h"

namespace VCL {

    typedef std::vector<float> DescData;
    typedef std::vector<float> DistanceData;

    class DescriptorsTileDB: public DescriptorsData{

    protected:
        const unsigned long MAX_DESC = 100000;

        tiledb::Context _tiledb_ctx;
        std::map<long, std::string> _labels_map;

        uint64_t _n_total;

        std::vector<long> _ids_vec; // we need to move this


        void compute_distances(float* q,
                               DistanceData& d,
                               DescData& data);

    public:
        /**
         *  Loads an existing collection located at collection_path
         *  or created a new collection if it does not exist
         *
         *  @param collection_path  Full Path to the collection folder
         */
        DescriptorsTileDB(const std::string &collection_path);

        DescriptorsTileDB(const std::string &collection_path, unsigned dim);

        ~DescriptorsTileDB();

        virtual void add(float* descriptors, unsigned n_descriptors, long* classes) = 0;

        virtual void train();

        bool is_trained() {return false;}

        virtual void search(float* query, unsigned n_queries, unsigned k,
                    long* descriptors, float* distances) = 0;

        virtual void classify(float* descriptors, unsigned n, long* labels);

        std::map<long, std::string> get_labels() {return _labels_map;}

        std::vector<std::string> get_labels(long* ids, unsigned n);

        void set_labels(std::map<long,std::string>& labels)
        {
            _labels_map = labels;
        }

        void get_descriptors(long* ids, unsigned n, float* descriptors);

        void store();
        void store(std::string set_path);
    };

    class DescriptorsTileDBDense : public DescriptorsTileDB {

    private:

        const unsigned long METADATA_OFFSET = MAX_DESC - 2;

        bool _flag_buffer_updated;
        std::vector<float> _buffer;

        void load_buffer();

    public:
        DescriptorsTileDBDense(const std::string &collection_path);

        DescriptorsTileDBDense(const std::string &collection_path,
                               unsigned dim);

        ~DescriptorsTileDBDense();

        void add(float* descriptors, unsigned n_descriptors, long* classes);

        void search(float* query, unsigned n_queries, unsigned k,
                    long* descriptors, float* distances);
    };

    class DescriptorsTileDBSparse : public DescriptorsTileDB {

    private:

        void load_neighbors(float* query, unsigned k,
                            std::vector<float>& descriptors,
                            std::vector<long>& desc_ids,
                            std::vector<long>& desc_labels);

    public:
        DescriptorsTileDBSparse(const std::string &collection_path);

        DescriptorsTileDBSparse(const std::string &collection_path,
                               unsigned dim);

        ~DescriptorsTileDBSparse();

        void add(float* descriptors, unsigned n_descriptors, long* classes);

        void search(float* query, unsigned n_queries, unsigned k,
                    long* descriptors, float* distances);


    };

};