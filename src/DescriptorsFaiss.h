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
#include <unordered_map>
#include <map>
#include <mutex>

#include "DescriptorsData.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

namespace VCL {

    class DescriptorsFaiss : public DescriptorsData {

    protected:

        faiss::Index* _index;
        // std::unordered_map<std::string, long> _labels_map;
        // std::vector<std::string> _labels_vec;
        std::vector<long> _ids_vec;

        std::map<long, std::string> _labels_map;

        float* _add_buffer;
        unsigned _buffer_elements;
        std::vector<long> _labels_buffer;
        // Protects internal data structures and make the implementation
        // thread safe
        // (only when using the same instance in multiple threads,
        //  which is the case of VDMS)
        std::mutex _lock;

        std::string _faiss_file;

        void write_maps();
        void read_maps();

    public:

        DescriptorsFaiss(const std::string &set_path);
        DescriptorsFaiss(const std::string &set_path, unsigned dim);

        ~DescriptorsFaiss();

        void add(float* descriptors, unsigned n_descriptors, long* classes);

        void train() override;

        bool is_trained() override;

        void search(float* query, unsigned n_queries, unsigned k,
                    long* descriptors, float* distances) override;

        void classify(float* descriptors, unsigned n, long* ids) override;

        std::map<long, std::string> get_labels() {return _labels_map;}

        std::vector<std::string> get_labels(long* ids, unsigned n);

        void set_labels(std::map<long,std::string>& labels);

        void get_descriptors(long* ids, unsigned n, float* descriptors) override;

        void store();
        void store(std::string set_path);

    private:
        void init_buffers();
        void flush_buffers();
    };

    class DescriptorsFaissFlatL2 : public DescriptorsFaiss {

    public:

        DescriptorsFaissFlatL2(const std::string& set_path);
        DescriptorsFaissFlatL2(const std::string& set_path, unsigned dim);

        // void add(float* descriptors, unsigned n_descriptors,
        //          std::vector<std::string>& classes);

    };

    class DescriptorsFaissIVFFlatL2 : public DescriptorsFaiss {

    public:

        DescriptorsFaissIVFFlatL2(const std::string& set_path);
        DescriptorsFaissIVFFlatL2(const std::string& set_path, unsigned dim);

    };

};