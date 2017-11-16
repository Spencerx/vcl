/**
* @file   DescriptorsData.h
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
* This file declares the C++ Interface for the abstract DescriptorsData object.
*/

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <map>

namespace VCL {

    class DescriptorsData {

    protected:

        std::string _set_path;
        unsigned _dimensions;

        inline bool file_exist(const std::string& name) {
            std::ifstream f(name.c_str());
            if (f.good()) {
                f.close();
                return true;
            }
            return false;
        }

    public:
        /**
         *  Loads an existing collection located at collection_path
         *  or created a new collection if it does not exist
         *  Returns error if the set does not exist
         *
         *  @param filename  Full Path to the collection folder
         */
        DescriptorsData(const std::string &filename);

        /**
         *  Creates collection located at filename
         *  or created a new collection if it does not exist
         *
         *  @param filename  Full Path to the collection folder
         *  @param dim  Dimension of the descriptor
         */
        DescriptorsData(const std::string &filename, unsigned dim);

        ~DescriptorsData();

        unsigned get_dimensions() { return _dimensions; }

        /**
         *  Inserts several Descriptors and their classes into the collection
         *  Both Descriptors and classes must have the same length.
         *
         *  @param Descriptors  Vector of Descriptors
         *  @param classes  Vector of classes
         */
        // void add(const std::vector<std::vector<float>>& descriptors,
        //          std::vector<std::string>& classes);

        /**
         *  Inserts several Descriptors and their classes into the collection
         *  Both Descriptors and classes must have the same length.
         *
         *  @param Descriptors  Pointer to buffer containing the Descriptors
         *  @param Descriptors_size  Number of elements per featureVector
         *  @param classes  Vector of classes
         */
        virtual void add(float* descriptors, unsigned n_descriptors,
                         long* labels = NULL) = 0;

        /**
         *  Trains the index with the data present in the collection
         *  using the specified metric
         *
         *  @param type Type of index, (kinda the type in Faiss)
         *  @param metric Metric for the index
         */
        // void train(VCL::Descriptors::IndexType type, VCL::Descriptors::Metric metric);
        virtual void train() {}

        /**
         *  Returns true if the index is trained (train() method called),
         *  false otherwhise.
         */
        virtual bool is_trained() {return false;}

        /**
         *  Search for the k closest neighborhs
         *
         *  @param k  Number of maximun neighbors to be returned
         *  @param query  Query vector
         *  @param distances  distances of each neighbor.
         */
        virtual void search(float* query, unsigned n, unsigned k,
                            long* descriptors, float* distances) = 0;

        /**
         *  Find the label of the feature vector, based on the closest
         *  neighbors.
         *
         *  @param k  Number of maximun neighbors to be returned
         *  @param query  Query vector
         *  @param radius  maximun distance allowed
         *  @param distances  distances of each neighbor
         */
        virtual void classify(float* descriptors, unsigned n, long* ids) = 0;

        virtual std::vector<std::string> get_labels(long* ids, unsigned n) = 0;

        virtual std::map<long, std::string> get_labels() = 0;

        void set_labels(std::vector<long>& ids,
                        std::vector<std::string>& labels);

        virtual void set_labels(std::map<long,std::string>& labels) = 0;

        virtual void get_descriptors(long* ids, unsigned n,
                                     float* descriptors) = 0;

        /**
         *  Writes the FeatureVector Index to the system. This will overwrite
         *  the original
         */
        virtual void store() = 0;
        virtual void store(std::string collection_path) = 0;
    };

};