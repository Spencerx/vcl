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

#include "Exception.h"

namespace VCL {

    // Move this to the inside of the class.
    class DescriptorsData;

    enum DescriptorsEngine {FaissFlatL2, FaissIVFFlatL2,
                            TileDBDense, TileDBSparse};

    enum DistanceMetric {L2, IP};

    // This will be changed to DescritorSet
    class Descriptors {

    private:

        DescriptorsData* _descriptors;

    public:
        /**
         *  Loads an existing collection located at set_path
         *
         *  @param set_path  Full Path to the collection folder
         */
        Descriptors(const std::string& set_path,
                    DescriptorsEngine eng = FaissIVFFlatL2);
        // TODO CHANGE THE DEFAULTS.

        /**
         *  Created a new collection, if it does not exist
         *
         *  @param set_path  Full Path to the set folder
         *  @param dim  Dimension of the descriptor
         *  @param eng  Descriptors Engine
         */
        Descriptors(const std::string &set_path, unsigned dim,
                    DescriptorsEngine eng = FaissFlatL2);
        // todo it is better not to have a default.

        ~Descriptors();

        /**
         *  Inserts several Descriptors and their labels into the collection
         *  Both Descriptors and labels must have the same length.
         *
         *  @param descriptors  Pointer to buffer containing the Descriptors
         *  @param n  Number of elements per descriptor
         *  @param labels  Vector of labels
         */
        void add(float* descriptors, unsigned n, std::vector<long>& labels);

        // No need to have n
        // this will return the first id.

        /**
         *  Inserts several Descriptors and their labels into the collection
         *  Both Descriptors and labels must have the same length.
         *
         *  @param descriptors  Pointer to buffer containing the Descriptors
         *  @param n  Number of elements per featureVector
         *  @param labels Array of labels
         */
        void add(float* descriptors, unsigned n, long* labels = NULL);

        /**
         *  Trains the index with the data present in the collection
         *  using the specified metric
         *
         *  @param type Type of index, (kinda the type in Faiss)
         *  @param metric Metric for the index
         */
        // void train(VCL::Descriptors::IndexType type,
        //            VCL::Descriptors::Metric metric);
        void train();
        // // Check if faiss support building of the index
        // //

        // void train(VCL::Descriptors::IndexType type,
        //            VCL::Descriptors::Metric metric,
        //            float* descriptors, unsigned n);

        /**
         *  Returns true if the index is trained (train() method called),
         *  false otherwhise.
         */
        bool is_trained();

        /**
         *  Search for the k closest neighborhs
         *      // Add comment on why we use k and n_queries.
         *      // We can also get rid of the
         *
         *  @param query  Query descriptors buffer
         *  @param n_queries Number of descriptors that will be queried
         *  @param k  Number of maximun neighbors to be returned
         *  @return distances  distances of each neighbor (size n * k).
         *  @return descriptors_ids  distances of each neighbor (size n * k).
         */
        void search(float* query, unsigned n_queries, unsigned k,
                    std::vector<long>& descriptor_ids,
                    std::vector<float>& distances);
        // todo we can typedef the vectors.

        /**
         *  Search for the k closest neighborhs
         *
         *      Add comments that -1 is used for saying that there is no result
         *
         *  @param query  Query descriptors buffer
         *  @param n_queries Number of descriptors that will be queried
         *  @param k  Number of maximun neighbors to be returned
         *  @return distances  distances of each neighbor (size n * k).
         *  @return descriptors_ids  distances of each neighbor (size n * k).
         */
        void search(float* query, unsigned n_queries, unsigned k,
                    long* descriptor_ids, float* distances);

        /**
         *  Search for neighborhs within a radius.
         *  **** This may be scketchy, probable we can leave this
         *  **** for whenever we have a real use case.
         *
         *  @param k  Number of maximun neighbors to be returned
         *  @param query  Query vector
         *  @param radius  maximun distance allowed
         *  @param distances  distances of each neighbor
         */
        void search(float* query, unsigned k, float radius,
                    long* descriptors, float* distances);

        /**
         *  Find the label of the feature vector, based on the closest
         *  neighbors.
         *
         *  @param k  Number of maximun neighbors to be returned
         *  @param query  Query vector
         *  @param radius  maximun distance allowed
         *  @param distances  distances of each neighbor
         */
        std::vector<long> classify(float* descriptors, unsigned n);

        /**
         *  Find the label of the feature vector, based on the closest
         *  neighbors.
         *
         *  @param k  Number of maximun neighbors to be returned
         *  @param query  Query vector
         *  @param radius  maximun distance allowed
         *  @param distances  distances of each neighbor
         */
        void classify(float* descriptors, unsigned n, long* ids);

        /**
         *  Set the matching between label id and the string corresponding
         *  to the label
         *
         *  @param ids  vector of ids of the labels
         *  @param labels  vector of string for each label
         */
        void set_labels(std::vector<long>& ids,
                        std::vector<std::string>& labels);

        /**
         *  Set the matching between label id and the string corresponding
         *  to the label
         *
         *  @param ids  ids of the labels
         *  @param labels  string for each label
         */
        void set_labels(std::map<long, std::string>& labels);

        /**
         *  Get the label of the descriptors for the spcified ids.
         *  NOTE: This is a vector becase this is what we return.
         *  We can, make wrapper functions that recieve arrays as well.
         *
         *  @param ids  vector of descriptor's id
         *  @return vector with the string labels
         */
        std::vector<std::string> get_labels(std::vector<long>& ids);

        /**
         *  Get the label of the descriptors for the spcified ids.
         *  NOTE: This is a vector becase this is what we return.
         *  We can, make wrapper functions that recieve arrays as well.
         *
         *  @param ids  vector of ids
         *  @return vector with the string labels
         */
        std::vector<std::string> label_id_to_string(std::vector<long>& l_id);

        /**
         *  Get the label of the descriptors for the spcified ids.
         *  NOTE: This is a vector becase this is what we return.
         *  We can, make wrapper functions that recieve arrays as well.
         *
         *  @param ids  vector of ids
         *  @return vector with the string labels
         */
        std::vector<std::string> get_labels(long* ids, unsigned n);

        /**
         *  Get the label of the descriptors for the spcified ids.
         *  NOTE: This is a vector becase this is what we return.
         *  We can, make wrapper functions that recieve arrays as well.
         *
         *  @param ids  vector of ids
         *  @return vector with the string labels
         */
        std::map<long, std::string> get_labels();

        /**
         *  Get the label of the descriptors for the spcified ids.
         *  NOTE: This is a vector becase this is what we return.
         *  We can, make wrapper functions that recieve arrays as well.
         *
         *  @param ids  vector of ids of size n
         *  @param descriptors return pointer for the float values (n * d)
         */
        void get_descriptors(std::vector<long>& ids, float* descriptors);

        /**
         *  Get the label of the descriptors for the spcified ids.
         *  NOTE: This is a vector becase this is what we return.
         *  We can, make wrapper functions that recieve arrays as well.
         *
         *  @param ids  buffer with ids
         *  @param n  number of ids to query
         *  @return pointer for the float values (n * dim)
         */
        void get_descriptors(long* ids, unsigned n, float* descriptors);

        /**
         *  Writes the FeatureVector Index to the system. This will overwrite
         *  the original
         */
        void store();

        /**
         *  Writes the FeatureVector Index to the system into a
         *  defined path. This will overwrite any other index under the same
         *  set_path.
         */
        void store(std::string set_path);
    };

};