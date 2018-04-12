#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <sys/types.h>
#include <dirent.h>

#include "DescriptorsFaiss.h"
#include "Exception.h"

#include <faiss/index_io.h>

#define MAX_DESC_BUFFER 100 // This will be multiplied by two

using namespace VCL;

DescriptorsFaiss::DescriptorsFaiss(const std::string &set_path):
    DescriptorsData(set_path)
{
    _faiss_file = _set_path + "/index.faiss";
    read_maps();
    init_buffers();
}

DescriptorsFaiss::DescriptorsFaiss(const std::string &set_path, unsigned dim)
    : DescriptorsData(set_path, dim)
{
    _faiss_file = _set_path + "/index.faiss";

    // This is probably an unecessary optimization
    // Prevents using find in the add call.
    // _labels_vec.push_back("empty(padding)");
    init_buffers();
}

DescriptorsFaiss::~DescriptorsFaiss()
{
    delete _add_buffer;
}

void DescriptorsFaiss::init_buffers()
{
    _add_buffer = new float[_dimensions*MAX_DESC_BUFFER*2];
    _buffer_elements = 0;
}

void DescriptorsFaiss::flush_buffers()
{
    if (_buffer_elements == 0) return;

    unsigned aux = _buffer_elements;
    // we need this aux because add will call this function forever
    _buffer_elements = 0;
    add(_add_buffer, aux, _labels_buffer.data());
    _labels_buffer.clear();
}

void DescriptorsFaiss::write_maps()
{
    std::ofstream out_ids(_set_path+"/ids.arr", std::ofstream::binary);

    unsigned ids_size = _ids_vec.size();
    out_ids.write((char*)&ids_size, sizeof(ids_size));
    out_ids.write((char*)_ids_vec.data(), sizeof(long) * ids_size);
    out_ids.close();

    std::ofstream out_labels(_set_path+"/labels.txt");
    for (auto& label : _labels_map) {
        out_labels << label.first << " " << label.second << std::endl;
    }
    out_labels.close();
}

void DescriptorsFaiss::read_maps()
{
    std::ifstream in_ids(_set_path+"/ids.arr", std::ofstream::binary);

    unsigned ids_size;
    in_ids.read((char*)&ids_size, sizeof(ids_size));
    _ids_vec.resize(ids_size);
    in_ids.read((char*)_ids_vec.data(), sizeof(long) * ids_size);
    in_ids.close();

    std::ifstream in_labels(_set_path+"/labels.txt");
    std::string str;
    _labels_map.clear();
    while (std::getline(in_labels, str)) {
        std::stringstream sstr(str);
        long id;
        sstr >> id;
        sstr >> str;
        _labels_map[id] = str;
    }
    in_labels.close();
}

void DescriptorsFaiss::add(float* descriptors, unsigned n, long* labels)
{
    assert(n > 0);
    // if (_buffer_elements > MAX_DESC_BUFFER) {
    //     flush_buffers();
    // }

    // if (n < MAX_DESC_BUFFER) {

    //     _lock.lock();
    //     float* copy_dst = _add_buffer + _dimensions * _buffer_elements;
    //     size_t copy_bytes = n * _dimensions * sizeof(float);
    //     std::memcpy((void*)copy_dst, descriptors, copy_bytes);

    //     _labels_buffer.insert(_labels_buffer.begin()+_labels_buffer.size(),
    //                           labels.begin(), labels.end());
    //     _buffer_elements += n;

    //     _lock.unlock(); // this is very important
    //     return;
    // }

    _lock.lock();
    if (!_index->is_trained) {
        // This is needed for IVFFlat, we should handle this differently
        _index->train(n, descriptors);
    }

    if (labels != NULL) {
        // printf("n: %d, total: %ld\n", 1, _index->ntotal);
        _ids_vec.resize(_index->ntotal + n);
        long* dst = _ids_vec.data() + _index->ntotal;
        std::memcpy(dst, labels, n * sizeof(long));
        // for (int i = 0; i < n; ++i) {
        //     _ids_vec.push_back(labels[i]);
        // }
    }

    _index->add(n, descriptors);
    _lock.unlock();
    store();
}

void DescriptorsFaiss::train()
{
    _lock.lock();
    // _index->train(nb, xb);
    _lock.unlock();
}

bool DescriptorsFaiss::is_trained()
{
    return _index->is_trained;
}

// void DescriptorsFaiss::search(int k, float* query,
//                          float radius,
//                          float* descriptors,
//                          std::vector<float> &distances)
// {
//     _index->search(k, query, )
// }

void DescriptorsFaiss::search(float* query, unsigned n_queries, unsigned k,
                              long* descriptors, float* distances)
{
    flush_buffers();

    _lock.lock();
    _index->search(n_queries, query, k, distances, descriptors);
    _lock.unlock();

    // Think about data types better.
    // long on IA64/linux are of size 8.
    // std::memcpy(descriptors.data(), ids, k*n_queries*sizeof(int));
    // This conversion is necessary, buy I don't like it.
    // for (int i = 0; i < k*n_queries; ++i) {
    //     descriptors[i] = (int)ids[i];
    // }
}

void DescriptorsFaiss::classify(float* descriptors, unsigned n, long* ids)
{
    flush_buffers();

    //TODO how we add the voting number
    int quorum = 7;
    float* distances = new float[n * quorum];
    long*  ids_aux   = new long [n * quorum];

    search(descriptors, n, quorum, ids_aux, distances);

    _lock.lock();
    for (int j = 0; j < n; ++j) {
        std::map<long, int> map_voting;
        long winner = 0;
        unsigned max = 0;
        for (int i = 0; i < quorum; ++i) {
            long idx = ids_aux[quorum*j + i];
            if (idx < 0) continue; // Means not found

            assert(idx < _ids_vec.size());
            long label_id = _ids_vec.at(idx);
            // printf("idx: %ld\n", idx);
            // printf("label_id: %ld\n", label_id);
            map_voting[label_id] += 1;
            if (max < map_voting[label_id]) {
                max = map_voting[label_id];
                winner = label_id;
            }
        }
        // printf("winner: %ld\n", winner);
        ids[j] = winner;
    }
    _lock.unlock();
}

std::vector<std::string> DescriptorsFaiss::get_labels(long* ids, unsigned n)
{
    // flush_buffers();
    _lock.lock();
    std::vector<std::string> ret_labels(n);

    for (int i = 0; i < n; ++i) {
        long idx = ids[i];
        if (idx > _ids_vec.size()){
            throw VCLException(ObjectNotFound, "Label id does not exists");
        }
        ret_labels[i] = _labels_map[_ids_vec[idx]];
    }
    _lock.unlock();

    return ret_labels;
}

void DescriptorsFaiss::set_labels(std::map<long,std::string>& labels) {
    _lock.lock();
    _labels_map = labels;
    _lock.unlock();
}

void DescriptorsFaiss::get_descriptors(long* ids, unsigned n,
                                       float* descriptors)
{
    flush_buffers();
    _lock.lock();
    std::vector<std::string> ret_classes;
    int offset = 0;
    for (int i = 0; i < n; ++i) {
        _index->reconstruct(ids[i], descriptors+offset*_dimensions);
        ++offset;
    }

    _lock.unlock();
}

void DescriptorsFaiss::store()
{
    store(_set_path);
}

void DescriptorsFaiss::store(std::string set_path)
{
    flush_buffers();
    _lock.lock();
    _set_path = set_path;
    _faiss_file = _set_path + "/index.faiss";

    DIR* faiss_dir = opendir(_set_path.c_str());
    if (faiss_dir) { /* Directory exists. */
        closedir(faiss_dir);
    }
    else {
        int ret = system(std::string("mkdir " + _set_path).c_str());
        if (ret != 0)
            throw VCLException(OpenFailed, set_path + " cannot create");
    }

    faiss::write_index((const faiss::IndexFlat*)(_index), _faiss_file.c_str());
    write_maps();
    _lock.unlock();
}

// DescriptorsFaissFlatL2

DescriptorsFaissFlatL2::DescriptorsFaissFlatL2(const std::string &set_path):
    DescriptorsFaiss(set_path)
{
    if (file_exist(_faiss_file)){
        _index = (faiss::IndexFlatL2*)faiss::read_index(_faiss_file.c_str());
        if (!_index) {
            throw VCLException(OpenFailed, "Problem reading: " + _faiss_file);
        }

        _dimensions = _index->d;
    }
    else{
        throw VCLException(OpenFailed, set_path + " does not exists");
    }
}

DescriptorsFaissFlatL2::DescriptorsFaissFlatL2(const std::string &set_path, unsigned dim)
    : DescriptorsFaiss(set_path, dim)
{
    if (file_exist(_faiss_file)){
        throw VCLException(OpenFailed, set_path + " already exists");
    }
    else{
        _index = new faiss::IndexFlatL2(_dimensions);
    }
}

// DescriptorsFaissIVFFlatL2

DescriptorsFaissIVFFlatL2::DescriptorsFaissIVFFlatL2(const std::string &set_path):
    DescriptorsFaiss(set_path)
{
    if (file_exist(_faiss_file)){
        _index = (faiss::IndexIVFFlat*)faiss::read_index(_faiss_file.c_str());
        if (!_index)
            throw VCLException(OpenFailed, "Problem reading: " + _faiss_file);

        _dimensions = _index->d;

        if (!_index->is_trained)
            std::cerr << "WARNING: Index Not Trained" << std::endl;
    }
    else{
        throw VCLException(OpenFailed, set_path + " does not exists");
    }
}

DescriptorsFaissIVFFlatL2::DescriptorsFaissIVFFlatL2(const std::string &set_path, unsigned dim)
    : DescriptorsFaiss(set_path, dim)
{
    if (file_exist(_faiss_file)){
        throw VCLException(OpenFailed, set_path + " already exists");
    }
    else{
        faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(_dimensions);

        // TODO revise nlist param
        int nlist = 4;
        _index = new faiss::IndexIVFFlat(quantizer, _dimensions,
                                         nlist, faiss::METRIC_L2);
                                         // call constructor

        _dimensions = _index->d;
    }
}
