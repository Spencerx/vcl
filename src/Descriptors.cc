#include <stdlib.h>
#include <string>
#include <iostream>

#include "Descriptors.h"
#include "DescriptorsFaiss.h"
#include "DescriptorsTileDB.h"

namespace VCL {

Descriptors::Descriptors(const std::string &set_path, DescriptorsEngine eng)
{
    // TODO, change based on format or file header.
    if (eng == DescriptorsEngine(FaissFlatL2))
        _descriptors = new DescriptorsFaissFlatL2(set_path);
    else if (eng == DescriptorsEngine(FaissIVFFlatL2))
        _descriptors = new DescriptorsFaissIVFFlatL2(set_path);
    else if (eng == DescriptorsEngine(TileDBDense))
        _descriptors = new DescriptorsTileDBDense(set_path);
    else if (eng == DescriptorsEngine(TileDBSparse))
        _descriptors = new DescriptorsTileDBSparse(set_path);
    else {
        std::cerr << "Index Not supported" << std::endl;
        throw VCLException(UnsupportedIndex, "Index not supported");
    }
}

Descriptors::Descriptors(const std::string &set_path,
                         unsigned dim,
                         DescriptorsEngine eng)
{
    if (eng == DescriptorsEngine(FaissFlatL2))
        _descriptors = new DescriptorsFaissFlatL2(set_path, dim);
    else if (eng == DescriptorsEngine(FaissIVFFlatL2))
        _descriptors = new DescriptorsFaissIVFFlatL2(set_path, dim);
    else if (eng == DescriptorsEngine(TileDBDense))
        _descriptors = new DescriptorsTileDBDense(set_path, dim);
    else if (eng == DescriptorsEngine(TileDBSparse))
        _descriptors = new DescriptorsTileDBSparse(set_path, dim);
    else {
        std::cerr << "Index Not supported" << std::endl;
        throw VCLException(UnsupportedIndex, "Index not supported");
    }
}

Descriptors::~Descriptors()
{
}

void Descriptors::add(float* descriptors, unsigned n,
                      std::vector<long>& labels)
{
    if (n != labels.size() && labels.size() != 0)
        throw VCLException(SizeMismatch, "labels size wrong");

    add(descriptors, n, labels.size() > 0 ? (long*) labels.data() : NULL);
}

void Descriptors::add(float* descriptors, unsigned n, long* labels)
{
    _descriptors->add(descriptors, n, labels);
}

void Descriptors::train()
{
    _descriptors->train();
}

bool Descriptors::is_trained()
{
    _descriptors->is_trained();
}

void Descriptors::search(float* queries, unsigned n_queries, unsigned k,
                        std::vector<long>& descriptors,
                        std::vector<float>& distances)
{
    descriptors.resize(n_queries * k);
    distances.resize(n_queries * k);
    search(queries, n_queries, k, descriptors.data(), distances.data());
}

void Descriptors::search(float* queries, unsigned n_queries, unsigned k,
                         long* descriptors, float* distances)
{
    _descriptors->search(queries, n_queries, k, descriptors, distances);
}

std::vector<long> Descriptors::classify(float* descriptors,
                                        unsigned n)
{
    std::vector<long> ids;
    ids.resize(n);
    classify(descriptors, n, ids.data());
    return ids;
}

void Descriptors::classify(float* descriptors, unsigned n, long* ids)
{
    _descriptors->classify(descriptors, n, ids);
}

void Descriptors::set_labels(std::vector<long>& ids,
                             std::vector<std::string>& labels)
{
    return _descriptors->set_labels(ids, labels);
}

void Descriptors::set_labels(std::map<long, std::string>& labels)
{
    return _descriptors->set_labels(labels);
}

std::vector<std::string> Descriptors::label_id_to_string(
                            std::vector<long>& l_id)
{
    std::vector<std::string> ret_labels(l_id.size());
    std::map<long, std::string> labels_map = _descriptors->get_labels();

    for (int i = 0; i < l_id.size(); ++i) {
        ret_labels[i] = labels_map[l_id[i]];
    }
    return ret_labels;
}

std::map<long, std::string> Descriptors::get_labels()
{
    return _descriptors->get_labels();
}

std::vector<std::string> Descriptors::get_labels(std::vector<long>& ids)
{
    return _descriptors->get_labels(ids.data(), ids.size());
}

void Descriptors::get_descriptors(std::vector<long>& ids, float* descriptors)
{
    get_descriptors(ids.data(), ids.size(), descriptors);
}

void Descriptors::get_descriptors(long* ids, unsigned n, float* descriptors)
{
    _descriptors->get_descriptors(ids, n, descriptors);
}

void Descriptors::store()
{
    _descriptors->store();
}

void Descriptors::store(std::string set_path)
{
    _descriptors->store(set_path);
}

}
