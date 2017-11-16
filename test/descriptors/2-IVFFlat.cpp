/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>

// #include "Chrono.h"
#include "HybridCNNReader.h"

inline bool file_exist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main() {
    int d = 4096;                            // dimension
    int nb = 100000;                       // database size
    int nq = 100;                        // nb of queries

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    int nlist = 100;
    int k = 10;

    char filename[] = "indexivf.faiss";

    faiss::IndexIVFFlat* index;
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(d);

    // ChronoCpu readtime("readtime");
    // readtime.tic();
    if ( file_exist(filename) )
        index = (faiss::IndexIVFFlat*)faiss::read_index(filename, true);
    else{
        printf("Creating new index\n");
        index = new faiss::IndexIVFFlat(quantizer, d,
                                        nlist, faiss::METRIC_L2);
    }
    // readtime.tac();

    // readtime.printTotalTime_ms();

    // here we specify METRIC_L2, by default it performs inner-product search
    // assert(!index->is_trained);
    // assert(index->is_trained);
    printf("ntotal before add = %ld\n", index->ntotal);

    HybridCNNReader yfcc("/mnt/datasets/hybridCNN_gmean_fc6_34.txt");
    // ChronoCpu read_nb("read_nb");
    // read_nb.tic();
    yfcc.read(nb, xb, nb*4*d);
    // read_nb.tac();
    // read_nb.printTotalTime_ms();

    index->train(nb, xb);

    for (int what = 0; what < 3; ++what) {

        yfcc.read(nb, xb, nb*4*d);
        index->add(nb, xb);
        // for(int i = 0; i < nb; i++) {
        //     for(int j = 0; j < d; j++)
        //         xb[d * i + j] = drand48();
        //     xb[d * i] += i / 1000.;
        // }
    }

    printf("ntotal = %ld\n", index->ntotal);

    // ChronoCpu chrono("search");
    printf("done inserting...\n");

    {   // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        printf("Searching... \n");

        // chrono.tic();
        index->search(nq, xq, k, D, I);
        // chrono.tac();

        printf("I=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        index->nprobe = 10;
        index->search(nq, xq, k, D, I);

        printf("I=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    // chrono.printTotalTime_ms();

    // ChronoCpu writetime("writetime");
    // writetime.tic();
    // faiss::write_index((const faiss::IndexFlat*)(index), filename);
    // writetime.tac();

    // writetime.printTotalTime_ms();

    delete [] xb;
    delete [] xq;

    delete quantizer;

    return 0;
}
