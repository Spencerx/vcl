/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

// #include "Chrono.h"

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

    // for(int i = 0; i < nb; i++) {
    //     for(int j = 0; j < d; j++)
    //         xb[d * i + j] = drand48();
    //     xb[d * i] += i / 1000.;
    // }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    char filename[]="index.faiss";

    faiss::IndexFlatL2* index;

    if (file_exist(filename))
        index = (faiss::IndexFlatL2*)faiss::read_index(filename);
    else{
        printf("Creating new index\n");
        index = new faiss::IndexFlatL2(d); // call constructor
    }

    printf("is_trained = %s\n", index->is_trained ? "true" : "false");

    for (int what = 0; what < 4; ++what) {
        for(int i = 0; i < nb; i++) {
            for(int j = 0; j < d; j++)
                xb[d * i + j] = drand48();
            xb[d * i] += i / 1000.;
        }
        index->add(nb, xb);
    }                  // add vectors to the index
    printf("ntotal = %ld\n", index->ntotal);

    int k = 10;

    {       // sanity check: search 5 first vectors of xb
        long  *I = new long[k * 5];
        float *D = new float[k * 5];

        printf("sanity searching... \n");

        index->search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    // ChronoCpu chrono("search");

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        printf("searching... \n");

        // chrono.tic();
        index->search(nq, xq, k, D, I);
        // chrono.tac();

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    // chrono.printTotalTime_ms();

    faiss::write_index((faiss::Index*)(index), filename);

    delete [] xb;
    delete [] xq;

    return 0;
}
