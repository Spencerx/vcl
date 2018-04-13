#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>

#include "VCL.h"
#include "helpers.h"

void generate_desc_linear_increase(int d, int nb, float* xb, float init)
{
    float val = init;
    for (int i = 1; i <= nb*d; ++i) {
        xb[i-1] = val;
        if ( i%d == 0) val++;
    }
}

float* generate_desc_linear_increase(int d, int nb, float init)
{
    float *xb = new float[d * nb];
    generate_desc_linear_increase(d, nb, xb, 0);
    return xb;
}

std::map<long, std::string> animals_map()
{
    std::map<long, std::string> class_map;
    class_map[0] = "no_class";
    class_map[1] = "dog";
    class_map[2] = "cat";
    class_map[3] = "messi";
    class_map[4] = "bird";
    class_map[5] = "condor";
    class_map[6] = "panda";

    return class_map;
}

std::vector<long> classes_increasing_offset(unsigned nb, unsigned offset)
{
    std::vector<long> classes(nb, 0);

    for (int i = 0; i < offset; ++i) {
        classes[i] = 1;
        classes[i+offset] = 2;
        classes[i+2*offset] = 3;
        classes[i+3*offset] = 4;
        classes[i+4*offset] = 5;
        classes[i+5*offset] = 6;
    }
    return classes;
}


