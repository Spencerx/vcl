#pragma once

#include <cstdio>
#include <cstdlib>

void generate_desc_linear_increase(int d, int nb, float* xb, float init = 0);

float* generate_desc_linear_increase(int d, int nb, float init = 0);

void check_arrays_float(float* a, float* b, int d);

std::map<long, std::string> animals_map();

std::vector<long> classes_increasing_offset(unsigned nb, unsigned offset);
