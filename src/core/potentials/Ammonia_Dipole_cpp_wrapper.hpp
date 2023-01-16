#pragma once
#include <vector>

// interfaces with fortran file
extern "C" void AQZfc_dipole_wrapper(double[], double[], int*, int*);
