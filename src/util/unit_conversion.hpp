#pragma once

#define boltzmann_constant 1.38064852e-23
#define fs_to_au 41.341374575751
#define joule_to_au 2.2937126583579e17
#define dalton_to_kg 1.6605390666050e-27
#define kg_to_au 1.0977691e30

double T_to_beta(const double);
double beta_to_T(const double);
double fs_au(const double);
double au_fs(const double);
double dalton_to_au(const double);
