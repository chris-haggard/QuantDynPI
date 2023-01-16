#include "unit_conversion.hpp"

double T_to_beta(const double T) {
  return (1.0 / (boltzmann_constant * T * joule_to_au));
}

double beta_to_T(const double b) {
  return T_to_beta(b);
}

double fs_au(const double fs) {
  return fs * fs_to_au;
}

double au_fs(const double au) {
  return (au * (1.0 / fs_to_au));
}

double dalton_to_au(const double dalton) {
  return dalton * dalton_to_kg * kg_to_au;
}
