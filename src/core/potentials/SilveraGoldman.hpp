#pragma once

#include "Potential.hpp"

class SilveraGoldman : public Potential {
 private:
  const double a = 1.713;
  const double b = 1.5671;
  const double g = 0.00993;
  const double rc = 8.32;

  const double C_6 = 12.14;
  const double C_8 = 215.2;
  const double C_9 = 143.1;
  const double C_10 = 4813.9;

  const double r_cutoff = 15.0;

  double SGForceMagnitude(const double r);

 public:
  SilveraGoldman(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  arma::cube Force(const arma::cube &) override;
};
