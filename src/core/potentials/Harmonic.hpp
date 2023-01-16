#pragma once

#include "Potential.hpp"

class Harmonic : public Potential {
 public:
  Harmonic(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  arma::cube Force(const arma::cube &) override;
};
