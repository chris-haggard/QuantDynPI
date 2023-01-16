#pragma once

#include "Potential.hpp"

class PartridgeSchwenke : public Potential {
 public:
  PartridgeSchwenke(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  arma::cube Force(const arma::cube &) override;
  arma::vec Pot(const arma::cube &) override;
  arma::cube Dipole(const arma::cube &) override;
};
