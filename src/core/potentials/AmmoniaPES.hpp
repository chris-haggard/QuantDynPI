#pragma once

#include "Potential.hpp"

// extern "C" void dipole_wrapper(std::vector<double>);

class AmmoniaPES : public Potential {
 public:
  AmmoniaPES(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  arma::cube Force(const arma::cube &) override;
  arma::vec Pot(const arma::cube &) override;
  arma::cube Dipole(const arma::cube &) override;
};

