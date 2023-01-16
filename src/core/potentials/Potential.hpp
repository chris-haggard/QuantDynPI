#pragma once
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>

#include "../Input.hpp"

class Potential {
 protected:
  const double box_length;
  const double box_length_reciprocal;

 public:
  Potential(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  virtual arma::cube Force(const arma::cube &);
  virtual arma::vec Pot(const arma::cube &);
  virtual arma::cube Dipole(const arma::cube &);
};
