#include "Potential.hpp"

#include <cmath>
#include <stdexcept>

Potential::Potential(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : box_length(SimParams.box_length),
      box_length_reciprocal(1.0 / box_length) {
  assert(std::isfinite(box_length) && std::isfinite(box_length_reciprocal));
}

arma::cube Potential::Force(const arma::cube &position_cart) {
  throw std::runtime_error("No force calculation in base class\n");
}

arma::vec Potential::Pot(const arma::cube &position_cart) {
  throw std::runtime_error("No potential calculation in base class\n");
}

arma::cube Potential::Dipole(const arma::cube &position_cart) {
  throw std::runtime_error("No dipole calculation in base class\n");
}
