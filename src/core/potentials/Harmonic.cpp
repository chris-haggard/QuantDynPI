#include "Harmonic.hpp"

#include "Potential.hpp"

Harmonic::Harmonic(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : Potential(SimParams, PotParams) {
}

/**
 * @brief Harmonic potential 1/2 x^2 with force -x
 *
 * @param position_cart
 *
 * @return -dV/dx i.e. -x
 */
arma::cube Harmonic::Force(const arma::cube &position_cart) {
  arma::cube f(arma::size(position_cart));
  f = -position_cart;
  return f;
}
