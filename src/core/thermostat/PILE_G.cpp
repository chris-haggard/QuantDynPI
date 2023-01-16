#include "PILE_G.hpp"

#include <cmath>

#include "PILE_L.hpp"

PILE_G::PILE_G(
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::DynamicsInput &DynamicsParams,
    std::shared_ptr<RingPolymer> rp)
    : PILE_L(ThermostatParams, DynamicsParams, rp),
      c(std::exp(-dt / ThermostatParams.PILE_tau0)) {
}

/**
 * @brief Equation 54 in 10.1063/1.3489925, where the sum over i to N is a sum
 * over degrees of freedom. i.e. N = 3 * number_of_centroids.
 *
 *
 * @return
 */
double PILE_G::K() {
  double out =
      0.5 * arma::as_scalar(arma::accu(
                arma::square(RP->momenta_nm.row(0)) / RP->mass_scaled.row(0)));
  return out;
}

/**
 * @brief Implementation of Eq. 52 in 10.1063/1.3489925, where the result is a
 * scalar.
 *
 * @return
 */
double PILE_G::alpha() {
  double alpha_sq = c;
  const double K_value = K();
  double out;
  double zeta1 = RandomNormal();
  double zeta2 = 0.0;
  for (arma::uword i = 0; i < RP->position_cart.n_cols; i++) {
    for (arma::uword j = 1; j < RP->position_cart.n_slices; j++) {
      zeta2 += std::pow(RandomNormal(), 2);
    }
  }
  const double temp = (1.0 - c) / (2.0 * RP->beta * K_value);
  alpha_sq += (std::pow(zeta1, 2) * zeta2 * temp);
  alpha_sq += 2.0 * zeta1 * std::sqrt(c * temp);
  out = std::sqrt(alpha_sq);
  // implement Eq 53
  return std::copysign(out, (zeta1 + std::sqrt(c / temp)));
}

/**
 * @brief Apply the PILE_G thermostat to the centroid
 */
void PILE_G::ThermostatStep() {
  const double a = alpha();
  assert(std::isfinite(a));
  RP->momenta_nm.row(0) *= a;
}
