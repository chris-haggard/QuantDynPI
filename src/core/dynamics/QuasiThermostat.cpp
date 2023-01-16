#include "QuasiThermostat.hpp"

#include <cmath>

QuasiThermostat::QuasiThermostat(
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::DynamicsInput &DynamicsParams,
    std::shared_ptr<QuasiCentroidRP> qc_rp)
    : RandomNormal(ThermostatParams.seed),
      c(std::exp(-DynamicsParams.dt / ThermostatParams.Quasi_tau0)) {
  QC_RP = qc_rp;
}

/**
 * @brief Equation 54 in 10.1063/1.3489925, where the sum over i to N is a sum
 * over degrees of freedom. i.e. N = 3 * number_of_centroids.
 *
 *
 * @return
 */
double QuasiThermostat::K() {
  double out = 0.5 * arma::as_scalar(arma::accu(
                         arma::square(QC_RP->qc_momenta) / QC_RP->qc_mass));
  return out;
}

/**
 * @brief Implementation of Eq. 52 in 10.1063/1.3489925, where the result is a
 * scalar.
 *
 * @return
 */
double QuasiThermostat::alpha() {
  double alpha_sq = c;
  const double K_value = K();
  double out;
  double zeta1 = RandomNormal();
  double zeta2 = 0.0;
  for (arma::uword i = 0; i < QC_RP->qc_momenta.n_cols; i++) {
    for (arma::uword j = 1; j < QC_RP->qc_momenta.n_slices; j++) {
      zeta2 += std::pow(RandomNormal(), 2);
    }
  }
  const double temp =
      (1.0 - c) /
      (2.0 * (QC_RP->beta) *
       K_value);  // Beta_n --> Beta as not a path-integral thermostat
  alpha_sq += (std::pow(zeta1, 2) * zeta2 * temp);
  alpha_sq += 2.0 * zeta1 * std::sqrt(c * temp);
  out = std::sqrt(alpha_sq);
  return std::copysign(out, (zeta1 + std::sqrt(c / temp)));
}

/**
 * @brief Apply the PILE_G thermostat to the centroid
 */
void QuasiThermostat::ThermostatStep() {
  const double a = alpha();
  assert(std::isfinite(a));
  QC_RP->qc_momenta *= a;
}
