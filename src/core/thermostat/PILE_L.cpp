#include "PILE_L.hpp"

#include <memory>

#include "Thermostat.hpp"

/**
 * @brief Construct the PILE_L thermostat. By default all modes are
 * thermostatted, as required in the pre-run thermalization of TRPMD and CMD.
 * For production runs, in which the centroid mode should not be thermostatted,
 * the c1 and c2 cubes are altered to remove the thermostattig of the centroid
 * mode. The PILE_G thermostat can be applied to the centroid mode in the
 * production run.
 *
 *
 * @param ThermostatParams
 * @param DynamicsParams
 * @param rp
 * @param therm_centroid
 */
PILE_L::PILE_L(
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::DynamicsInput &DynamicsParams,
    std::shared_ptr<RingPolymer> rp)
    : Thermostat(ThermostatParams, DynamicsParams, rp),
      lambda(ThermostatParams.PILE_L_lambda),
      tau0(ThermostatParams.PILE_tau0) {
  c1 = Compute_c1();
  c2 = Compute_c2();
}

arma::cube PILE_L::Compute_c1() {
  arma::cube c(arma::size(RP->momenta_nm));
  c.fill(-dt / 2.0);
  if (c.n_rows > 1) {
    c.subcube(1, 0, 0, c.n_rows - 1, c.n_cols - 1, c.n_slices - 1) %=
        2.0 * lambda *
        RP->freq_scaled.subcube(
            1, 0, 0, c.n_rows - 1, c.n_cols - 1, c.n_slices - 1);
  }

  c.row(0) *= 1.0 / tau0;
  c = arma::exp(c);

  assert(arma::is_finite(c));
  return c;
}

arma::cube PILE_L::Compute_c2() {
  arma::cube c(arma::size(RP->momenta_nm));
  c = arma::sqrt(1.0 - arma::square(c1));
  assert(arma::is_finite(c));
  return c;
}

void PILE_L::RemoveCentroidThermostat() {
  c1.row(0).fill(1.0);
  c2.row(0).fill(0.0);
}

void PILE_L::ThermostatStep() {
  assert(arma::size(c1) == arma::size(c2));
  assert(arma::size(c1) == arma::size(RP->momenta_nm));

  arma::cube rand_cube(arma::size(c2));
  rand_cube.imbue([&]() { return RandomNormal(); });

  RP->momenta_nm %= c1;
  RP->momenta_nm +=
      (c2 % rand_cube % RP->mass_scaled_sqrt * (1.0 / std::sqrt(RP->beta)));
}
