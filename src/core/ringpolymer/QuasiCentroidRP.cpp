#include "QuasiCentroidRP.hpp"

#include "../../util/RingPolymerUtils.hpp"

QuasiCentroidRP::QuasiCentroidRP(
    const InputHandler::RingPolymerInput &RingPolymerParams, Ensemble &ens)
    : AdiabaticRP(RingPolymerParams, ens),
      qc_position(1, 3, position_cart.n_slices),
      qc_momenta(arma::size(qc_position), arma::fill::zeros),
      qc_force(arma::size(qc_position), arma::fill::zeros),
      qc_mass(arma::size(qc_position)) {
  qc_position = Centroid();

  if (gamma > 0.0) {
    ScaleMassQC();
  }

  for (arma::uword i = 0; i < position_cart.n_slices; i++) {
    qc_mass.slice(i) = mass_atomic.slice(i).row(0);
  }

  for (arma::uword k = 0; k < qc_momenta.n_slices; k++) {
    qc_momenta.slice(k) =
        InitializeMomenta(qc_momenta.slice(k), qc_mass.slice(k), beta);
  }

  // mass of centroid mode has changed
  momenta_nm = FillInitialMomentaNM();
  momenta_nm.print("inital momenta in qc_rp constructor");

  std::cout << "calling quasi\n";
  qc_mass.print("qc_mass in constructor");
  mass_scaled.print(
      "mass scaled in QC constructor, should be the same as atomic mass");
}

/**
 * @brief Scale the centroid mass by gamma^-2
 */
void QuasiCentroidRP::ScaleMassQC() {
  mass_scaled.row(0) /= (gamma * gamma);
  mass_scaled_sqrt = arma::sqrt(mass_scaled);
}

arma::cube QuasiCentroidRP::TrajectoryVariable() {
  return qc_position;
}

arma::cube QuasiCentroidRP::DynamicalVariable() {
  return qc_position;
}

void QuasiCentroidRP::PlaceQuasicentroid() {
  qc_position = arma::mean(position_cart, 0);
}
