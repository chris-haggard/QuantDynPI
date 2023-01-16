#include "AdiabaticRP.hpp"
#include <math.h>

AdiabaticRP::AdiabaticRP(
    const InputHandler::RingPolymerInput &RingPolymerParams, Ensemble &ens)
    : RingPolymer(RingPolymerParams, ens), gamma(RingPolymerParams.gamma) {
  if (gamma > 0.0) {
    ScaleFreq();
    ScaleMass();
  }
  // mass of normal modes has changed
  momenta_nm = FillInitialMomentaNM();
}

/**
 * @brief Scale the freq on the normal modes, not including the zeroth normal
 * mode (the centroid)
 */
void AdiabaticRP::ScaleFreq() {
  double omega = (gamma / beta_n);
  freq_scaled.rows(1, freq_scaled.n_rows - 1).fill(omega);
}

/**
 * @brief Scale the masses such that mass_scaled * freq_scaled**2 recovers
 * mass_atomic * freq**2
 */
void AdiabaticRP::ScaleMass() {
  assert(arma::approx_equal(mass_atomic, mass_scaled, "abs", 1e-14));
  mass_scaled.rows(1, mass_scaled.n_rows - 1) %= arma::square(
      (freq.rows(1, mass_scaled.n_rows - 1) /
       freq_scaled.rows(1, mass_scaled.n_rows - 1)));
  mass_scaled_sqrt = arma::sqrt(mass_scaled);
}

arma::cube AdiabaticRP::TrajectoryVariable() {
  return Centroid();
}

arma::cube AdiabaticRP::DynamicalVariable() {
  return Centroid();
}
