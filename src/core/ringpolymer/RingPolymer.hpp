#pragma once
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "../../util/AtomData.hpp"
#include "../Ensemble.hpp"
#include "../Input.hpp"
#include "../normal_modes/NormalModes.hpp"

class RingPolymer {
 public:
  RingPolymer(const InputHandler::RingPolymerInput &, Ensemble &);
  const std::vector<std::string> labels;
  const unsigned int n_beads;
  const unsigned int n_atoms;
  const unsigned int n_molec;
  const unsigned int n_total_atoms;
  const double beta;
  const double beta_n;
  arma::cube position_cart;
  arma::cube position_nm;
  arma::cube momenta_cart;
  arma::cube momenta_nm;
  arma::cube force_cart;
  arma::cube force_nm;
  const arma::cube mass_atomic;
  arma::cube mass_atomic_sqrt;
  arma::cube mass_scaled;
  arma::cube mass_scaled_sqrt;
  arma::cube freq;
  arma::cube freq_scaled;
  NormalModes NMConv;
  arma::cube Centroid();
  virtual arma::cube TrajectoryVariable();
  virtual arma::cube DynamicalVariable();
  virtual void PlaceQuasicentroid();

 protected:
  arma::cube FillInitialMomentaNM();

 private:
  arma::cube FillAtomicMass();
  arma::cube FillFreq();
};
