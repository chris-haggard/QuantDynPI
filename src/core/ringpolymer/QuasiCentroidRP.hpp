#pragma once

#include "AdiabaticRP.hpp"

class QuasiCentroidRP : public AdiabaticRP {
 public:
  QuasiCentroidRP(const InputHandler::RingPolymerInput &, Ensemble &);
  arma::cube qc_position;
  arma::cube qc_momenta;
  arma::cube qc_force;
  arma::cube qc_mass;
  void ScaleMassQC();
  arma::cube TrajectoryVariable() override;
  arma::cube DynamicalVariable() override;
  void PlaceQuasicentroid() override;
};
