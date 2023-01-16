#pragma once

#include "RingPolymer.hpp"

class AdiabaticRP : public RingPolymer {
 public:
  AdiabaticRP(const InputHandler::RingPolymerInput &, Ensemble &);
  arma::cube TrajectoryVariable() override;
  arma::cube DynamicalVariable() override;

 protected:
  double gamma;

 private:
  void ScaleMass();
  void ScaleFreq();
};
