#pragma once

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "../../util/NormalDistribution.hpp"
#include "../Input.hpp"
#include "../ringpolymer/RingPolymer.hpp"

class Thermostat {
 private:
  const double seed;

 protected:
  std::shared_ptr<RingPolymer> RP;
  NormalDistribution RandomNormal;
  const double dt;

 public:
  Thermostat(
      const InputHandler::ThermostatInput &,
      const InputHandler::DynamicsInput &, std::shared_ptr<RingPolymer>);
  virtual void ThermostatStep();
};

