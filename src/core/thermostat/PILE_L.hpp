#pragma once

#include <memory>

#include "Thermostat.hpp"

class PILE_L : public Thermostat {
 protected:
  const double lambda;
  const double tau0;
  arma::cube c1;
  arma::cube c2;
  arma::cube Compute_c1();
  arma::cube Compute_c2();
  void RemoveCentroidThermostat();

 public:
  PILE_L(
      const InputHandler::ThermostatInput &,
      const InputHandler::DynamicsInput &, std::shared_ptr<RingPolymer>);
  void ThermostatStep() override;
};

