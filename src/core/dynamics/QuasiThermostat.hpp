#pragma once
#include "../../util/NormalDistribution.hpp"
#include "../Input.hpp"
#include "../ringpolymer/QuasiCentroidRP.hpp"

/**
 * @brief Implementation of the Global PILE thermostat (PILE_G) from
 * 10.1063/1.3489925
 */
class QuasiThermostat {
 private:
  double c;
  NormalDistribution RandomNormal;

 public:
  QuasiThermostat(
      const InputHandler::ThermostatInput &,
      const InputHandler::DynamicsInput &, std::shared_ptr<QuasiCentroidRP>);
  std::shared_ptr<QuasiCentroidRP> QC_RP;
  double K();
  double alpha();
  void ThermostatStep();
};
