#pragma once

#include "PILE_L.hpp"

/**
 * @brief Implementation of the Global PILE thermostat (PILE_G) from
 * 10.1063/1.3489925
 */
class PILE_G : public PILE_L {
 private:
  double c;

 public:
  PILE_G(
      const InputHandler::ThermostatInput &,
      const InputHandler::DynamicsInput &, std::shared_ptr<RingPolymer>);
  double K();
  double alpha();
  void ThermostatStep() override;
};
