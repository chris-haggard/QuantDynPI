#pragma once

#include <memory>

#include "Dynamics.hpp"

class ClassicalDynamics : public Dynamics {
 public:
  ClassicalDynamics(
      const InputHandler::DynamicsInput &,
      const InputHandler::ThermostatInput &,
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &, std::shared_ptr<RingPolymer>);
  void ThermStep() override;
  void Step() override;
};

