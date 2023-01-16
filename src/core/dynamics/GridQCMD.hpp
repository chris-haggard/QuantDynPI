#pragma once

#include "QCDynamics.hpp"

class GridQCMD : public QCDynamics {
 public:
  GridQCMD(
      const InputHandler::DynamicsInput &,
      const InputHandler::ThermostatInput &,
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &, std::shared_ptr<RingPolymer>);
  void ThermStep() override;
  void Step() override;

 private:
  arma::vec r_grid, theta_grid;
};
