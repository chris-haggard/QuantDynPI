#pragma once

#include "PILE_G.hpp"

/**
 * @brief Apply PILE_L to non-centroid modes and PILE_G to centroid mode
 */
class PILE : public PILE_G {
 private:
  const bool thermalizing;
  const bool thermostat_centroid;

 public:
  PILE(
      const InputHandler::ThermostatInput &,
      const InputHandler::DynamicsInput &, std::shared_ptr<RingPolymer>, bool);
  void ThermostatStep() override;
};
