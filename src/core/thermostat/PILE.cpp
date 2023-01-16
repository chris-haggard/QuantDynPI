#include "PILE.hpp"

PILE::PILE(
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::DynamicsInput &DynamicsParams,
    std::shared_ptr<RingPolymer> rp, bool thermalization_dynamics)
    : PILE_G(ThermostatParams, DynamicsParams, rp),
      thermostat_centroid(ThermostatParams.centroid_therm),
      thermalizing(thermalization_dynamics) {
  // if it is a production run then don't apply PILE_L to the centroid mode (but
  // PILE_G can still be applied)
  if (!thermalizing) {
    PILE_L::RemoveCentroidThermostat();
  }
}
void PILE::ThermostatStep() {
  // if thermostat centroid is on and not in a thermalization run (where the
  // centroid is thermostatted with PILE_L) then apply PILE_G to the centroid
  if (thermostat_centroid && !thermalizing) {
    PILE_G::ThermostatStep();
  }
  // apply PILE_G first as PILE_L will change action of PILE_G <-- no it won't?
  PILE_L::ThermostatStep();
}
