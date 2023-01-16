#include "ClassicalDynamics.hpp"

#include <memory>
#include <stdexcept>

#include "../../util/RingPolymerUtils.hpp"
#include "Dynamics.hpp"

ClassicalDynamics::ClassicalDynamics(
    const InputHandler::DynamicsInput &DynamicsParams,
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotentialParams,
    std::shared_ptr<RingPolymer> rp)
    : Dynamics(
          DynamicsParams, ThermostatParams, SimParams, PotentialParams, rp) {
  assert(RP->position_cart.n_rows == 1);
}

void ClassicalDynamics::ThermStep() {
  ThermThermostat->ThermostatStep();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  UpdatePosition(RP->position_nm, RP->momenta_nm, RP->mass_atomic);
  UpdateExternalForce();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  ThermThermostat->ThermostatStep();
}

/**
 * @brief Propagate for one timestep without the action of a thermostat i.e.
 * RPMD run
 */
void ClassicalDynamics::Step() {
  RunThermostat->ThermostatStep();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  UpdatePosition(RP->position_nm, RP->momenta_nm, RP->mass_atomic);
  UpdateExternalForce();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  RunThermostat->ThermostatStep();
}
