#include "Thermostat.hpp"

#include <stdexcept>
/**
 * @brief Base class for thermostats. Creates and seeds the RandomNormal object.
 * Constructed by default in the Dynamics class and then overwritten by using
 * PILE_L etc. Has a virtual ThermostatStep function that derived classes
 * override.
 *
 * @param ThermostatParams
 * @param RingPolymerParams
 * @param DynamicsParams
 */
Thermostat::Thermostat(
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::DynamicsInput &DynamicsParams,
    std::shared_ptr<RingPolymer> rp)
    : seed(ThermostatParams.seed), RandomNormal(seed), dt(DynamicsParams.dt) {
  RP = rp;
}

void Thermostat::ThermostatStep() {
  throw std::runtime_error("No ThermostatStep() calculation in base class\n");
}

