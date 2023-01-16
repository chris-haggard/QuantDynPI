#include "Simulation.hpp"

#include <memory>

#include "dynamics/ClassicalDynamics.hpp"
#include "dynamics/QCDynamics.hpp"
#include "ringpolymer/AdiabaticRP.hpp"
#include "ringpolymer/QuasiCentroidRP.hpp"

Simulation::Simulation(const InputHandler &InputParams, Ensemble &ens)
    : RP(std::make_shared<RingPolymer>(InputParams.RingPolymerParameters, ens)),
      n_traj(InputParams.SimulationParameters.n_traj),
      initial_therm_length(
          InputParams.SimulationParameters.initial_therm_length),
      therm_length(InputParams.SimulationParameters.therm_length),
      run_length(InputParams.SimulationParameters.run_length),
      stride(InputParams.SimulationParameters.stride) {
  if (InputParams.SimulationParameters.sim_type == "CMD") {
    RP = std::make_shared<AdiabaticRP>(InputParams.RingPolymerParameters, ens);
  } else if (InputParams.SimulationParameters.sim_type == "QCMD") {
    RP = std::make_shared<QuasiCentroidRP>(
        InputParams.RingPolymerParameters, ens);
  }
  if (InputParams.SimulationParameters.sim_type == "QCMD") {
    Dyn = std::make_unique<QCDynamics>(
        InputParams.DynamicsParameters, InputParams.ThermostatParameters,
        InputParams.SimulationParameters, InputParams.PotentialParameters, RP);
  } else if (InputParams.SimulationParameters.sim_type == "Classical") {
    Dyn = std::make_unique<ClassicalDynamics>(
        InputParams.DynamicsParameters, InputParams.ThermostatParameters,
        InputParams.SimulationParameters, InputParams.PotentialParameters, RP);
  } else {
    Dyn = std::make_unique<Dynamics>(
        InputParams.DynamicsParameters, InputParams.ThermostatParameters,
        InputParams.SimulationParameters, InputParams.PotentialParameters, RP);
  }
}

Simulation::Simulation(
    const InputHandler &InputParams, Ensemble &ens, std::string sim_type)
    : RP(std::make_shared<RingPolymer>(InputParams.RingPolymerParameters, ens)),
      n_traj(InputParams.SimulationParameters.n_traj),
      initial_therm_length(
          InputParams.SimulationParameters.initial_therm_length),
      therm_length(InputParams.SimulationParameters.therm_length),
      run_length(InputParams.SimulationParameters.run_length),
      stride(InputParams.SimulationParameters.stride) {
  if (sim_type == "CMD") {
    RP = std::make_shared<AdiabaticRP>(InputParams.RingPolymerParameters, ens);
  } else if (sim_type == "QCMD") {
    RP = std::make_shared<QuasiCentroidRP>(
        InputParams.RingPolymerParameters, ens);
  }
  if (sim_type == "QCMD") {
    Dyn = std::make_unique<QCDynamics>(
        InputParams.DynamicsParameters, InputParams.ThermostatParameters,
        InputParams.SimulationParameters, InputParams.PotentialParameters, RP);
  } else if (sim_type == "Classical") {
    Dyn = std::make_unique<ClassicalDynamics>(
        InputParams.DynamicsParameters, InputParams.ThermostatParameters,
        InputParams.SimulationParameters, InputParams.PotentialParameters, RP);
  } else {
    Dyn = std::make_unique<Dynamics>(
        InputParams.DynamicsParameters, InputParams.ThermostatParameters,
        InputParams.SimulationParameters, InputParams.PotentialParameters, RP);
  }
}
