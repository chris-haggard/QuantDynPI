#include "GridQCMD.hpp"

GridQCMD::GridQCMD(
    const InputHandler::DynamicsInput &DynamicsParams,
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotentialParams,
    std::shared_ptr<RingPolymer> rp)
    : QCDynamics(
          DynamicsParams, ThermostatParams, SimParams, PotentialParams, rp) {
}

void QCDynamics::ThermStep() {
  O1();
  O2();

  B1();
  B2();

  A2();

  Dynamics::UpdateExternalForce();
  QC_Funcs->update_qc_force();
  Dynamics::InternalForceNM();

  B1();
  B2();

  O1();
  O2();
}

