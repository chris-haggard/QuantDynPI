#pragma once

#include <memory>

#include "../ringpolymer/QuasiCentroidRP.hpp"
#include "Dynamics.hpp"
#include "QuasiThermostat.hpp"
#include "qcmd/QCMD.hpp"

/**
 * @brief Class for QCMD dynamics.
 */
class QCDynamics : public Dynamics {
 protected:
  // Need access to the QC system variables, qc_position, qc_momenta etc. This
  // is done with a static_cast as at this point RP must be of type
  // QuasicentroidRP.
  std::shared_ptr<QuasiCentroidRP> QC_RP;
  std::unique_ptr<QCMD> QC_Funcs;
  void MShake(const arma::cube &);
  void Shake(const arma::cube &);
  void Rattle();
  void O1();
  void O2();
  void B1();
  void B2();
  void A1();
  void A2();
  arma::mat lambda_prev;
  std::unique_ptr<QuasiThermostat> QuasiTherm;

 public:
  QCDynamics(
      const InputHandler::DynamicsInput &,
      const InputHandler::ThermostatInput &,
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &, std::shared_ptr<RingPolymer>);
  void ThermStep() override;
  void Step() override;
};

