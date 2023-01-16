#pragma once
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <memory>
#include <string>

#include "../Input.hpp"
#include "../potentials/Potential.hpp"
#include "../ringpolymer/RingPolymer.hpp"
#include "../thermostat/Thermostat.hpp"

class Dynamics {
 public:
  const double dt;
  std::unique_ptr<Potential> PES;
  void UpdatePosition(arma::cube &, const arma::cube &, const arma::cube &);
  void UpdateMomentum(arma::cube &, const arma::cube &);
  void InternalForceCart();
  void InternalForceNM();
  void UpdateForce();
  void UpdateExternalForce();
  void FreeRPStep();
  virtual void ThermStep();
  virtual void Step();
  Dynamics(
      const InputHandler::DynamicsInput &,
      const InputHandler::ThermostatInput &,
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &, std::shared_ptr<RingPolymer>);

 protected:
  arma::field<arma::cube> FreeRPPropagator;
  arma::field<arma::cube> FreeRPPropagatorCreate();
  const std::string thermostat_type;
  std::shared_ptr<RingPolymer> RP;
  // Thermostat to apply in pre-run thermalization
  std::unique_ptr<Thermostat> ThermThermostat;
  // Thermostat to apply during run, most obvious example is thermostatting
  // non-centorid modes in CMD during run
  std::unique_ptr<Thermostat> RunThermostat;
};
