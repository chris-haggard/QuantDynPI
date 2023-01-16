#include "Dynamics.hpp"

#include "../potentials/AmmoniaPES.hpp"
#include "../potentials/GasQTIP4P.hpp"
#include "../potentials/Harmonic.hpp"
#include "../potentials/PartridgeSchwenke.hpp"
#include "../potentials/SplinePot.hpp"
#include "../potentials/SilveraGoldman.hpp"
#include "../thermostat/PILE.hpp"
#include "../thermostat/PILE_G.hpp"
#include "../thermostat/PILE_L.hpp"

/**
 * @brief Construct the Dynamics object and create the required Therm and PES
 * object as a unique_ptr
 *
 * @param DynamicsParams
 * @param ThermostatParams
 * @param SimParams
 * @param PotentialParams
 * @param rp
 */
Dynamics::Dynamics(
    const InputHandler::DynamicsInput& DynamicsParams,
    const InputHandler::ThermostatInput& ThermostatParams,
    const InputHandler::SimulationInput& SimParams,
    const InputHandler::PotentialInput& PotentialParams,
    std::shared_ptr<RingPolymer> rp)
    : dt(DynamicsParams.dt),
      ThermThermostat(
          std::make_unique<Thermostat>(ThermostatParams, DynamicsParams, rp)),
      PES(std::make_unique<Potential>(SimParams, PotentialParams)) {
  RP = rp;
  ThermThermostat =
      std::make_unique<PILE>(ThermostatParams, DynamicsParams, rp, true);
  RunThermostat =
      std::make_unique<PILE>(ThermostatParams, DynamicsParams, rp, false);

  // determine PES
  if (PotentialParams.potential_type == "Silvera-Goldman") {
    PES = std::make_unique<SilveraGoldman>(SimParams, PotentialParams);
  } else if (PotentialParams.potential_type == "gas-qtip") {
    assert(
        RP->labels.at(0) == "O" && RP->labels.at(1) == "H" &&
        RP->labels.at(2) == "H");  //  && RP->labels.at(3) == "O");
    PES = std::make_unique<GasQTIP4P>(SimParams, PotentialParams);
  } else if (PotentialParams.potential_type == "gas-PS") {
    assert(
        RP->labels.at(0) == "O" && RP->labels.at(1) == "H" &&
        RP->labels.at(2) == "H");  //  && RP->labels.at(3) == "O");
    PES = std::make_unique<PartridgeSchwenke>(SimParams, PotentialParams);
  } else if (PotentialParams.potential_type == "gas-ammonia") {
    assert(
        RP->labels.at(0) == "H" && RP->labels.at(1) == "H" &&
        RP->labels.at(2) == "H" && RP->labels.at(3) == "N");
    PES = std::make_unique<AmmoniaPES>(SimParams, PotentialParams);
  } else if (PotentialParams.potential_type == "spline") {
    PES = std::make_unique<SplinePot>(SimParams, PotentialParams);
  }

  else {
    throw std::runtime_error("Invalid potental supplied\n");
  }

  FreeRPPropagator = FreeRPPropagatorCreate();
}

/**
 * @brief Velocity-Verlet position update through dt/2
 *
 * @param position
 * @param momentum
 * @param mass
 */
void Dynamics::UpdatePosition(
    arma::cube& position, const arma::cube& momentum, const arma::cube& mass) {
  position += (momentum * dt) / mass;
}

/**
 * @brief Velocity-Verlet momenta update through dt/2
 *
 * @param momentum
 * @param force
 */
void Dynamics::UpdateMomentum(arma::cube& momentum, const arma::cube& force) {
  momentum += 0.5 * dt * force;
}

/**
 * @brief Exact evolution of the ringpolymer coordinates and momenta through a
 * time interval dt under the influence of the free ring polymer Hamiltonian
 * (10.1063/1.3489925)
 */
void Dynamics::FreeRPStep() {
  arma::mat temp(2, 3);
  for (arma::uword i = 0; i < RP->position_nm.n_slices; i++) {
    for (arma::uword j = 0; j < RP->position_nm.n_rows; j++) {
      temp.row(0) = RP->momenta_nm.slice(i).row(j);
      temp.row(1) = RP->position_nm.slice(i).row(j);
      temp = FreeRPPropagator(i % RP->n_atoms).slice(j) * temp;
      RP->momenta_nm.slice(i).row(j) = temp.row(0);
      RP->position_nm.slice(i).row(j) = temp.row(1);
    }
  }
}

/**
 * @brief Compute the internal ringpolymer forces in Cartesian
 *
 * NOTE: assumes that position_cart is up to date. This is rarely the case.
 * Propagation shoud be done in normal modes.
 *
 */
void Dynamics::InternalForceCart() {
  for (arma::uword k = 0; k < RP->force_cart.n_slices; k++) {
    RP->force_cart.slice(k) +=
        (-RP->mass_atomic.slice(k) / (RP->beta_n * RP->beta_n)) %
        ((2.0 * RP->position_cart.slice(k)) -
         arma::shift(RP->position_cart.slice(k), +1) -
         arma::shift(RP->position_cart.slice(k), -1));
  }
}

void Dynamics::InternalForceNM() {
  RP->force_nm -=
      RP->mass_scaled % arma::square(RP->freq_scaled) % RP->position_nm;
}

/**
 * @brief Update the force with both the external potential and the internal
 * ringpolymer potential (all in Cartesian)
 */
void Dynamics::UpdateForce() {
  UpdateExternalForce();
  InternalForceNM();
}

/**
 * @brief Update just using the external potential. Useful for when using the
 * free ringpolymer propagator
 */
void Dynamics::UpdateExternalForce() {
  RP->NMConv.NMToCart(RP->position_nm, RP->position_cart);
  // RP->position_cart.print("position cart in update externalforce");
  RP->force_cart = PES->Force(RP->position_cart);
  RP->NMConv.CartToNM(RP->force_cart, RP->force_nm);
}

/**
 * @brief Creates a field of cubes where n_elem is the number of atoms per
 * molecule. e.g for H20 this would be a field with 3 elements each with is a
 * propagator cube
 *
 * @return
 */
arma::field<arma::cube> Dynamics::FreeRPPropagatorCreate() {
  double f, m;

  arma::field<arma::cube> propagator(RP->n_atoms);

  for (arma::uword s = 0; s < propagator.n_elem; s++) {
    arma::cube propagator_cube(2, 2, RP->n_beads);

    propagator_cube(0, 0, 0) = 1.0;
    propagator_cube(0, 1, 0) = 0.0;
    propagator_cube(1, 0, 0) = dt / arma::as_scalar(RP->mass_scaled(0, 0, s));
    propagator_cube(1, 1, 0) = 1.0;
#ifdef FROZEN_CENTROID
    propagator_cube(1, 0, 0) = 0.0;
#endif
    for (arma::uword k = 1; k < RP->n_beads; k++) {
      f = arma::as_scalar(RP->freq_scaled.slice(s).row(k).col(0));
      m = arma::as_scalar(RP->mass_scaled.slice(s).row(k).col(0));
      propagator_cube(0, 0, k) = std::cos(f * dt);
      propagator_cube(0, 1, k) = -(m * f * std::sin(f * dt));
      propagator_cube(1, 0, k) = (1.0 / (m * f)) * std::sin(f * dt);
      propagator_cube(1, 1, k) = std::cos(f * dt);
    }
    propagator(s) = propagator_cube;
  }
  return propagator;
}

/**
 * @brief Propagate for one timestep under the action of the thermostat
 */
void Dynamics::ThermStep() {
  ThermThermostat->ThermostatStep();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  FreeRPStep();
  UpdateExternalForce();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  ThermThermostat->ThermostatStep();
  RP->NMConv.NMToCart(RP->position_nm, RP->position_cart);
}

/**
 * @brief Propagate for one timestep without the action of a thermostat i.e.
 * RPMD run
 */
void Dynamics::Step() {
  RunThermostat->ThermostatStep();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  FreeRPStep();
  UpdateExternalForce();
  UpdateMomentum(RP->momenta_nm, RP->force_nm);
  RunThermostat->ThermostatStep();
  RP->NMConv.NMToCart(RP->position_nm, RP->position_cart);
}
