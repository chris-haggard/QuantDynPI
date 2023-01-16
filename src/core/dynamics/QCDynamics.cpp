#include "QCDynamics.hpp"

#include <memory>
#include <stdexcept>

#include "../../util/RingPolymerUtils.hpp"
#include "Dynamics.hpp"
#include "QuasiThermostat.hpp"
#include "qcmd/TetraatomicQCMD.hpp"
#include "qcmd/TriatomicQCMD.hpp"

QCDynamics::QCDynamics(
    const InputHandler::DynamicsInput &DynamicsParams,
    const InputHandler::ThermostatInput &ThermostatParams,
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotentialParams,
    std::shared_ptr<RingPolymer> rp)
    : Dynamics(
          DynamicsParams, ThermostatParams, SimParams, PotentialParams, rp) {
  std::cout << "QCDynamics constructor\n";

  QC_RP = std::static_pointer_cast<QuasiCentroidRP>(RP);

  QC_Funcs = std::make_unique<QCMD>(QC_RP);

  if (QC_RP->n_atoms == 3) {
    QC_Funcs = std::make_unique<TriatomicQCMD>(QC_RP);
  } else {
    QC_Funcs = std::make_unique<TetraatomicQCMD>(QC_RP);
  }

  QuasiTherm = std::make_unique<QuasiThermostat>(
      ThermostatParams, DynamicsParams, QC_RP);

  lambda_prev =
      arma::mat(QC_RP->n_atoms * 3, QC_RP->n_molec, arma::fill::zeros);

  Dynamics::UpdateForce();
}

/**
 * @brief Enforce the position constraints on the position normal modes using
 * the MSHAKE algorithm. Requires that positon_cart be up to date with
 * position_nm.
 *
 * @param q_prev
 */
void QCDynamics::MShake(const arma::cube &q_prev) {
  const size_t n_constraints = QC_RP->n_atoms * 3;

  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += QC_RP->n_atoms) {
    arma::vec lambda(n_constraints, arma::fill::zeros);
    arma::vec delta_lambda(n_constraints, arma::fill::zeros);
    arma::vec g(n_constraints);
    arma::mat GramMat(n_constraints, n_constraints);
    arma::field<arma::cube> dg_nm(n_constraints);

    lambda.zeros();
    arma::uword j = i + QC_RP->n_atoms - 1;

    arma::field<arma::cube> dg_cart = QC_Funcs->ConstraintsGradient(
        q_prev.slices(i, j), QC_RP->qc_position.slices(i, j),
        QC_RP->mass_atomic.slices(i, j));

    // get cubes of correct size in dg_nm for the normal mode transform
    dg_nm = dg_cart;
    for (arma::uword k = 0; k < n_constraints; k++) {
      QC_RP->NMConv.CartToNM(dg_cart(k), dg_nm(k));
      dg_nm(k) /= QC_RP->mass_scaled.slices(i, j);
      // hack to get around correct mass scaling for Gramian mat
      dg_cart(k) = dg_nm(k) % QC_RP->mass_scaled.slices(i, j);
    }

    // don't divide by the mass twice for the Gramian matrix
    GramMat = QC_Funcs->GramianMat(dg_nm, dg_cart);

    size_t iterations = 0;
    while (iterations < 30) {
      g = QC_Funcs->Constraints(
          QC_RP->position_cart.slices(i, j), QC_RP->qc_position.slices(i, j));

      if (arma::all(arma::abs(g) < 1e-8)) {
        break;
      }

      delta_lambda = arma::solve(GramMat, g, arma::solve_opts::no_approx);
      lambda += delta_lambda;

      // dg_nm already divided by the mass
      for (arma::uword k = 0; k < n_constraints; k++) {
        QC_RP->position_nm.slices(i, j) -=
            delta_lambda(k) * dg_nm(k);
      }

      // operate on all slices -  but does not work for subcube type
      QC_RP->NMConv.NMToCart(QC_RP->position_nm, QC_RP->position_cart);

      iterations++;
    }
    if (iterations == 30) {
      throw std::runtime_error(
          "constraints failed to converge in given iterations");
    }
    for (arma::uword k = 0; k < n_constraints; k++) {
      QC_RP->momenta_nm.slices(i, j) -=
          lambda(k) * dg_cart(k) / dt;  // dg_cart is really dg_nm but already
                                        // multiplied by masses(as needed here)
    }
  }
}

/**
 * @brief Enforce constraints on momenta normal modes. Requires position_cart to
 * be up to date which it should be on all calls since only A2 and MSHAKE alter
 * position_nm/cart and it leaves those functions up to date.
 */
void QCDynamics::Rattle() {
  const size_t n_constraints = QC_RP->n_atoms * 3;
  arma::vec mu(n_constraints);
  arma::field<arma::cube> dg_nm(n_constraints);
  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += QC_RP->n_atoms) {
    arma::uword j = i + QC_RP->n_atoms - 1;

    arma::field<arma::cube> dg_cart = QC_Funcs->ConstraintsGradient(
        QC_RP->position_cart.slices(i, j), QC_RP->qc_position.slices(i, j),
        QC_RP->mass_atomic.slices(i, j));

    dg_nm = dg_cart;

    for (arma::uword k = 0; k < n_constraints; k++) {
      QC_RP->NMConv.CartToNM(dg_cart(k), dg_nm(k));
      dg_nm(k) /= QC_RP->mass_scaled.slices(i, j);
    }
    // hack to get around correct mass scaling for Gramian mat
    for (arma::uword k = 0; k < n_constraints; k++) {
      dg_cart(k) = dg_nm(k) % QC_RP->mass_scaled.slices(i, j);
    }

    mu = arma::solve(
        -QC_Funcs->GramianMat(dg_nm, dg_cart),
        QC_Funcs->RattleVec(
            QC_RP->momenta_nm.slices(i, j) / QC_RP->mass_scaled.slices(i, j),
            dg_cart));
    for (arma::uword k = 0; k < n_constraints; k++) {
      QC_RP->momenta_nm.slices(i, j) += mu(k) * dg_cart(k);
    }
  }
}

/**
 * @brief Propagate the quasicentroid momenta P for half a time step (∆t/2)
 * under the action of a Langevin thermostat.
 */
void QCDynamics::O1() {
  QuasiTherm->ThermostatStep();
}

/**
 * @brief Propagate the bead momenta p for ∆t/2 under the path-integral
 * Langevin (PILE) thermostat, followed by RATTLE to constrain the
 * quasicentroid components
 */
void QCDynamics::O2() {
  ThermThermostat->ThermostatStep();
  Rattle();
}

/**
 * @brief Propagate P for ∆t/2 under the quasicentroid forces.
 */
void QCDynamics::B1() {
  UpdateMomentum(QC_RP->qc_momenta, QC_RP->qc_force);
}

/**
 * @brief Propagate p for ∆t/2 under the forces derived from the ring-polymer
 * force, followed by RATTLE
 */
void QCDynamics::B2() {
  UpdateMomentum(QC_RP->momenta_nm, QC_RP->force_nm);
  Rattle();
}

/**
 * @brief Propagate the quasicentroid positions Q for a full time step ∆t
 * according to the current values of the momenta P.
 */
void QCDynamics::A1() {
  UpdatePosition(QC_RP->qc_position, QC_RP->qc_momenta, QC_RP->qc_mass);
}

/**
 * @brief Propagate the bead positions q for ∆t according to the current
 * values of p, followed by SHAKE, which constrains the ring-polymer geometry
 * to be consistent with the quasicentroid configuration Q at the end of step
 * A1
 */
void QCDynamics::A2() {
  QC_RP->NMConv.NMToCart(QC_RP->position_nm, QC_RP->position_cart);
  arma::cube pos_prev = QC_RP->position_cart;
  UpdatePosition(QC_RP->position_nm, QC_RP->momenta_nm, QC_RP->mass_scaled);
  QC_RP->NMConv.NMToCart(QC_RP->position_nm, QC_RP->position_cart);
  MShake(pos_prev);
}

void QCDynamics::Step() {
  ThermStep();
}

void QCDynamics::ThermStep() {
  O1();
  O2();

  B1();
  B2();

#ifndef FROZEN_CENTROID
  A1();
#endif
  A2();

  Dynamics::UpdateExternalForce();
  QC_Funcs->update_qc_force();
  Dynamics::InternalForceNM();

  B1();
  B2();

  O1();
  O2();
}

