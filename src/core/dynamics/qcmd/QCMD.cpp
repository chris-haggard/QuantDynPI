#include "QCMD.hpp"

#include <memory>
#include <stdexcept>

QCMD::QCMD(std::shared_ptr<QuasiCentroidRP> qc_rp) {
  QC_RP = qc_rp;
  std::cout << "QCMD object constructor\n";
  QC_RP->qc_position.print("in QCMD");
}

/**
 * @brief Construct the matrix of coefficients for the Lagrangian multipliers in
 * the Shake step. This forms a set of linear equations which is the solved to
 * find \delta\lambda. See Tuckerman Eq. 3.9.13
 *
 * @param constraint_grad
 * @param constraint_grad_prev
 *
 * @return
 */
arma::mat QCMD::GramianMat(
    const arma ::field<arma::cube> &constraint_grad,
    const arma::field<arma::cube> &constraint_grad_prev) {
  arma::mat out(
      constraint_grad.n_elem, constraint_grad.n_elem, arma::fill::zeros);
  for (arma::uword l = 0; l < constraint_grad.n_elem; l++) {
    for (arma::uword k = 0; k < constraint_grad.n_elem; k++) {
      out(l, k) += arma::accu(constraint_grad(l) % constraint_grad_prev(k));
    }
  }
  return out;
}

/**
 * @brief Construct the vector used in the Rattle step (momenta constraints).
 * Additionally, this is the same vector that can be use to check the velocity
 * constraint (Tuckerman 3.9.18). Note use of velocity rather than momenta.
 *
 * @param velocity
 * @param constraint_grad
 *
 * @return
 */
arma::vec QCMD::RattleVec(
    const arma::cube &velocity,
    const arma::field<arma::cube> &constraint_grad) {
  arma::vec out(constraint_grad.n_elem, arma::fill::zeros);
  for (arma::uword k = 0; k < constraint_grad.n_elem; k++) {
    for (arma::uword i = 0; i < QC_RP->n_atoms; i++) {
      for (arma::uword j = 0; j < QC_RP->n_beads; j++) {
        out(k) += arma::dot(
            constraint_grad(k).slice(i).row(j), velocity.slice(i).row(j));
      }
    }
  }
  return out;
}

void QCMD::update_qc_force() {
  throw std::runtime_error("No functions in QCMD base class");
}

arma::vec QCMD::Constraints(const arma::cube &q, const arma::cube &qc) {
  throw std::runtime_error("No functions in QCMD base class");
}

arma::field<arma::cube> QCMD::ConstraintsGradient(
    const arma::cube &q, const arma::cube &qc, const arma::cube &m) {
  throw std::runtime_error("No functions in QCMD base class");
}

