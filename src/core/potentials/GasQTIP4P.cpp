#include "GasQTIP4P.hpp"

#include "../../util/RingPolymerUtils.hpp"
#include "Potential.hpp"

GasQTIP4P::GasQTIP4P(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : Potential(SimParams, PotParams) {
}

/**
 * @brief The harmonic bond angle force in the QTIP4P liquid water model.
 * Requires OHH ordering.
 *
 * @param pos
 *
 * @return
 */
arma::cube GasQTIP4P::AngleForce(const arma::cube &pos) {
  arma::cube force(arma::size(pos));
  arma::mat OH1vec, OH2vec, theta, pot, f_th;
  arma::mat OH1_norm(arma::size(pos.slice(0)));
  arma::mat OH2_norm(arma::size(pos.slice(0)));
  arma::mat cos_theta(arma::size(pos.slice(0)));

  for (arma::uword i = 0; i < pos.n_slices; i += 3) {
    arma::mat OH1vec = pos.slice(i + 1) - pos.slice(i);
    arma::mat OH2vec = pos.slice(i + 2) - pos.slice(i);
    for (arma::uword j = 0; j < pos.n_rows; j++) {
      OH1_norm.row(j).fill(arma::norm(OH1vec.row(j)));
      OH2_norm.row(j).fill(arma::norm(OH2vec.row(j)));
      cos_theta.row(j).fill(arma::norm_dot(OH1vec.row(j), OH2vec.row(j)));
    }
    theta = arma::acos(cos_theta);
    // compute potential and -dV/dTheta for each bead
    pot = 0.5 * k_theta * arma::square(theta - theta_eq);
    // dV/dTheta =
    f_th = k_theta * (theta - theta_eq);
    // work out -dTheta/dO_x, -dTheta/dH_1x etc
    // -dV/dH_1
    force.slice(i + 1) =
        f_th % (((OH2vec / (OH1_norm % OH2_norm)) -
                 ((cos_theta % OH1vec) / arma::square(OH1_norm))) /
                (arma::sqrt(1.0 - arma::square(cos_theta))));
    // -dV/H_2
    force.slice(i + 2) =
        f_th % (((OH1vec / (OH1_norm % OH2_norm)) -
                 ((cos_theta % OH2vec) / arma::square(OH2_norm))) /
                (arma::sqrt(1.0 - arma::square(cos_theta))));
    // -dV/dO = -(-dV/H_1 + -dV/DH_2)
    force.slice(i) = -(force.slice(i + 1) + force.slice(i + 2));
  }
  return force;
}

/**
 * @brief The bond force in the qtip potential.
 *
 * @param pos
 *
 * @return
 */
arma::cube GasQTIP4P::BondForce(const arma::cube &pos) {
  arma::cube force(arma::size(pos), arma::fill::zeros);
  arma::mat OH1_vec, OH2_vec, OH1_norm(arma::size(force.slice(0))),
      OH2_norm(arma::size(force.slice(0))), f_bond_1, f_bond_2;
  for (arma::uword i = 0; i < pos.n_slices; i += 3) {
    OH1_vec = (pos.slice(i + 1) - pos.slice(i));
    OH2_vec = (pos.slice(i + 2) - pos.slice(i));
    OH1_norm.each_col() = arma::sqrt(arma::sum(OH1_vec % OH1_vec, 1));
    OH2_norm.each_col() = arma::sqrt(arma::sum(OH2_vec % OH2_vec, 1));
    // -dV/dr_1
    f_bond_1 =
        -D_r * ((2.0 * alpha_r * alpha_r * (OH1_norm - r_eq)) -
                (3.0 * std::pow(alpha_r, 3) * (arma::square(OH1_norm - r_eq))) +
                ((28.0 / 12.0) * std::pow(alpha_r, 4) *
                 arma::pow(OH1_norm - r_eq, 3)));
    // -dV/dr_2
    f_bond_2 =
        -D_r * ((2.0 * alpha_r * alpha_r * (OH2_norm - r_eq)) -
                (3.0 * std::pow(alpha_r, 3) * (arma::square(OH2_norm - r_eq))) +
                ((28.0 / 12.0) * std::pow(alpha_r, 4) *
                 arma::pow(OH2_norm - r_eq, 3)));
    // dr_1/dH_1, dr_2/dH_2
    force.slice(i + 1) = f_bond_1 % OH1_vec / OH1_norm;
    force.slice(i + 2) = f_bond_2 % OH2_vec / OH2_norm;
    force.slice(i) = -(force.slice(i + 1) + force.slice(i + 2));
  }
  return force;
}

arma::cube GasQTIP4P::IntramolecularForce(const arma::cube &pos) {
  return AngleForce(pos) + BondForce(pos);
}

/**
 * @brief Internal forces for qtip. Tested against implementation in qclpysim
 * and obtained same results.
 *
 * @param pos
 *
 * @return
 */
arma::cube GasQTIP4P::Force(const arma::cube &pos) {
  return IntramolecularForce(pos);
}

/**
 * @brief Linear dipole moment for qtip. Requires OHH ordering.
 *
 * @param pos
 *
 * @return
 */
arma::cube GasQTIP4P::Dipole(const arma::cube &pos) {
  // each molecule has single dipole moment
  arma::cube dipole(pos.n_rows, 3, pos.n_slices / 3);
  for (arma::uword i = 0; i < pos.n_slices; i += 3) {
    dipole.slice(i / 3) =
        (-2.0 * pos.slice(i)) + pos.slice(i + 1) + pos.slice(i + 2);
  }
  return dipole * gamma * q_H;
}

arma::vec GasQTIP4P::BondPot(const arma::vec &r) {
  arma::vec V_r =
      D_r * ((std::pow(alpha_r, 2) * arma::square(r)) -
             (std::pow(alpha_r, 3) * arma::pow(r, 3)) +
             ((7.0 / 12.0) * std::pow(alpha_r, 4) * arma::pow(r, 4)));

  return V_r;
}

arma::vec GasQTIP4P::AnglePot(const arma::vec &theta) {
  return 0.5 * k_theta * arma::square(theta);
}

arma::vec GasQTIP4P::Pot(const arma::cube &pos) {
  arma::vec out(pos.n_slices / 3, arma::fill::zeros);
  arma::mat OH1, OH2;
  arma::vec r_1, r_2, theta;
  for (arma::uword i = 0; i < pos.n_slices; i += 3) {
    OH1 = BondMat(pos, i, i + 1);
    OH2 = BondMat(pos, i, i + 2);
    r_1 = BondLength(OH1) - r_eq;
    r_2 = BondLength(OH2) - r_eq;
    theta = arma::acos(CosThetaVec(OH1, OH2)) - theta_eq;
    out(i / 3) = arma::mean(BondPot(r_1) + BondPot(r_2) + AnglePot(theta));
  }
  return out;
}
