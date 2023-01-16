#include "PartridgeSchwenke.hpp"

#include "../../util/RingPolymerUtils.hpp"
#include "PartridgeSchwenkeWrappers.hpp"

PartridgeSchwenke::PartridgeSchwenke(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : Potential(SimParams, PotParams) {
}

arma::cube PartridgeSchwenke::Force(const arma::cube &q) {
  // fortran function units: q : Angstrom, v : kcal/mol, f : kcal/mol/Angstrom
  arma::cube f_cart(arma::size(q), arma::fill::zeros);
  arma::vec pot(q.n_slices / 3, arma::fill::zeros);
  arma::mat pot_mat(q.n_rows, q.n_slices / 3, arma::fill::zeros);
  arma::mat q_temp(3, 3, arma::fill::zeros);
  arma::mat f_cart_temp(3, 3, arma::fill::zeros);

  for (arma::uword i = 0; i < q.n_slices; i += 3) {
    for (arma::uword j = 0; j < q.n_rows; j++) {
      q_temp.row(0) = q.slice(i).row(j);
      q_temp.row(1) = q.slice(i + 1).row(j);
      q_temp.row(2) = q.slice(i + 2).row(j);
      q_temp *= 0.529177210903;  // bohr to angstrom
      Partridge_Schwenke_wrapper(
          q_temp.memptr(), f_cart_temp.memptr(), &pot_mat(j, i / 3));
      f_cart_temp *= -1.0 * 0.00159360247 *
                     0.529177210903;  // kcal mol-1 --> Eh, A-1 --> au-1
      f_cart.slice(i).row(j) = f_cart_temp.row(0);
      f_cart.slice(i + 1).row(j) = f_cart_temp.row(1);
      f_cart.slice(i + 2).row(j) = f_cart_temp.row(2);
      pot(i / 3) += pot_mat(j, i / 3);
    }
  }
  pot /= q.n_rows;
  pot *= 0.00159360247;
  return f_cart;
}

arma::cube PartridgeSchwenke::Dipole(const arma::cube &q) {
  arma::cube out(q.n_rows, q.n_cols, q.n_slices / 3, arma::fill::zeros);
  for (arma::uword i = 0; i < q.n_slices; i += 3) {
    double q1, q2;
    arma::mat OH1 = BondMat(q, i, i + 1);
    arma::mat OH2 = BondMat(q, i, i + 2);
    arma::vec r1 = BondLength(OH1);
    arma::vec r2 = BondLength(OH2);
    arma::vec ctheta = CosThetaVec(OH1, OH2);
    for (arma::uword j = 0; j < q.n_rows; j++) {
      Partridge_Schwenke_dipole_wrapper(&r1(j), &r2(j), &ctheta(j), &q1, &q2);
      out.slice(i / 3).row(j) = (q1 * OH1.row(j)) + (q2 * OH2.row(j));
    }
  }
  return out;
}

arma::vec PartridgeSchwenke::Pot(const arma::cube &q) {
  arma::cube f_cart(arma::size(q), arma::fill::zeros);
  arma::vec pot(q.n_slices / 3, arma::fill::zeros);
  arma::mat pot_mat(q.n_rows, q.n_slices / 3, arma::fill::zeros);
  arma::mat q_temp(3, 3, arma::fill::zeros);
  arma::mat f_cart_temp(3, 3, arma::fill::zeros);

  for (arma::uword i = 0; i < q.n_slices; i += 3) {
    for (arma::uword j = 0; j < q.n_rows; j++) {
      q_temp.row(0) = q.slice(i).row(j);
      q_temp.row(1) = q.slice(i + 1).row(j);
      q_temp.row(2) = q.slice(i + 2).row(j);
      q_temp *= 0.529177210903;  // bohr to angstrom
      Partridge_Schwenke_wrapper(
          q_temp.memptr(), f_cart_temp.memptr(), &pot_mat(j, i / 3));
      f_cart_temp *= -1.0 * 0.00159360247 *
                     0.529177210903;  // kcal mol-1 --> Eh, A-1 --> au-1
      f_cart.slice(i).row(j) = f_cart_temp.row(0);
      f_cart.slice(i + 1).row(j) = f_cart_temp.row(1);
      f_cart.slice(i + 2).row(j) = f_cart_temp.row(2);
      pot(i / 3) += pot_mat(j, i / 3);
    }
  }
  pot /= q.n_rows;
  pot *= 0.00159360247;
  return pot;
}
