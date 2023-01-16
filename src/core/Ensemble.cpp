#include "Ensemble.hpp"

#include "../util/unit_conversion.hpp"

Ensemble::Ensemble(const InputHandler::EnsembleInput &EnsembleParams)
    : beta(T_to_beta(EnsembleParams.T)) {
}

double Ensemble::V_estimator(const arma::vec &V) {
  // assumes that V returns 1/N Sum V(q_i) for each molecule
  return arma::sum(V);
}

double Ensemble::Virial_estimator(
    const arma::cube &q_cart, const arma::cube &f_cart) {
  assert(q_cart.n_rows > 1);
  double out = 3.0 * q_cart.n_slices * 0.5 / beta;
  double temp = 0.0;
  arma::cube centroid = arma::mean(q_cart, 0);
  for (arma::uword i = 0; i < q_cart.n_rows; i++) {
    for (arma::uword j = 0; j < q_cart.n_slices; j++) {
      temp += arma::dot(
          q_cart.slice(j).row(i) - centroid.slice(j), f_cart.slice(j).row(i));
    }
  }
  return out + (temp * 0.5 / q_cart.n_rows);
}
