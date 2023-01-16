#include "SilveraGoldman.hpp"

#include <cmath>

#include "Potential.hpp"

SilveraGoldman::SilveraGoldman(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : Potential(SimParams, PotParams) {
}

double SilveraGoldman::SGForceMagnitude(const double r) {
  if (r <= rc) {
    double w =
        (-1.0 * std::exp(-std::pow((-1.0 + rc / r), 2)) *
         ((-10.0 * C_10 / std::pow(r, 11)) + (9.0 * C_9 / std::pow(r, 10)) +
          (-8.0 * C_8 / std::pow(r, 9)) + (-6.0 * C_6 / std::pow(r, 7)))) -
        (2.0 * std::exp(-std::pow((-1.0 + rc / r), 2)) *
         ((C_10 / std::pow(r, 10)) - (C_9 / std::pow(r, 9)) +
          (C_8 / std::pow(r, 8)) + (C_6 / std::pow(r, 6))) *
         rc * (-1.0 + (rc / r))) /
            std::pow(r, 2) +
        (std::exp(a - (b * r) - (g * r * r)) * (-b - (2 * r * g)));
    return w;
  } else {
    double w =
        (((10.0 * C_10 / std::pow(r, 11)) + (-9.0 * C_9 / std::pow(r, 10)) +
          (8.0 * C_8 / std::pow(r, 9)) + (6.0 * C_6 / std::pow(r, 7)))) +
        (exp(a - (b * r) - (g * r * r)) * (-b - (2.0 * r * g)));
    return w;
  }
}

// https://doi.org/10.1063/1.1893956
arma::cube SilveraGoldman::Force(const arma::cube &position_cart) {
  arma::cube f(arma::size(position_cart), arma::fill::zeros);
  for (arma::uword i = 0; i < position_cart.n_slices; i++) {
    for (arma::uword j = i + 1; j < position_cart.n_slices; j++) {
      arma::rowvec centroid_distance_vector_raw =
          arma::mean(position_cart.slice(j)) -
          arma::mean(position_cart.slice(i));
      arma::rowvec centroid_distance_vector =
          centroid_distance_vector_raw -
          (box_length *
           arma::round(centroid_distance_vector_raw * box_length_reciprocal));
      double centroid_distance = arma::norm(centroid_distance_vector);

      if (centroid_distance <= 15.0 && j != i) {
        arma::mat distance_mat =
            position_cart.slice(j) - position_cart.slice(i);
        for (arma::uword bead = 0; bead < position_cart.n_rows; bead++) {
          arma::rowvec distance_raw_vec = distance_mat.row(bead);

          arma::rowvec distance_vec =
              distance_raw_vec -
              (box_length *
               arma::round(distance_raw_vec * box_length_reciprocal));
          // distance vec same as distance raw vec if boundary NOT crossed

          double distance = arma::norm(distance_vec);

          // this is -dV/dr  i.e. -dV/dx1 ... etc
          double force_magnitude = SGForceMagnitude(distance);

          // \vec{F} = norm(\vec{F}) / r * \vec{x, y}
          f.slice(i).row(bead) += (force_magnitude * (distance_vec / distance));
          // pairwise interaction is just negative of previous
          f.slice(j).row(bead) -= (force_magnitude * (distance_vec / distance));
        }
      }
    }
  }
  assert(arma::is_finite(f));
  return f;
}
