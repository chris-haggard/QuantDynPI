#include "RingPolymerUtils.hpp"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <cassert>
#include <stdexcept>

arma::mat BondMat(
    const arma::cube &pos_cart, const size_t first_index,
    const size_t second_index) {
  return pos_cart.slice(second_index) - pos_cart.slice(first_index);
}

arma::vec BondLength(const arma::mat &bond_matrix) {
  return arma::sqrt(arma::sum(arma::square(bond_matrix), 1));
}

arma::vec CosThetaVec(
    const arma::mat &bond_matrix1, const arma::mat &bond_matrix2) {
  arma::vec out(bond_matrix1.n_rows);
  for (arma::uword i = 0; i < bond_matrix1.n_rows; i++) {
    out(i) = arma::norm_dot(bond_matrix1.row(i), bond_matrix2.row(i));
  }
  return out;
}

arma::vec CosThetaVec(
    const arma::mat &bond_matrix1, const arma::mat &bond_matrix2,
    const arma::vec &r_1, const arma::vec &r_2) {
  return arma::sum(bond_matrix1 % bond_matrix2, 1) / (r_1 % r_2);
}

arma::vec SinThetaVec(
    const arma::mat &bond_matrix1, const arma::mat &bond_matrix2) {
  return arma::sqrt(
      1.0 - arma::square(CosThetaVec(bond_matrix1, bond_matrix2)));
}

arma::vec SinThetaVec(const arma::vec &cos_theta_vec) {
  return arma::sqrt(1.0 - arma::square(cos_theta_vec));
}

/**
 * @brief Slicewise vector cross product of two cubes (with a single row).
 * Useful in QCMD constraints.
 *
 * @param a
 * @param b
 *
 * @return
 */
arma::cube CubeCross(const arma::cube &a, const arma::cube &b) {
  assert(arma::size(a) == arma::size(b) && a.n_rows == 1 && a.n_cols == 3);
  arma::cube out(arma::size(a));
  for (arma::uword i = 0; i < a.n_slices; i++) {
    // rowvec in slice --> colvec --> cross --> colvec to rowvec
    out.slice(i) = arma::cross(a.slice(i).t(), b.slice(i).t()).t();
  }
  return out;
}

arma::cube CentroidCube(const arma::cube &q) {
  return arma::mean(q, 0);
}

/**
 * @brief Centre-of-Mass of the centroids/quasicentroids of each atom in a
 * molecule. Pass a subcube of each molec i.e. qc_position.slices(0, 2)
 *
 * @return
 */
arma::mat COM(const arma::cube &q, const arma::cube &m) {
  assert(q.n_rows == 1 && q.n_slices == m.n_slices);
  return arma::sum(q % m.row(0), 2) / arma::sum(m, 2);
}

/**
 * @brief Initialize momenta from the  Maxwell-Boltzmann distribution. Normal
 * distribution with mean 0 and stddev sqrt(m/beta).
 *
 * unclear if this is viable for scaled systems, better to start with very low
 * random momenta
 *
 *
 * only used for qc momenta, breaks if used for beads
 *
 *
 * @param mom
 * @param m
 * @param b
 *
 * @return
 */
arma::rowvec InitializeMomenta(
    const arma::rowvec &mom, const arma::rowvec &m, const double &b) {
  arma::rowvec out(size(mom));
  assert(arma::size(mom) == arma::size(m));
  arma::vec temp = arma::unique(m);
  assert(temp.n_elem == 1);
  // get actual randomness
  boost::random_device dev;
  boost::random::mt19937_64 gen(dev);
  boost::random::normal_distribution<double> dist(0, std::sqrt(m(0) / b));
  for (arma::uword j = 0; j < mom.n_cols; j++) {
    out(j) = dist(gen);
  }
  return out;
}

/**
 * @brief Orientation of ammonia replicas/centroid/quasicentroid. Used to detect
 * inversion of ammonia molecule
 *
 * @param q_cart
 *
 * @return
 */
arma::mat AmmoniaOrientation(const arma::cube &q_cart) {
  assert(q_cart.n_slices % 4 == 0);
  // shape:each field =  a molecule, each row = xyz dim, each col = a bond, each
  // slice = replica of molecule
  arma::mat det_mat(q_cart.n_slices / 4, q_cart.n_rows);
  arma::field<arma::cube> out(q_cart.n_slices / 4);
  arma::cube field_elements(3, 3, q_cart.n_rows);

  for (arma::uword i = 0; i < out.n_elem; i++) {
    out(i) = field_elements;
  }

  for (arma::uword i = 0; i < q_cart.n_slices; i += 4) {
    arma::mat NH1 = BondMat(q_cart, i + 3, i);
    arma::mat NH2 = BondMat(q_cart, i + 3, i + 1);
    arma::mat NH3 = BondMat(q_cart, i + 3, i + 2);
    arma::vec r_1 = BondLength(NH1);
    arma::vec r_2 = BondLength(NH2);
    arma::vec r_3 = BondLength(NH3);
    NH1.each_col() /= r_1;
    NH2.each_col() /= r_2;
    NH3.each_col() /= r_3;
    for (arma::uword j = 0; j < q_cart.n_rows; j++) {
      out(i / 4).slice(j).col(0) = NH1.row(j).t();
      out(i / 4).slice(j).col(1) = NH2.row(j).t();
      out(i / 4).slice(j).col(2) = NH3.row(j).t();
    }
  }

  for (arma::uword i = 0; i < out.n_elem; i++) {
    for (arma::uword k = 0; k < q_cart.n_rows; k++) {
      det_mat(i, k) = arma::det(out(i).slice(k));
    }
  }
  return det_mat;
}

void WaterGeometry(
    arma::cube &qcart, const double r1, const double r2, const double th) {
  assert(qcart.n_slices % 3 == 0);
  // make sure ringpolymer is collapsed on centroid
  arma::cube centroid = CentroidCube(qcart);
  arma::cube qcart_copy = qcart;
  for (arma::uword i = 0; i < qcart.n_rows; i++) {
    qcart.row(i) = centroid.row(0);
  }
  // rotation around y axis matrix
  arma::mat rotation_mat = {
      {std::cos(th), 0.0, std::sin(th)},
      {0.0, 1.0, 0.0},
      {-std::sin(th), 0.0, std::cos(th)}};
  for (arma::uword i = 0; i < qcart.n_slices; i += 3) {
    // extend H1 along x axis
    arma::mat temp = arma::mat(arma::size(qcart.slice(i)), arma::fill::zeros);
    temp.col(0) += r1;
    qcart.slice(i + 1) = qcart.slice(i) + temp;
    arma::mat OH2 = BondMat(qcart, i, i + 1);
    OH2.col(0) += r2 - r1;
    // rotate OH2 around y axis and add bond matrix to oxygen position to give
    // position of H2
    for (arma::uword j = 0; j < qcart.n_rows; j++) {
      qcart.slice(i + 2).row(j) =
          ((rotation_mat * OH2.row(j).t()).t()) + qcart.slice(i).row(j);
    }
  }
}

