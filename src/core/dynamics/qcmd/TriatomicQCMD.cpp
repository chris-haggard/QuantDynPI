// see doi.org/10.17863/CAM.54877 & 10.1063/1.5100587

#include "TriatomicQCMD.hpp"

#include <stdexcept>

#include "../../../util/RingPolymerUtils.hpp"

TriatomicQCMD::TriatomicQCMD(std::shared_ptr<QuasiCentroidRP> qc_rp)
    : QCMD(qc_rp) {
  std::cout << "TriatomicQCMD constructor\n";
  // must be water/triatomic
  assert(QC_RP->n_atoms == 3);
  QC_RP->qc_position.print("qc position");
}

/**
 * @brief Given a bead configuration find the cartesian quasicentroids. Used to
 * initially populate QC_RP->qc_position;
 *
 * Not finished, currently just initialise at centroids
 *
 * @return
 */
arma::cube TriatomicQCMD::FetchQuasi() {
  arma::cube out(arma::size(QC_RP->qc_position), arma::fill::zeros);
  // arma::cube centroid_com = QC_RP->CentroidCOM();
  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += QC_RP->n_atoms) {
    arma::mat OH_1 = BondMat(QC_RP->position_cart, i, i + 1);
    arma::mat OH_2 = BondMat(QC_RP->position_cart, i, i + 2);
    double R_1 = arma::mean(BondLength(OH_1));
    double R_2 = arma::mean(BondLength(OH_2));
    double Theta = arma::mean(arma::acos(CosThetaVec(OH_1, OH_2)));
    // H_1 on x-axis
    out(0, 0, i + 1) = R_1;
    out(0, 0, i + 2) = R_2 * std::cos(Theta);
    out(0, 1, i + 2) = R_2 * std::sin(Theta);
  }
  throw std::runtime_error("not implemented full\n");
  return out;
}

/**
 * @brief Converts the cartesian forces on the beads to forces on the
 * curivlinear centorids f_R_k and f_Theta. Requires OHH ordering. Requires that
 * RP.force_cart is up to date.
 *
 * Tested against qclpysim, gives same results.
 *
 * @return cube of forces ordered by slice: f_R_1, f_R_2, f_R_Theta for each
 * molecule.
 */
arma::cube TriatomicQCMD::CartToCurvlinearForces() {
  arma::cube out(
      1, 1, 3 * QC_RP->n_molec);  // 3 curvilinear centroids per molecule
  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += 3) {
    arma::mat OH_1 = BondMat(QC_RP->position_cart, i, i + 1);
    arma::mat OH_2 = BondMat(QC_RP->position_cart, i, i + 2);
    arma::vec r_1 = BondLength(OH_1);
    arma::vec r_2 = BondLength(OH_2);
    // f_R_1
    out.slice(i) = arma::as_scalar(arma::mean(
        arma::sum(OH_1 % QC_RP->force_cart.slice(i + 1), 1) / r_1, 0));
    // f_R_2
    out.slice(i + 1) = arma::as_scalar(arma::mean(
        arma::sum(OH_2 % QC_RP->force_cart.slice(i + 2), 1) / r_2, 0));
    // f_Theta
    arma::vec cos_theta = CosThetaVec(OH_1, OH_2);

    OH_1.each_col() %= cos_theta;
    OH_1.each_col() /= r_1;
    OH_2.each_col() /= r_2;

    out.slice(i + 2) = arma::as_scalar(arma::mean(
        (r_1 / SinThetaVec(cos_theta)) %
            arma::sum((OH_2 - (OH_1)) % -QC_RP->force_cart.slice(i + 1), 1),
        0));
  }
  return out;
}

/**
 * @brief Convert curivlinear forces (f_R_k, f_Theta) to forces on the cartesian
 * quasicentroids (Q_O, Q_H_1, Q_H_2)
 *
 * Tested against qclpysim, gives same results.
 *
 * @return
 */
arma::cube TriatomicQCMD::CurvilinearToQCCartForces() {
  arma::cube out(arma::size(QC_RP->qc_force));
  arma::cube curivlinear_forces = CartToCurvlinearForces();
  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += 3) {
    arma::mat QC_OH_1 = BondMat(QC_RP->qc_position, i, i + 1);
    arma::mat QC_OH_2 = BondMat(QC_RP->qc_position, i, i + 2);
    double R_1 = arma::as_scalar(BondLength(QC_OH_1));
    double R_2 = arma::as_scalar(BondLength(QC_OH_2));
    double cos_Theta = arma::as_scalar(CosThetaVec(QC_OH_1, QC_OH_2));
    out.slice(i + 1) = curivlinear_forces.slice(i) * (QC_OH_1 / R_1);
    out.slice(i + 1) -= (curivlinear_forces.slice(i + 2) /
                         (std::sqrt(1.0 - std::pow(cos_Theta, 2)) * R_1)) *
                        ((QC_OH_2 / R_2) - ((QC_OH_1 / R_1) * cos_Theta));
    out.slice(i + 2) = curivlinear_forces.slice(i + 1) * (QC_OH_2 / R_2);
    out.slice(i + 2) -= (curivlinear_forces.slice(i + 2) /
                         (std::sqrt(1.0 - std::pow(cos_Theta, 2)) * R_2)) *
                        ((QC_OH_1 / R_1) - ((QC_OH_2 / R_2) * cos_Theta));
    out.slice(i) = -(out.slice(i + 1) + out.slice(i + 2));
  }
  return out;
}

/**
 * @brief Implement the holonomic constraints for subcube q corresponding to one
 * molecule. Constraints indexed 0-2 are constraining the mean replica bond
 * lengths and angle to that of the quasicentroid system. 4-9 are the Eckart
 * conditions, Eq 34a and 34b in 10.1063/1.5100587.
 *
 * @return
 */
arma::vec TriatomicQCMD::Constraints(
    const arma::cube &q, const arma::cube &qc_q) {
  arma::vec out(9, arma::fill::zeros);

  double m_tot =
      arma::accu(QC_RP->qc_mass.subcube(0, 0, 0, 0, 0, QC_RP->n_atoms - 1));

  arma::mat OH_1 = BondMat(q, 0, 1);
  arma::mat OH_2 = BondMat(q, 0, 2);
  arma::vec r_1 = BondLength(OH_1);
  arma::vec r_2 = BondLength(OH_2);
  arma::vec theta = arma::acos(CosThetaVec(OH_1, OH_2));
  arma::mat QC_OH_1 = BondMat(qc_q, 0, 1);
  arma::mat QC_OH_2 = BondMat(qc_q, 0, 2);
  double R_1 = arma::as_scalar(BondLength(QC_OH_1));
  double R_2 = arma::as_scalar(BondLength(QC_OH_2));
  double Theta = std::acos(arma::as_scalar(CosThetaVec(QC_OH_1, QC_OH_2)));

  arma::mat qc_q_com = COM(qc_q, QC_RP->qc_mass.slices(0, 2));
  arma::cube temp = qc_q.each_slice() - qc_q_com;

  arma::cube eckart_reference = (CentroidCube(q) - qc_q);
  arma::cube eckart_com_constraint =
      QC_RP->qc_mass.slices(0, 2) % eckart_reference;

  // Eq. 34b
  arma::cube eckart_orientation_constraint =
      CubeCross(QC_RP->qc_mass.slices(0, 2) % temp, eckart_reference);

  out(0) = arma::accu(eckart_com_constraint.col(0)) / m_tot;
  out(1) = arma::accu(eckart_com_constraint.col(1)) / m_tot;
  out(2) = arma::accu(eckart_com_constraint.col(2)) / m_tot;
  out(3) = arma::accu(eckart_orientation_constraint.col(0)) / m_tot;
  out(4) = arma::accu(eckart_orientation_constraint.col(1)) / m_tot;
  out(5) = arma::accu(eckart_orientation_constraint.col(2)) / m_tot;
  out(6) = arma::as_scalar(arma::mean(r_1)) - R_1;
  out(7) = arma::as_scalar(arma::mean(r_2)) - R_2;
  out(8) = arma::as_scalar(arma::mean(theta) - Theta);

  return out;
}

/**
 * @brief Gradient of the constriant functions. Each row is a replica, each
 * col a dimension, each slice an atom, and each field a constraint (9
 * constraints)
 *
 * @param q bead positions for molec i.e. QC_RP->position_cart.slices(0, 2)
 * @param qc_q quasicentroid positions
 *
 * @return
 */
arma::field<arma::cube> TriatomicQCMD::ConstraintsGradient(
    const arma::cube &q, const arma::cube &qc_q, const arma::cube &m) {
  arma::field<arma::cube> out(9);
  arma::cube field_elements(arma::size(q), arma::fill::zeros);

  for (arma::uword i = 0; i < out.n_elem; i++) {
    out(i) = field_elements;
  }
  double m_tot =
      arma::accu(QC_RP->qc_mass.subcube(0, 0, 0, 0, 0, QC_RP->n_atoms - 1));

  arma::mat basis(3, 3, arma::fill::eye);

  arma::mat OH_1 = BondMat(q, 0, 1);
  arma::mat OH_2 = BondMat(q, 0, 2);
  arma::vec r_1 = BondLength(OH_1);
  arma::vec r_2 = BondLength(OH_2);
  arma::vec cos_theta = CosThetaVec(OH_1, OH_2);
  arma::vec sin_theta = SinThetaVec(cos_theta);
  arma::vec theta = arma::acos(cos_theta);

  OH_1.each_col() /= r_1;
  OH_2.each_col() /= r_2;

  out(6).slice(0) = -OH_1;
  out(6).slice(1) = OH_1;

  out(7).slice(0) = -OH_2;
  out(7).slice(2) = OH_2;

  arma::mat d_theta_d_H_1 = ((OH_1.each_col() % cos_theta) - OH_2);
  d_theta_d_H_1.each_col() %= (1.0 / (r_1 % sin_theta));

  out(8).slice(1) = d_theta_d_H_1;

  arma::mat d_theta_d_H_2 = ((OH_2.each_col() % cos_theta) - (OH_1));
  d_theta_d_H_2.each_col() %= (1.0 / (r_2 % sin_theta));

  out(8).slice(2) = d_theta_d_H_2;
  out(8).slice(0) = -(out(8).slice(1) + out(8).slice(2));

  // derivative of Eckart rotation constraint
  // more efficient way of doing this by cycling through indicies
  arma::mat qc_q_com = COM(qc_q, QC_RP->qc_mass.slices(0, 2));
  arma::cube temp = qc_q.each_slice() - qc_q_com;
  arma::vec res(3);
  for (arma::uword s = 0; s < q.n_slices; s++) {
    for (arma::uword i = 0; i < q.n_rows; i++) {
      for (arma::uword j = 0; j < q.n_cols; j++) {
        res = arma::cross(
            arma::as_scalar(m(i, j, s)) * temp.slice(s).t(), basis.col(j));
        out(3)(i, j, s) = res(0);
        out(4)(i, j, s) = res(1);
        out(5)(i, j, s) = res(2);
      }
    }
  }

  out(3) /= m_tot;
  out(4) /= m_tot;
  out(5) /= m_tot;

  // derivative of the Eckart COM constraint
  out(0).col(0) = m.col(0) / m_tot;
  out(1).col(1) = m.col(1) / m_tot;
  out(2).col(2) = m.col(2) / m_tot;

  return out;
}

void TriatomicQCMD::update_qc_force() {
  QC_RP->qc_force = CurvilinearToQCCartForces();
}
