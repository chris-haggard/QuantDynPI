#include "TetraatomicQCMD.hpp"

#include <stdexcept>

#include "../../../util/RingPolymerUtils.hpp"
#include "TriatomicQCMD.hpp"

TetraatomicQCMD::TetraatomicQCMD(std::shared_ptr<QuasiCentroidRP> qc_rp)
    : QCMD(qc_rp) {
  std::cout << "TetratomicQCMD constructor\n";
  assert(QC_RP->n_atoms == 4);
}

arma::mat TetraatomicQCMD::PhiABC(
    const arma::vec &cos_theta_vec_A, const arma::mat &bondB,
    const arma::mat &bondC) {
  return (cos_theta_vec_A % bondB.each_col()) - bondC;
}

arma::mat TetraatomicQCMD::AlpABC(
    const arma::mat &phi_mat_A, const arma::vec &sin_theta_vec_B,
    const arma::vec &bond_length_C) {
  return phi_mat_A.each_col() / (sin_theta_vec_B % bond_length_C);
}

arma::mat TetraatomicQCMD::XiABC(
    const arma::vec &thA, const arma::mat &alpB, const arma::mat &alpC) {
  return (thA % alpB.each_col()) - alpC;
}

/**
 * @brief Convert the cartesian forces on the beads to the forces on the
 * curvilinear centroids f_R_1, f_R_2, f_R_3, f_th_1, f_th_2, f_th_3
 *
 * Requires: HHHN ordering
 * ordering
 *
 * @return
 */
arma::cube TetraatomicQCMD::CartToCurvlinearForces() {
  arma::uword j;  // index of curvilinear forces cube
  arma::cube out(
      1, 1, 6 * QC_RP->n_molec);  // 6 curvilinear centroids per molecule
  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += QC_RP->n_atoms) {
    assert(i % 4 == 0);
    j = (i / 4) * 6;

    arma::mat NH_1 = BondMat(QC_RP->position_cart, i + 3, i);
    arma::mat NH_2 = BondMat(QC_RP->position_cart, i + 3, i + 1);
    arma::mat NH_3 = BondMat(QC_RP->position_cart, i + 3, i + 2);
    arma::vec r_1 = BondLength(NH_1);
    arma::vec r_2 = BondLength(NH_2);
    arma::vec r_3 = BondLength(NH_3);
    arma::vec cos_theta1 = CosThetaVec(NH_2, NH_3);
    arma::vec cos_theta2 = CosThetaVec(NH_1, NH_3);
    arma::vec cos_theta3 = CosThetaVec(NH_1, NH_2);
    arma::vec sin_theta1 = SinThetaVec(cos_theta1);
    arma::vec sin_theta2 = SinThetaVec(cos_theta2);
    arma::vec sin_theta3 = SinThetaVec(cos_theta3);

    NH_1.each_col() /= r_1;
    NH_2.each_col() /= r_2;
    NH_3.each_col() /= r_3;

    arma::mat phi123 = PhiABC(cos_theta1, NH_2, NH_3);
    arma::mat phi231 = PhiABC(cos_theta2, NH_3, NH_1);
    arma::mat phi312 = PhiABC(cos_theta3, NH_1, NH_2);
    arma::mat phi132 = PhiABC(cos_theta1, NH_3, NH_2);
    arma::mat phi213 = PhiABC(cos_theta2, NH_1, NH_3);
    arma::mat phi321 = PhiABC(cos_theta3, NH_2, NH_1);

    arma::mat alp123 = AlpABC(phi123, sin_theta1, r_2);
    arma::mat alp231 = AlpABC(phi231, sin_theta2, r_3);
    arma::mat alp312 = AlpABC(phi312, sin_theta3, r_1);
    arma::mat alp132 = AlpABC(phi132, sin_theta1, r_3);
    arma::mat alp213 = AlpABC(phi213, sin_theta2, r_1);
    arma::mat alp321 = AlpABC(phi321, sin_theta3, r_2);

    arma::vec th213 =
        (cos_theta2 - (cos_theta1 % cos_theta3)) / (sin_theta1 % sin_theta3);
    arma::vec th321 =
        (cos_theta3 - (cos_theta2 % cos_theta1)) / (sin_theta2 % sin_theta1);
    arma::vec th132 =
        (cos_theta1 - (cos_theta3 % cos_theta2)) / (sin_theta3 % sin_theta2);

    arma::mat xi213 = XiABC(th213, alp321, alp123);
    arma::mat xi321 = XiABC(th321, alp132, alp231);
    arma::mat xi132 = XiABC(th132, alp213, alp312);

    // f_R_1 = (NH_1 / r_1) * f_H_1 and similar for f_R_2, f_R_3
    out.slice(j) = arma::as_scalar(
        arma::mean(arma::sum(NH_1 % QC_RP->force_cart.slice(i), 1), 0));
    out.slice(j + 1) = arma::as_scalar(
        arma::mean(arma::sum(NH_2 % QC_RP->force_cart.slice(i + 1), 1), 0));
    out.slice(j + 2) = arma::as_scalar(
        arma::mean(arma::sum(NH_3 % QC_RP->force_cart.slice(i + 2), 1), 0));

    out.slice(j + 3) = arma::mean(
        arma::sum(
            (xi213.each_col() %
             (arma::square(r_2) / (arma::square(th213) - 1.0))) %
                QC_RP->force_cart.slice(i + 1),
            1),
        0);

    out.slice(j + 4) = arma::mean(
        arma::sum(
            (xi321.each_col() %
             (arma::square(r_3) / (arma::square(th321) - 1.0))) %
                QC_RP->force_cart.slice(i + 2),
            1),
        0);

    out.slice(j + 5) = arma::mean(
        arma::sum(
            (xi132.each_col() %
             (arma::square(r_1) / (arma::square(th132) - 1.0))) %
                QC_RP->force_cart.slice(i),
            1),
        0);
  }
  return out;
}

arma::cube TetraatomicQCMD::CurvilinearToQCCartForces() {
  arma::cube out(arma::size(QC_RP->qc_force));
  double f_R_1, f_R_2, f_R_3, f_th_1, f_th_2, f_th_3;
  arma::uword j;  // starting index in curivliear_forces for molecule
  arma::cube curivlinear_forces =
      CartToCurvlinearForces();  // <-- 6 slices per molec
  for (arma::uword i = 0; i < QC_RP->n_total_atoms; i += QC_RP->n_atoms) {
    assert(i % 4 == 0);
    j = (i / 4) * 6;

    // get the curvilinear forces
    f_R_1 = arma::as_scalar(curivlinear_forces(0, 0, j));
    f_R_2 = arma::as_scalar(curivlinear_forces(0, 0, j + 1));
    f_R_3 = arma::as_scalar(curivlinear_forces(0, 0, j + 2));
    f_th_1 = arma::as_scalar(curivlinear_forces(0, 0, j + 3));
    f_th_2 = arma::as_scalar(curivlinear_forces(0, 0, j + 4));
    f_th_3 = arma::as_scalar(curivlinear_forces(0, 0, j + 5));

    arma::mat QC_NH_1 = BondMat(QC_RP->qc_position, i + 3, i);
    arma::mat QC_NH_2 = BondMat(QC_RP->qc_position, i + 3, i + 1);
    arma::mat QC_NH_3 = BondMat(QC_RP->qc_position, i + 3, i + 2);
    double R_1 = arma::as_scalar(BondLength(QC_NH_1));
    double R_2 = arma::as_scalar(BondLength(QC_NH_2));
    double R_3 = arma::as_scalar(BondLength(QC_NH_3));
    double cos_Theta1 = arma::as_scalar(CosThetaVec(QC_NH_2, QC_NH_3));
    double cos_Theta2 = arma::as_scalar(CosThetaVec(QC_NH_1, QC_NH_3));
    double cos_Theta3 = arma::as_scalar(CosThetaVec(QC_NH_1, QC_NH_2));
    double sin_Theta1 = std::sqrt(1.0 - (cos_Theta1 * cos_Theta1));
    double sin_Theta2 = std::sqrt(1.0 - (cos_Theta2 * cos_Theta2));
    double sin_Theta3 = std::sqrt(1.0 - (cos_Theta3 * cos_Theta3));

    QC_NH_1 /= R_1;
    QC_NH_2 /= R_2;
    QC_NH_3 /= R_3;

    out.slice(i) =
        (QC_NH_1 * f_R_1) +
        ((f_th_2 / (R_1 * sin_Theta2)) * ((QC_NH_1 * cos_Theta2) - QC_NH_3)) +
        ((f_th_3 / (R_1 * sin_Theta3)) * ((QC_NH_1 * cos_Theta3) - QC_NH_2));

    out.slice(i + 1) =
        (QC_NH_2 * f_R_2) +
        ((f_th_3 / (R_2 * sin_Theta3)) * ((QC_NH_2 * cos_Theta3) - QC_NH_1)) +
        ((f_th_1 / (R_2 * sin_Theta1)) * ((QC_NH_2 * cos_Theta1) - QC_NH_3));

    out.slice(i + 2) =
        (QC_NH_3 * f_R_3) +
        ((f_th_1 / (R_3 * sin_Theta1)) * ((QC_NH_3 * cos_Theta1) - QC_NH_2)) +
        ((f_th_2 / (R_3 * sin_Theta2)) * ((QC_NH_3 * cos_Theta2) - QC_NH_1));

    out.slice(i + 3) = -(out.slice(i) + out.slice(i + 1) + out.slice(i + 2));
  }

  return out;
}

void TetraatomicQCMD::update_qc_force() {
  QC_RP->qc_force = CurvilinearToQCCartForces();
}

arma::vec TetraatomicQCMD::Constraints(
    const arma::cube &q, const arma::cube &qc_q) {
  arma::vec out(12, arma::fill::zeros);

  double m_tot =
      arma::accu(QC_RP->qc_mass.subcube(0, 0, 0, 0, 0, QC_RP->n_atoms - 1));

  arma::mat NH_1 = BondMat(q, 3, 0);
  arma::mat NH_2 = BondMat(q, 3, 1);
  arma::mat NH_3 = BondMat(q, 3, 2);
  arma::vec r_1 = BondLength(NH_1);
  arma::vec r_2 = BondLength(NH_2);
  arma::vec r_3 = BondLength(NH_3);
  arma::vec theta1 = arma::acos(CosThetaVec(NH_2, NH_3));
  arma::vec theta2 = arma::acos(CosThetaVec(NH_1, NH_3));
  arma::vec theta3 = arma::acos(CosThetaVec(NH_1, NH_2));

  arma::mat QC_NH_1 = BondMat(qc_q, 3, 0);
  arma::mat QC_NH_2 = BondMat(qc_q, 3, 1);
  arma::mat QC_NH_3 = BondMat(qc_q, 3, 2);
  double R_1 = arma::as_scalar(BondLength(QC_NH_1));
  double R_2 = arma::as_scalar(BondLength(QC_NH_2));
  double R_3 = arma::as_scalar(BondLength(QC_NH_3));
  double Theta1 = arma::as_scalar(arma::acos(CosThetaVec(QC_NH_2, QC_NH_3)));
  double Theta2 = arma::as_scalar(arma::acos(CosThetaVec(QC_NH_1, QC_NH_3)));
  double Theta3 = arma::as_scalar(arma::acos(CosThetaVec(QC_NH_1, QC_NH_2)));

  arma::mat qc_q_com = COM(qc_q, QC_RP->qc_mass.slices(0, 3));
  arma::cube temp = qc_q.each_slice() - qc_q_com;

  arma::cube eckart_reference = (CentroidCube(q) - qc_q);
  arma::cube eckart_com_constraint =
      QC_RP->qc_mass.slices(0, 3) % eckart_reference;

  // Eq. 34b
  // not clear why we set com as origin here
  arma::cube eckart_orientation_constraint =
      CubeCross(QC_RP->qc_mass.slices(0, 3) % temp, eckart_reference);

  out(0) = arma::as_scalar(arma::mean(r_1)) - R_1;
  out(1) = arma::as_scalar(arma::mean(r_2)) - R_2;
  out(2) = arma::as_scalar(arma::mean(r_3)) - R_3;
  out(3) = arma::as_scalar(arma::mean(theta1) - Theta1);
  out(4) = arma::as_scalar(arma::mean(theta2) - Theta2);
  out(5) = arma::as_scalar(arma::mean(theta3) - Theta3);
  out(6) = arma::accu(eckart_com_constraint.col(0)) / m_tot;
  out(7) = arma::accu(eckart_com_constraint.col(1)) / m_tot;
  out(8) = arma::accu(eckart_com_constraint.col(2)) / m_tot;
  out(9) = arma::accu(eckart_orientation_constraint.col(0)) / m_tot;
  out(10) = arma::accu(eckart_orientation_constraint.col(1)) / m_tot;
  out(11) = arma::accu(eckart_orientation_constraint.col(2)) / m_tot;

  return out;
}

arma::field<arma::cube> TetraatomicQCMD::ConstraintsGradient(
    const arma::cube &q, const arma::cube &qc_q, const arma::cube &m) {
  arma::field<arma::cube> out(12);
  arma::cube field_elements(arma::size(q), arma::fill::zeros);

  for (arma::uword i = 0; i < out.n_elem; i++) {
    out(i) = field_elements;
  }
  double m_tot =
      arma::accu(QC_RP->qc_mass.subcube(0, 0, 0, 0, 0, QC_RP->n_atoms - 1));

  arma::mat basis(3, 3, arma::fill::eye);

  arma::mat NH_1 = BondMat(q, 3, 0);
  arma::mat NH_2 = BondMat(q, 3, 1);
  arma::mat NH_3 = BondMat(q, 3, 2);
  arma::vec r_1 = BondLength(NH_1);
  arma::vec r_2 = BondLength(NH_2);
  arma::vec r_3 = BondLength(NH_3);
  arma::vec cos_theta1 = CosThetaVec(NH_2, NH_3);
  arma::vec cos_theta2 = CosThetaVec(NH_1, NH_3);
  arma::vec cos_theta3 = CosThetaVec(NH_1, NH_2);
  arma::vec sin_theta1 = SinThetaVec(cos_theta1);
  arma::vec sin_theta2 = SinThetaVec(cos_theta2);
  arma::vec sin_theta3 = SinThetaVec(cos_theta3);

  NH_1.each_col() /= r_1;
  NH_2.each_col() /= r_2;
  NH_3.each_col() /= r_3;

  out(0).slice(3) = -NH_1;
  out(0).slice(0) = NH_1;

  out(1).slice(3) = -NH_2;
  out(1).slice(1) = NH_2;

  out(2).slice(3) = -NH_3;
  out(2).slice(2) = NH_3;

  // theta1 constraint
  out(3).slice(1) = (cos_theta1 % NH_2.each_col()) -
                    (NH_3.each_col() / (r_2 % sin_theta1));  // H2
  out(3).slice(2) =
      (cos_theta1 % NH_3.each_col()) - (NH_2.each_col() / (r_3 % sin_theta1));
  ;  // H3
  out(3).slice(3) = -(out(3).slice(1) + out(3).slice(2));

  // theta2
  out(4).slice(2) = (cos_theta2 % NH_3.each_col()) -
                    (NH_1.each_col() / (r_3 % sin_theta2));  // H3
  out(4).slice(0) =
      (cos_theta2 % NH_1.each_col()) - (NH_3.each_col() / (r_1 % sin_theta2));
  ;  // H1
  out(4).slice(3) = -(out(4).slice(2) + out(4).slice(0));

  // theta3
  out(5).slice(0) = (cos_theta3 % NH_1.each_col()) -
                    (NH_2.each_col() / (r_1 % sin_theta3));  // H1
  out(5).slice(1) =
      (cos_theta3 % NH_2.each_col()) - (NH_1.each_col() / (r_2 % sin_theta3));
  ;  // H2
  out(5).slice(3) = -(out(4).slice(0) + out(4).slice(1));
  // derivative of Eckart rotation constraint
  // more efficient way of doing this by cycling through indicies
  arma::mat qc_q_com = COM(qc_q, QC_RP->qc_mass.slices(0, 3));
  arma::cube temp = qc_q.each_slice() - qc_q_com;
  arma::vec res(3);
  for (arma::uword s = 0; s < q.n_slices; s++) {
    for (arma::uword i = 0; i < q.n_rows; i++) {
      for (arma::uword j = 0; j < q.n_cols; j++) {
        res = arma::cross(
            arma::as_scalar(m(i, j, s)) * temp.slice(s).t(), basis.col(j));
        out(9)(i, j, s) = res(0);
        out(10)(i, j, s) = res(1);
        out(11)(i, j, s) = res(2);
      }
    }
  }

  out(9) /= m_tot;
  out(10) /= m_tot;
  out(11) /= m_tot;

  // derivative of the Eckart COM constraint
  out(6).col(0) = m.col(0) / m_tot;
  out(7).col(1) = m.col(1) / m_tot;
  out(8).col(2) = m.col(2) / m_tot;
  return out;
}

