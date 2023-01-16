#include "SplinePot.hpp"
#include <cstring>
#include <stdexcept>

#include "../../util/RingPolymerUtils.hpp"
#include "../../util/IO_utils.hpp"

SplinePot::SplinePot(
    const InputHandler::SimulationInput &SimParams,
    const InputHandler::PotentialInput &PotParams)
    : PartridgeSchwenke(SimParams, PotParams),
      R1_spline_fname("splines/R1_CMD.spline"),
      R2_spline_fname("splines/R2_CMD.spline"),
      Theta_spline_fname("splines/Theta_CMD.spline"),
      R1_spline(R1_spline_fname),
      R2_spline(R2_spline_fname),
      Theta_spline(Theta_spline_fname) {
}

arma::cube SplinePot::Force(const arma::cube &q) {
  assert(q.n_rows == 1);
  arma::cube out(arma::size(q));
  for (arma::uword i = 0; i < q.n_slices; i += 3) {
    arma::mat OH_1 = BondMat(q, i, i + 1);
    arma::mat OH_2 = BondMat(q, i, i + 2);
    double R_1 = arma::as_scalar(BondLength(OH_1));
    double R_2 = arma::as_scalar(BondLength(OH_2));
    double cos_Theta = arma::as_scalar(CosThetaVec(OH_1, OH_2));

    std::vector<double> temp = {R_1, R_2, cos_Theta};

    double f_R_1 = R1_spline.eval(temp);
    double f_R_2 = R1_spline.eval(temp);
    double f_R_Th = Theta_spline.eval(temp);

    out.slice(i + 1) = f_R_1 * (OH_1 / R_1);
    out.slice(i + 1) -=
        (f_R_Th / (std::sqrt(1.0 - std::pow(cos_Theta, 2)) * R_1)) *
        ((OH_2 / R_2) - ((OH_1 / R_1) * cos_Theta));
    out.slice(i + 2) = f_R_2 * (OH_2 / R_2);
    out.slice(i + 2) -=
        (f_R_Th / (std::sqrt(1.0 - std::pow(cos_Theta, 2)) * R_2)) *
        ((OH_1 / R_1) - ((OH_2 / R_2) * cos_Theta));
    out.slice(i) = -(out.slice(i + 1) + out.slice(i + 2));
  }
  return out;
}

arma::vec SplinePot::Pot(const arma::cube &q) {
  throw std::runtime_error("No potential in spline potential file.");
}

