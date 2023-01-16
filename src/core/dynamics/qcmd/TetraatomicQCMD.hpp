#pragma once

#include "QCMD.hpp"

class TetraatomicQCMD : public QCMD {
 public:
  TetraatomicQCMD(std::shared_ptr<QuasiCentroidRP>);
  void update_qc_force() override;
  arma::vec Constraints(const arma::cube &, const arma::cube &) override;
  arma::field<arma::cube> ConstraintsGradient(
      const arma::cube &, const arma::cube &, const arma::cube &) override;

  arma::cube FetchQuasi();

 private:
  arma::mat PhiABC(const arma::vec &, const arma::mat &, const arma::mat &);
  arma::mat AlpABC(const arma::mat &, const arma::vec &, const arma::vec &);
  arma::mat XiABC(const arma::vec &, const arma::mat &, const arma::mat &);
  arma::cube CartToCurvlinearForces();
  arma::cube CurvilinearToQCCartForces();
};
