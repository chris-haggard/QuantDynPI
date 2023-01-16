#pragma once

#include "QCMD.hpp"

class TriatomicQCMD : public QCMD {
 public:
  TriatomicQCMD(std::shared_ptr<QuasiCentroidRP>);
  void update_qc_force() override;
  arma::vec Constraints(const arma::cube &, const arma::cube &) override;
  arma::field<arma::cube> ConstraintsGradient(
      const arma::cube &, const arma::cube &, const arma::cube &) override;

  arma::cube FetchQuasi();

 private:
  arma::cube CartToCurvlinearForces();
  arma::cube CurvilinearToQCCartForces();
};
