#pragma once
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <memory>

#include "../../ringpolymer/QuasiCentroidRP.hpp"

/**
 * @brief Base class for triatomic/tetraatomic QCMD implementation
 */
class QCMD {
 public:
  QCMD(std::shared_ptr<QuasiCentroidRP>);

  std::shared_ptr<QuasiCentroidRP> QC_RP;

  virtual arma::vec Constraints(const arma::cube &, const arma::cube &);
  virtual arma::field<arma::cube> ConstraintsGradient(
      const arma::cube &, const arma::cube &, const arma::cube &);
  virtual void update_qc_force();

  arma::mat GramianMat(
      const arma::field<arma::cube> &, const arma::field<arma::cube> &);
  arma::vec RattleVec(const arma::cube &, const arma::field<arma::cube> &);
};
