#pragma once
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <functional>

#include "Input.hpp"

class Ensemble {
 public:
  Ensemble(const InputHandler::EnsembleInput &);
  const double beta;
  double V_estimator(const arma::vec &);
  double Virial_estimator(const arma::cube &, const arma::cube &);
};
