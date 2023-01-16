#pragma once

#include <cstddef>
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>

class TCF {
 public:
  TCF(const arma::cube &, size_t, size_t, size_t, size_t);
  const size_t stride;
  const size_t tcf_steps;
  virtual void Record(const arma::cube &, size_t pos);
  virtual void CorrelateTCFs();
  virtual void NormaliseTCFs(const unsigned int);
  arma::cube storage;
  arma::vec tcf;
  arma::mat out_buffer;
};
