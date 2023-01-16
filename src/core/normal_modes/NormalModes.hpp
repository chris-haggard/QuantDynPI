#pragma once

#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>

#include "mkl_dfti.h"

class NormalModes {
 public:
  NormalModes(const arma::cube &);

  DFTI_DESCRIPTOR_HANDLE desc_handle;
  MKL_LONG status;

  void CartToNM(arma::cube &, arma::cube &);
  void NMToCart(arma::cube &, arma::cube &);
  arma::cube ForwardScale;
  arma::cube BackwardScale;
};

// ******* Testing functions only, do not use
arma::mat CartToNM_TEST(const arma::mat &);
arma::mat CartToNM_TEST2(const arma::mat &);
arma::mat NMToCart_TEST(const arma::mat &);

// get rid of these at some point they are no longer needed
arma::mat convToNormalModes(const arma::mat &);
arma::mat convToCartesian(const arma::mat &);  // this is broken as armadillo
                                               // only accepts cx_mat for ifft
