#pragma once
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>

arma::mat BondMat(const arma::cube &, const size_t, const size_t);
arma::vec BondLength(const arma::mat &);
arma::vec CosThetaVec(const arma::mat &, const arma::mat &);
arma::vec SinThetaVec(const arma::mat &, const arma::mat &);
arma::vec SinThetaVec(const arma::vec &);
arma::vec CosThetaVec(
    const arma::mat &, const arma::mat &, const arma::vec &, const arma::vec &);
arma::cube CubeCross(const arma::cube &, const arma::cube &);
arma::cube CentroidCube(const arma::cube &);
arma::mat COM(const arma::cube &, const arma::cube &);
arma::rowvec InitializeMomenta(
    const arma::rowvec &, const arma::rowvec &, const double &);
arma::mat AmmoniaOrientation(const arma::cube &);
void WaterGeometry(arma::cube &, const double, const double, const double);

