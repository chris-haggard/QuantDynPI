#pragma once
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "Potential.hpp"

// requires OHH ordering
class GasQTIP4P : public Potential {
 protected:
  // convert the two below??
  const double e = 0.1825;
  const double sigma = 3.1589;
  //***
  const double q_M = -1.1128;     // charge on the M site (not oxygen)
  const double q_H = -q_M / 2.0;  // charge on each H
  const double gamma =
      0.73612;  // fractional position of M site along O to com vector
  const double D_r = 0.185;                   // 116.09 kcal/mol
  const double alpha_r = 1.21;                // 2.287 angstrom^-1
  const double r_eq = 1.78;                   // 0.9419 angstrom
  const double k_theta = 0.14;                // 87.85 kcal/mol rad^2
  const double theta_eq = 1.874483616641901;  // 107.4 degrees
 private:
  arma::cube AngleForce(const arma::cube &);
  arma::cube BondForce(const arma::cube &);
  arma::vec BondPot(const arma::vec &);
  arma::vec AnglePot(const arma::vec &);

 public:
  GasQTIP4P(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  arma::cube IntramolecularForce(const arma::cube &);
  arma::cube Force(const arma::cube &) override;
  arma::cube Dipole(const arma::cube &) override;
  arma::vec Pot(const arma::cube &) override;
};
