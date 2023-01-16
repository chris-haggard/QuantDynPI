#pragma once

#include <bspline.h>
#include <bsplinebuilder.h>
#include <datatable.h>

#include "PartridgeSchwenke.hpp"
#include "PartridgeSchwenkeWrappers.hpp"

class SplinePot : public PartridgeSchwenke {
 public:
  SplinePot(
      const InputHandler::SimulationInput &,
      const InputHandler::PotentialInput &);
  arma::cube Force(const arma::cube &) override;
  arma::vec Pot(const arma::cube &) override;

 private:
  std::vector<std::string> spline_names;
  std::string R1_spline_fname;
  std::string R2_spline_fname;
  std::string Theta_spline_fname;
  SPLINTER::BSpline R1_spline;
  SPLINTER::BSpline R2_spline;
  SPLINTER::BSpline Theta_spline;
};
