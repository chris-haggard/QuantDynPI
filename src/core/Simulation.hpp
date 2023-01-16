#pragma once
#include <memory>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "Ensemble.hpp"
#include "Input.hpp"
#include "dynamics/Dynamics.hpp"
#include "ringpolymer/RingPolymer.hpp"

class Simulation {
 public:
  Simulation(const InputHandler &, Ensemble &);
  Simulation(const InputHandler &, Ensemble &, std::string);
  const double initial_therm_length;
  const double therm_length;
  const double run_length;
  const unsigned int n_traj;
  const size_t stride;
  // shared_ptr as Dynamics needs to hold one as well
  std::shared_ptr<RingPolymer> RP;
  std::unique_ptr<Dynamics> Dyn;
};
