#pragma once
#include <stdexcept>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace pt = boost::property_tree;

/**
 * @brief Class to read json formatted input files and load to property tree.
 *
 */
class JsonInputLoader {
 public:
  pt::ptree tree;
  void load(const std::string &);
};

struct InputHandler {
  /**
   * @brief Template function to check that parameter is supplied in input file.
   * Property tree sets default value on parameters that are not supplied to a
   * value that evaluates to false (i.e. 0, 0.0, false)
   *
   * obvious problem is that some value do actually need to be set to false:
   * i.e. thermostat_centroid
   *
   * @tparam T
   * @param x
   */
  template <typename T>
  void ValidateInput(T x, const std::string s) {
    if (!x) {
      throw std::runtime_error(
          "Parameter <" + s + "> not set and is required. Exiting.\n");
    }
  }

  /**
   * @brief Overloading template function to deal with strings
   *
   * @param str_parameter
   * @param s
   */
  void ValidateInput(const std::string str_parameter, const std::string s) {
    if (str_parameter.empty()) {
      throw std::runtime_error(
          "Parameter <" + s + "> not set and is required. Exiting.\n");
    }
  }
  struct SimulationInput {
    SimulationInput(const pt::ptree &);
    std::string sim_type;
    std::string init_file;
    double box_length;
    unsigned int n_traj;
    double initial_therm_length;
    double therm_length;
    double run_length;
    size_t stride;
  };
  struct EnsembleInput {
    EnsembleInput(const pt::ptree &);
    double T;
  };
  struct RingPolymerInput {
    RingPolymerInput(const pt::ptree &);
    unsigned int n_beads;
    unsigned int n_atoms;
    unsigned int n_molec;
    double gamma;
    arma::cube position_from_init;
    std::vector<std::string> labels_from_init;
  };
  struct ThermostatInput {
    ThermostatInput(const pt::ptree &);
    double seed;
    double PILE_L_lambda;
    double PILE_tau0;
    double Quasi_tau0;
    bool centroid_therm;
  };
  struct PotentialInput {
    PotentialInput(const pt::ptree &);
    std::string potential_type;
    std::string spline_files;
  };
  struct DynamicsInput {
    DynamicsInput(const pt::ptree &);
    double dt;
  };

  SimulationInput SimulationParameters;
  RingPolymerInput RingPolymerParameters;
  EnsembleInput EnsembleParameters;
  ThermostatInput ThermostatParameters;
  PotentialInput PotentialParameters;
  DynamicsInput DynamicsParameters;
  InputHandler(const pt::ptree &);
};
