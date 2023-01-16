#include "Input.hpp"
#include <string>

#include "../util/IO_utils.hpp"
#include "../util/unit_conversion.hpp"
#include "thermostat/Thermostat.hpp"

/**
 * @brief Read json input file and save to instance of property tree.
 *
 * @param filename
 */
void JsonInputLoader::load(const std::string &filename) {
  pt::read_json(filename, tree);
}

InputHandler::SimulationInput::SimulationInput(const pt::ptree &tree)
    : box_length(tree.get("simulation.box_length", 0.0)) {
  sim_type = tree.get<std::string>("simulation.type");
  init_file = tree.get<std::string>("simulation.init_file");
  n_traj = tree.get<size_t>("simulation.trajectories");
  initial_therm_length =
      fs_au(tree.get<double>("simulation.initial_therm_length"));
  therm_length = fs_au(tree.get<double>("simulation.therm_length"));
  run_length = fs_au(tree.get<double>("simulation.run_length"));
  stride = tree.get<size_t>("simulation.stride");
}

InputHandler::RingPolymerInput::RingPolymerInput(const pt::ptree &tree)
    : gamma(tree.get("ring-polymer.gamma", 0.0)) {
  n_beads = tree.get<size_t>("ring-polymer.beads");
  n_atoms = tree.get<size_t>("ring-polymer.atoms");
  n_molec = tree.get<size_t>("ring-polymer.molecules");
  position_from_init = arma::cube(n_beads, 3, n_molec * n_atoms);
  labels_from_init = std::vector<std::string>(n_molec * n_atoms);
}

InputHandler::EnsembleInput::EnsembleInput(const pt::ptree &tree) {
  T = tree.get<double>("ensemble.T");
}

/**
 * @brief Construct a new Input Handler:: Thermostat Input:: Thermostat Input
 * object
 *
 * Assigns property tree values to class members. Defaults to a false value if
 * the parameter is not found in the property tree/input file.
 *
 * @param tree
 */
InputHandler::ThermostatInput::ThermostatInput(const pt::ptree &tree) {
  seed = tree.get<double>("thermostat.seed");
  PILE_L_lambda = tree.get<double>("thermostat.PILE.lambda");
  PILE_tau0 = fs_au(tree.get<double>("thermostat.PILE.tau0"));
  Quasi_tau0 = fs_au(tree.get<double>("thermostat.PILE.quasi_tau0"));
  centroid_therm = tree.get<bool>("thermostat.thermostat_centroid_in_dynamics");
}

InputHandler::PotentialInput::PotentialInput(const pt::ptree &tree) {
  potential_type = tree.get<std::string>("potential.type");
  spline_files = tree.get<std::string>("potential.spline_files", "");
}

InputHandler::DynamicsInput::DynamicsInput(const pt::ptree &tree) {
  dt = fs_au(tree.get<double>("dynamics.dt"));
}

/**
 * @brief Construct a new Input Handler:: Input Handler object
 *
 * Input is property tree created from reading json input file. Constructs
 * other structs that contain member variables corresponding to values in
 * property tree. Then checks that input values are valid (must be done here
 * due to access rights of nested classes).
 *
 * @param tree
 */
InputHandler::InputHandler(const pt::ptree &tree)
    : SimulationParameters(tree),
      RingPolymerParameters(tree),
      EnsembleParameters(tree),
      ThermostatParameters(tree),
      PotentialParameters(tree),
      DynamicsParameters(tree) {
  ValidateInput(SimulationParameters.box_length, "box_length");

  ReadInitFile(
      SimulationParameters.init_file, RingPolymerParameters.n_beads,
      RingPolymerParameters.position_from_init,
      RingPolymerParameters.labels_from_init);

  if (SimulationParameters.sim_type == "CMD" ||
      SimulationParameters.sim_type == "cmd" ||
      SimulationParameters.sim_type == "QCMD" ||
      SimulationParameters.sim_type == "qcmd") {
    ValidateInput(RingPolymerParameters.gamma, "gamma (adabaticity parameter)");
  }
}
