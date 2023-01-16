#include "IO_utils.hpp"

#include <cassert>
#include <cctype>
#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>

#include "unit_conversion.hpp"

/*__*
 * @brief Read in space delimited string to a row vector
 *
 * Uses rowvec string constructor found in armadillo docs.
 *
 * @param s
 * @return arma::rowvec
 */
arma::rowvec ReadString(const std::string s) {
  return arma::rowvec(s);
}

/**
 * @brief Read init.xyz into a position and label cube.
 *
 * init file xyz form:
 *
 * n_molec*n_atom*n_beads
 *
 * H 1 2 3
 * H 3 4 5
 *
 *
 * @param filename
 * @param n_beads
 * @param position
 * @param labels
 */
void ReadInitFile(
    const std::string filename, const unsigned int n_beads,
    arma::cube &position, std::vector<std::string> &labels) {
  std::fstream init_file;
  init_file.open(filename, std::ios::in);
  assert(init_file.is_open());
  if (init_file.is_open()) {
    std::string line;
    // get the total number line
    getline(init_file, line);
    size_t n_tot = std::stoul(line);
    assert(n_tot / n_beads == position.n_slices);
    // get the comment line
    getline(init_file, line);
    size_t molec_counter = 0;
    size_t row_counter = 0;
    while (getline(init_file, line)) {
      // split label and coordinates
      std::string label = line.substr(0, line.find(" "));
      std::string value = line.substr(line.find(" "), std::string::npos);
      labels.at(molec_counter) = label;
      position.slice(molec_counter).row(row_counter % n_beads) =
          ReadString(value);
      row_counter++;
      // increment only when all beads are read
      if (row_counter % n_beads == 0) {
        molec_counter++;
      }
    }
    assert(
        molec_counter == position.n_slices &&
        "Number of atoms in init file not equal to number of molecules");
  }
  init_file.close();
}

std::string BaseFileNamer(
    const std::string sim_type, const std::string pot_type, const double T,
    const size_t n_molec, const size_t n_bead, const double s,
    const double gamma) {
  std::string out = sim_type + "_" + pot_type + "_" +
                    RemoveAfterDecimalPoint(std::to_string(std::round(T))) +
                    "K_" + std::to_string(n_molec) + "mol_" +
                    std::to_string(n_bead) + "N_" +
                    RemoveAfterDecimalPoint(std::to_string(s)) + "S";
  if (sim_type == "QCMD" || sim_type == "CMD") {
    out +=
        "_" + RemoveAfterDecimalPoint(std::to_string(std::round(gamma))) + "G";
  }
  return out;
}

std::string OutputFileNamer(const std::string base, const std::string id) {
  std::string extension =
      id.find("traj") != std::string::npos ? ".xyz" : ".txt";
  return base + "_" + id + extension;
}

/**
 * @brief Hack to get around std::to_string adding zeros after decimal
 * point. Used to name the output file.
 *
 * @param s
 *
 * @return
 */
std::string RemoveAfterDecimalPoint(std::string s) {
  return s.substr(0, s.find(".", 0));
}

std::vector<std::string> SplitString(std::string s) {
  std::stringstream ss(s);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  std::vector<std::string> out(begin, end);
  return out;
}
