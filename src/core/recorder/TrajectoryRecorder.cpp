#include "TrajectoryRecorder.hpp"

#include <fstream>
#include <ios>
#include <string>

void TrajectoryRecorder::AppendFrame(
    const arma::cube &position, const std::vector<std::string> &labels,
    const double t) {
  std::fstream outfile(trajectory_file, std::ios::app | std::ios::out);
  outfile << std::to_string(position.n_slices * position.n_rows) << "\n";
  outfile << "# t = " << std::to_string(t) << "\n";
  for (arma::uword k = 0; k < position.n_slices; k++) {
    for (arma::uword i = 0; i < position.n_rows; i++) {
      outfile << labels.at(k) << " " << std::to_string(position(i, 0, k)) << " "
              << std::to_string(position(i, 1, k)) << " "
              << std::to_string(position(i, 2, k)) << std::endl;
    }
  }
  outfile.close();
}
