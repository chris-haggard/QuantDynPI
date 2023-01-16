#pragma once
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <string>

class TrajectoryRecorder {
 public:
  std::string trajectory_file;
  void AppendFrame(
      const arma::cube &, const std::vector<std::string> &, const double);
};
