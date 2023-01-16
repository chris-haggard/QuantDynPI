#pragma once
#include <string>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <fstream>

void ReadInitFile(
    const std::string, const unsigned int, arma::cube &,
    std::vector<std::string> &);
arma::rowvec ReadString(const std::string);

template <typename T>
void WriteFile(const T &v, std::string filename) {
  v.save(filename, arma::raw_ascii);
}

template <typename T>
void PrintVector(const std::vector<T> &v) {
  for (auto i : v) {
    std::cout << i << std::endl;
  }
  std::cout << std::endl;
}

std::string BaseFileNamer(
    const std::string, const std::string, const double, const size_t,
    const size_t, const double, const double);

std::string OutputFileNamer(const std::string, const std::string);

std::string RemoveAfterDecimalPoint(const std::string);

std::vector<std::string> SplitString(std::string);

