#include "rand_utils.hpp"

#include <algorithm>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cassert>
#include <stdexcept>

/**
 * @brief Determine if elements in vector are unique
 *
 * @param v
 * @return true if elements are unique
 * @return false otherwise
 */
bool VectorIsUnique(std::vector<double> v) {
  if (v.empty()) {
    throw std::range_error("vector is empty");
  }
  std::sort(v.begin(), v.end());
  return std::adjacent_find(v.begin(), v.end()) == v.end();
}

/**
 * @brief Generate a vector with random doubles used as seeds for rng for each
 * trajectory (MPI proc)
 *
 * @param seed Seed used to seed rng used to gen other seeds
 * @param n_proc number of MPI porcesses
 * @return std::vector<double>
 */
std::vector<double> SeedVector(double seed, size_t n_proc) {
  std::vector<double> v(n_proc);
  boost::random::mt19937_64 generator(seed);
  boost::random::uniform_real_distribution<double> distribution(0.0, 1e8);
  // fill vector then check that it is unique
  do {
    for (size_t i = 0; i < n_proc; i++) {
      v.at(i) = distribution(generator);
    }
  } while (!VectorIsUnique(v));
  assert(v.size() == n_proc);
  assert(VectorIsUnique(v));
  return v;
}
