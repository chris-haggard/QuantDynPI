#include "NormalDistribution.hpp"

/**
 * @brief Constructor for NormalDistribution object. Assigns seed
 * seeds the rng.
 *
 * @param s seed
 */
NormalDistribution::NormalDistribution(double s)
    : seed(s), generator(s), distribution(0.0, 1.0) {
}

/**
 * @brief Get next random number by calling NormalDistribution()
 *
 * @return
 */
double NormalDistribution::operator()() {
  return distribution(generator);
}
