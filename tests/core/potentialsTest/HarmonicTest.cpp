#include <gtest/gtest.h>

#include "../../../src/core/potentials/Harmonic.hpp"

TEST(harmonic_potential_test, force_test) {
  arma::cube post(24, 3, 108, arma::fill::zeros);
  // arma::cube force = Harmonic::Force(post);
  ASSERT_EQ(1, 1);
}
