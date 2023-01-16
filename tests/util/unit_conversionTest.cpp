#include <gtest/gtest.h>

#include <stdexcept>

#include "../../src/util/unit_conversion.hpp"

TEST(unit_conversion_tests, AllTests) {
  ASSERT_EQ(1.0, T_to_beta(beta_to_T(1.0)));
}
