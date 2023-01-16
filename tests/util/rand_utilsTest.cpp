#include <gtest/gtest.h>

#include <stdexcept>

#include "../../src/util/rand_utils.hpp"

TEST(rand_utils_tests, VectorIsUniqueTest) {
  std::vector<double> v{1.0, 2.0};
  ASSERT_TRUE(VectorIsUnique(v));
  v = {1.0, 1.0, 2.0, 0.0, 9.9};
  ASSERT_FALSE(VectorIsUnique(v));
  v = {};
  ASSERT_THROW(VectorIsUnique(v), std::range_error);
  v = {0.0, 1.0e-9, -1.0e-9};
  ASSERT_TRUE(VectorIsUnique(v));
}
