#include <gtest/gtest.h>

#include "../../src/util/RingPolymerUtils.hpp"

class RingPolymerUtilsTest : public ::testing::Test {
 protected:
  arma::cube x, y;
  arma::mat a, b;

  RingPolymerUtilsTest()
      : x(48, 3, 108, arma::fill::zeros),
        y(x),
        a(x.n_rows, x.n_cols, arma::fill ::zeros),
        b(a) {
  }

  ~RingPolymerUtilsTest() {
    x.reset();
    y.reset();
    a.reset();
    b.reset();
  }
};

TEST_F(RingPolymerUtilsTest, CosThetaVecTest) {
  a.col(0).fill(1.0);
  b.col(1).fill(1.0);
  arma::vec temp(a.n_rows);
  temp.fill(arma::datum::pi / 2.0);
  arma::vec cos_theta = arma::acos(CosThetaVec(a, b));
  ASSERT_TRUE(arma::approx_equal(cos_theta, temp, "absdiff", 1e-14));
}

TEST_F(RingPolymerUtilsTest, CosSinThetaTest) {
  a.randn();
  b.randn();
  arma::vec temp(a.n_rows);
  temp.ones();
  ASSERT_TRUE(arma::approx_equal(
      arma::square(CosThetaVec(a, b)) + arma::square(SinThetaVec(a, b)), temp,
      "absdiff", 1e-14));
}

TEST_F(RingPolymerUtilsTest, CubeCrossTest) {
  arma::cube i(1, 3, 108, arma::fill::zeros);
  arma::cube j(i), k(i);
  i.col(0).fill(1.0);
  j.col(1).fill(1.0);
  k.col(2).fill(1.0);
  ASSERT_TRUE(arma::approx_equal(CubeCross(i, j), k, "abs", 1e-14));
  ASSERT_TRUE(arma::approx_equal(CubeCross(j, i), -k, "abs", 1e-14));
  ASSERT_TRUE(arma::approx_equal(CubeCross(i, k), -j, "abs", 1e-14));
}
