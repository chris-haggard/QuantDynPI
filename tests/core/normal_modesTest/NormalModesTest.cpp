#include <gtest/gtest.h>

#include "../../../src/core/normal_modes/NormalModes.hpp"

class NormalModesTest : public ::testing::Test {
 protected:
  arma::cube x, orig_x, y, orig_y, output_buffer;
  NormalModes NMConv;

  NormalModesTest()
      : x(48, 3, 108, arma::fill::randn),
        orig_x(x),
        y(arma::size(x), arma::fill::randn),
        orig_y(y),
        output_buffer(arma::size(x)),
        NMConv(x) {
  }

  ~NormalModesTest() {
    x.reset();
    orig_x.reset();
    y.reset();
    orig_y.reset();
    output_buffer.reset();
  }
};

TEST_F(NormalModesTest, FFT_does_not_alter_first_argument_forward) {
  // cannot pass first argument as const so checking it is not altered in out of
  // place fft
  NMConv.CartToNM(x, output_buffer);
  ASSERT_TRUE(arma::approx_equal(x, orig_x, "absdiff", 1e-14));
  ASSERT_FALSE(arma::approx_equal(output_buffer, orig_x, "absdiff", 1e-14));
}

TEST_F(NormalModesTest, FFT_does_not_alter_first_argument_backward) {
  NMConv.NMToCart(x, output_buffer);
  ASSERT_TRUE(arma::approx_equal(y, orig_y, "absdiff", 1e-14));
  ASSERT_FALSE(arma::approx_equal(x, orig_y, "absdiff", 1e-14));
}

TEST_F(NormalModesTest, FFT_mat_equivalence) {
  arma::cube temp(arma::size(x));
  for (arma::uword k = 0; k < x.n_slices; k++) {
    temp.slice(k) = CartToNM_TEST(x.slice(k));
  }
  temp /= std::sqrt(temp.n_rows);
  NMConv.CartToNM(x, output_buffer);
  for (arma::uword k = 0; k < x.n_slices; k++) {
    output_buffer.slice(k) = arma::sort(output_buffer.slice(k));
    temp.slice(k) = arma::sort(temp.slice(k));
    ASSERT_TRUE(arma::approx_equal(
        output_buffer.slice(k), temp.slice(k), "absdiff", 1e-12));
  }
}

TEST_F(NormalModesTest, FFT_centroid_test) {
  arma::cube centroid = arma::mean(x, 0);
  NMConv.CartToNM(x, output_buffer);
  ASSERT_TRUE(
      arma::approx_equal(output_buffer.row(0), centroid, "absdiff", 1e-14));
}

TEST_F(NormalModesTest, Mat_centroid_test) {
  arma::cube temp(arma::size(x));
  arma::cube centroid = arma::mean(x, 0);
  for (arma::uword k = 0; k < x.n_slices; k++) {
    temp.slice(k) = CartToNM_TEST(x.slice(k));
  }
  ASSERT_TRUE(arma::approx_equal(
      (1.0 / std::sqrt(temp.n_rows)) * temp.row(0), centroid, "absdiff",
      1e-14));
}

TEST_F(NormalModesTest, FFT_forward_backward_equivalence) {
  NMConv.CartToNM(x, y);
  ASSERT_TRUE(arma::approx_equal(x, orig_x, "absdiff", 1e-14));
  ASSERT_FALSE(arma::approx_equal(y, orig_x, "absdiff", 1e-14));
  ASSERT_FALSE(arma::approx_equal(y, x, "absdiff", 1e-14));
  NMConv.NMToCart(y, output_buffer);
  ASSERT_TRUE(arma::approx_equal(x, output_buffer, "absdiff", 1e-14));
}

class NormalModesClassicalTest : public ::testing::Test {
 protected:
  arma::cube x, orig_x, y, orig_y, output_buffer;
  NormalModes NMConv;

  NormalModesClassicalTest()
      : x(1, 3, 108, arma::fill::randn),
        orig_x(x),
        y(arma::size(x), arma::fill::randn),
        orig_y(y),
        output_buffer(arma::size(x)),
        NMConv(x) {
  }

  ~NormalModesClassicalTest() {
    x.reset();
    orig_x.reset();
    y.reset();
    orig_y.reset();
    output_buffer.reset();
  }
};

TEST_F(NormalModesClassicalTest, FFT_classical_fills_out_buffer_with_input) {
  NMConv.CartToNM(x, output_buffer);
  ASSERT_TRUE(arma::approx_equal(x, orig_x, "absdiff", 1e-14));
  ASSERT_TRUE(arma::approx_equal(output_buffer, x, "absdiff", 1e-14));
}

TEST_F(
    NormalModesClassicalTest, Back_FFT_classical_fills_out_buffer_with_input) {
  NMConv.NMToCart(y, output_buffer);
  ASSERT_TRUE(arma::approx_equal(y, orig_y, "absdiff", 1e-14));
  ASSERT_TRUE(arma::approx_equal(output_buffer, y, "absdiff", 1e-14));
}

