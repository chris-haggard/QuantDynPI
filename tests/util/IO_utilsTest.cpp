#include <gtest/gtest.h>

#include "../../src/util/IO_utils.hpp"

TEST(IO_utils_tests, ReadStringTest) {
  arma::rowvec temp({1.1, 2.2, 3.3});
  ASSERT_TRUE(
      arma::approx_equal(ReadString(" 1.1 2.2 3.3"), temp, "absdiff", 1e-12));
  ASSERT_TRUE(
      arma::approx_equal(ReadString("1.1 2.2 3.3 "), temp, "absdiff", 1e-12));
  ASSERT_TRUE(
      arma::approx_equal(ReadString("1.1 2.2  3.3"), temp, "absdiff", 1e-12));
  ASSERT_FALSE(
      arma::approx_equal(ReadString("9.9 2.2  3.3"), temp, "absdiff", 1e-12));
}

TEST(IO_utils_tests, ReadInitFileTest) {
  // set up correct position cube and label vector
  int n_beads = 2;
  int n_molec = 5;
  arma::cube position_correct(n_beads, 3, n_molec, arma::fill::zeros);
  position_correct.slice(0).col(0).fill(1.0);
  position_correct.slice(0).col(1).fill(2.0);
  position_correct.slice(0).col(2).fill(3.0);
  position_correct.slice(1).col(0).fill(4.0);
  position_correct.slice(1).col(1).fill(5.0);
  position_correct.slice(1).col(2).fill(6.0);
  position_correct.slice(2).col(0).fill(7.0);
  position_correct.slice(2).col(1).fill(8.0);
  position_correct.slice(2).col(2).fill(9.0);
  position_correct.slice(3).col(0).fill(1.0);
  position_correct.slice(3).col(1).fill(2.0);
  position_correct.slice(3).col(2).fill(3.0);
  position_correct.slice(4).col(0).fill(1e-12);
  position_correct.slice(4).col(1).fill(2.0e-13);
  position_correct.slice(4).col(2).fill(99e8);

  std::vector<std::string> labels_correct = {"H", "H", "H", "H", "H"};

  arma::cube position(n_beads, 3, n_molec, arma::fill::zeros);
  std::vector<std::string> labels(n_molec);

  ReadInitFile(
      "initTest.xyz", n_beads, position,
      labels);

  ASSERT_TRUE(arma::approx_equal(position_correct, position, "absdiff", 1e-12));
  ASSERT_EQ(labels_correct.size(), labels.size());
  for (size_t i = 0; i < labels_correct.size(); i++) {
    ASSERT_TRUE(labels_correct.at(i) == labels.at(i));
  }
}
