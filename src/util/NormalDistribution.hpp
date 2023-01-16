#pragma once
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

class NormalDistribution {
 private:
  const double seed;
  boost::random::mt19937_64 generator;
  boost::random::normal_distribution<double> distribution;

 public:
  NormalDistribution(double);
  double operator()();
};
