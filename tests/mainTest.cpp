#include <gtest/gtest.h>

#include "core/normal_modesTest/NormalModesTest.cpp"
#include "core/potentialsTest/HarmonicTest.cpp"
#include "core/potentialsTest/PotentialTest.cpp"
#include "core/potentialsTest/SilveraGoldmanTest.cpp"
#include "core/ringpolymerTest/RingPolymerTest.cpp"
#include "util/IO_utilsTest.cpp"
#include "util/RingPolymerUtilsTest.cpp"
#include "util/rand_utilsTest.cpp"
#include "util/unit_conversionTest.cpp"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
