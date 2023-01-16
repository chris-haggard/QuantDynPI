add_library(
  PotentialTest
  PotentialTest.cpp HarmonicTest.cpp SilveraGoldmanTest.cpp)
target_link_libraries(PotentialTest Potential)
