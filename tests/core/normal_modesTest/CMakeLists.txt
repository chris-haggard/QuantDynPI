# add library for each test
add_library(
  NormalModesTest
  NormalModesTest.cpp)
# link src library to test library, not quite sure why this is necessary
target_link_libraries(NormalModesTest NormalModes)
