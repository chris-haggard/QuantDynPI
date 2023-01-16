#add_library(
  # IO_utilsTest IO_utilsTest.cpp ${CMAKE_SOURCE_DIR}/src/util/IO_utils.cpp
  #            ${CMAKE_SOURCE_DIR}/src/util/IO_utils.hpp)
             #target_link_libraries(IO_utilsTest IO_utils)


add_library(UtilTests IO_utilsTest.cpp rand_utilsTest.cpp RingPolymerUtilsTest.cpp unit_conversionTest.cpp)
target_link_libraries(UtilTests IO_utils rand_utils RingPolymerUtils unit_conversion)
