add_library(Thermostat Thermostat.cpp PILE_L.cpp PILE_G.cpp PILE.cpp)
add_dependencies(Thermostat Input NormalDistribution RingPolymer)
target_include_directories(Thermostat PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(Thermostat RingPolymer NormalDistribution ${ARMADILLO_LIBRARIES})
# OpenMP::OpenMP_CXX
