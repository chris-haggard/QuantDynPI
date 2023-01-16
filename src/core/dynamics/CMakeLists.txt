add_subdirectory(qcmd)

add_library(QuasiThermostat QuasiThermostat.cpp)
add_dependencies(QuasiThermostat Input RingPolymer NormalDistribution)
target_include_directories(QuasiThermostat PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(QuasiThermostat Input RingPolymer NormalDistribution)

add_library(Dynamics Dynamics.cpp QCDynamics.cpp ClassicalDynamics.cpp GridQCMD.cpp)
add_dependencies(Dynamics Input Thermostat RingPolymer Potential QCMD QuasiThermostat)
target_include_directories(Dynamics PUBLIC ${ARMADILLO_INCLUDE_DIRS}) 
target_link_libraries(Dynamics Input RingPolymer Thermostat Potential QCMD RingPolymerUtils QuasiThermostat ${ARMADILLO_LIBRARIES})
