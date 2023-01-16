add_executable(${PROJECT_NAME} main.cpp)
set(GRID_PROJECT_NAME ${PROJECT_NAME}_GRID)
add_executable(${GRID_PROJECT_NAME} main_grid.cpp)
# executable in bin directory
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(${GRID_PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


# preprocessor
option(SAVE_TRAJECTORY "Save trajectory on rank 0 to file" OFF)
option(PRINT_STEPS "Print steps" OFF)
option(SKIP_THERMAL "Skip thermalization" OFF)
option(VIRIAL_ESTIMATOR "Output Virial estimator to file" OFF)
option(FROZEN_CENTROID "Quasicentroids are frozen" OFF)

if(SAVE_TRAJECTORY)
  add_definitions(-DSAVE_TRAJECTORY)
  message("Trajectory of dynamical variable and replica will be saved to file.")
endif()
if(PRINT_STEPS)
  add_definitions(-DPRINT_STEPS)
  message("Printing steps")
endif()
if(SKIP_THERMAL)
  add_definitions(-DSKIP_THERMAL)
  message("Skipping thermalization")
endif()
if(VIRIAL_ESTIMATOR)
  add_definitions(-DVIRIAL_ESTIMATOR)
  message("Saving Virial estimator")
endif()
if(FROZEN_CENTROID)
  add_definitions(-DFROZEN_CENTROID)
  message("Quaiscentroids are frozen")
endif()


add_subdirectory(ringpolymer)
add_subdirectory(thermostat)
add_subdirectory(potentials)
add_subdirectory(normal_modes)
add_subdirectory(recorder)
add_subdirectory(dynamics)

add_library(Input Input.cpp)
add_library(Ensemble Ensemble.cpp)
add_library(Simulation Simulation.cpp)

# tell cmake about library dependecies
add_dependencies(Input IO_utils unit_conversion)
add_dependencies(Ensemble Input unit_conversion)
add_dependencies(Simulation RingPolymer Dynamics Input)

# Boost does not need to be linked as property_tree is header only
target_include_directories(Input PUBLIC ${Boost_INCLUDE_DIRS}
                                        ${ARMADILLO_INCLUDE_DIRS})
target_include_directories(Simulation PUBLIC ${ARMADILLO_INCLUDE_DIRS})

target_link_libraries(Input IO_utils unit_conversion ${ARMADILLO_LIBRARIES})
target_link_libraries(Ensemble Input unit_conversion)
target_link_libraries(Simulation RingPolymer Dynamics Input ${ARMADILLO_LIBRARIES})

target_link_libraries(
  ${PROJECT_NAME}
  Input
  IO_utils
  rand_utils
  Ensemble
  Simulation
  TCF_Traj
  RingPolymerUtils
  ${MPI_LIBRARIES})

# main_grid.cpp requires spline library

target_include_directories(${GRID_PROJECT_NAME} PUBLIC ${SPLINTER_INCLUDE_DIRS})

target_link_libraries(
  ${GRID_PROJECT_NAME}
  Input
  IO_utils
  rand_utils
  Ensemble
  Simulation
  TCF_Traj
  RingPolymerUtils
  ${MPI_LIBRARIES}
  ${SPLINTER_LIBRARIES})
