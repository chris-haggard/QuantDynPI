cmake_minimum_required(VERSION 3.16)

project(QuantDynPI LANGUAGES CXX)
enable_language(Fortran) # <-- required to link fortran, must be here, no idea why

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_Fortran_FLAGS_RELEASE "-O3")

# parallel build on 12 cores
# set(CMAKE_BUILD_PARALLEL_LEVEL 12) <--- has no effect?
# set some variables with location of required libraries
set(ARMADILLO_DIR ~/Software/armadillo-11.2.3)
set(BOOST_DIR ~/Software/boost_1_77_0)
set(SPLINTER_DIR ~/Software/splinter)

# set a variable that can be changed on command line with -D
set(GENERAL CACHE BOOL ON)

# if general install then find required packages, if non-general install find
# manually installed libraries and specify include directories

if(GENERAL)
  message("General install")
  find_package(Armadillo REQUIRED)
  find_package(Boost REQUIRED)
  find_package(MPI REQUIRED)
  find_package(OpenMP REQUIRED)
  message("Armadillo version ${ARMADILLO_VERSION_STRING} found")
  message("Boost version ${Boost_VERSION_STRING} found")
  message("MPI version ${MPI_VERSION} found")
endif(GENERAL)

if(NOT GENERAL)
  message("Non-general install")
  # find boost in manually installed place
  set(BOOST_ROOT "~/Software" CACHE PATH "Boost library path")
  # tell cmake not to look for other boost
  set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL "Do not search system for Boost")
  find_package(Boost 1.77.0 EXACT REQUIRED COMPONENTS random)
  message("Boost found = ${Boost_FOUND}")
  message("Boost include location  = ${Boost_INCLUDE_DIRS}")
  message("Boost library location  = ${Boost_LIBRARY_DIRS}")
  message("Boost libraries = ${Boost_LIBRARIES}")
  # variable set to same name returned by the general install above
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_DIR}/include)
  # tells cmake where to look for mpi (works for odyssey). This is a bit dodgy:
  # somtimes work other times not, check .rsyncignore
  set(CMAKE_PREFIX_PATH share/apps/intel)
  message(${CMAKE_PREFIX_PATH})
  find_package(MPI REQUIRED)

  if(MPI_FOUND)
    message("MPI_FOUND = ${MPI_FOUND}")
    message("MPI library location: ${MPI_LIBRARIES}")
    include_directories(${MPI_INCLUDE_PATH})
  endif(MPI_FOUND)

  find_package(OpenMP REQUIRED)

  # this finds the armadillo library, seems to be the only way to link currently
  find_library(
    ARMADILLO_LIBRARIES
    NAMES libarmadillo.so.11.2.3
    HINTS ${ARMADILLO_DIR}/lib/ REQUIRED)

  # check that it is correct
  if(ARMADILLO_LIBRARIES)
    message("Armadillo lib location: ${ARMADILLO_LIBRARIES}")
  endif(ARMADILLO_LIBRARIES)

  set(SPLINTER_INCLUDE_DIRS ${SPLINTER_DIR}/include/SPLINTER)
  set(SPLINTER_LIBRARY_DIRS ${SPLINTER_DIR}/lib)
  #set(SPLINTER_LIBRARIES ${SPLINTER_DIR}/x86-64/libsplinter-3-0.so)
  find_library(
    SPLINTER_LIBRARIES
    NAMES libsplinter-3-0.so
    HINTS ${SPLINTER_LIBRARY_DIRS} REQUIRED)
  if(SPLINTER_LIBRARIES)
    message("Splinter lib location ${SPLINTER_LIBRARIES}")
  endif(SPLINTER_LIBRARIES)


endif(NOT GENERAL)

include_directories(${MPI_INCLUDE_PATH})

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

# allow testing
enable_testing()

# add googletest
add_subdirectory(${CMAKE_SOURCE_DIR}/googletest)
include_directories(${CMAKE_SOURCE_DIR}/googletest/googletest/include)
# add test subdir
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
