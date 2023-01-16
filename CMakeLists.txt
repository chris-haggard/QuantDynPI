cmake_minimum_required(VERSION 3.16)

project(QuantDynPI LANGUAGES CXX)
enable_language(Fortran) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_Fortran_FLAGS_RELEASE "-O3")

set(ARMADILLO_DIR ~/Software/armadillo-11.2.3)
set(BOOST_DIR ~/Software/boost_1_77_0)
set(SPLINTER_DIR ~/Software/splinter)

set(GENERAL CACHE BOOL ON)

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
  set(BOOST_ROOT "~/Software" CACHE PATH "Boost library path")
  set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL "Do not search system for Boost")
  find_package(Boost 1.77.0 EXACT REQUIRED COMPONENTS random)
  message("Boost found = ${Boost_FOUND}")
  message("Boost include location  = ${Boost_INCLUDE_DIRS}")
  message("Boost library location  = ${Boost_LIBRARY_DIRS}")
  message("Boost libraries = ${Boost_LIBRARIES}")
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_DIR}/include)
  set(CMAKE_PREFIX_PATH share/apps/intel)
  message(${CMAKE_PREFIX_PATH})
  find_package(MPI REQUIRED)

  if(MPI_FOUND)
    message("MPI_FOUND = ${MPI_FOUND}")
    message("MPI library location: ${MPI_LIBRARIES}")
    include_directories(${MPI_INCLUDE_PATH})
  endif(MPI_FOUND)

  find_package(OpenMP REQUIRED)

  find_library(
    ARMADILLO_LIBRARIES
    NAMES libarmadillo.so.11.2.3
    HINTS ${ARMADILLO_DIR}/lib/ REQUIRED)

  if(ARMADILLO_LIBRARIES)
    message("Armadillo lib location: ${ARMADILLO_LIBRARIES}")
  endif(ARMADILLO_LIBRARIES)

  set(SPLINTER_INCLUDE_DIRS ${SPLINTER_DIR}/include/SPLINTER)
  set(SPLINTER_LIBRARY_DIRS ${SPLINTER_DIR}/lib)
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

enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/googletest)
include_directories(${CMAKE_SOURCE_DIR}/googletest/googletest/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
