add_library(AmmoniaDipoleFortran Ammonia_Dipole_AQZfc.f90)

add_library(PartridgeSchwenkeFortran PartridgeSchwenke.f90 PartridgeSchwenkeDMS.f90)

add_library(AmmoniaPES AmmoniaPES.cpp Ammonia_PES_Constants.hpp Ammonia_Dipole_cpp_wrapper.hpp)
add_dependencies(AmmoniaPES Input AmmoniaDipoleFortran)
target_include_directories(AmmoniaPES PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(AmmoniaPES Input AmmoniaDipoleFortran ${ARMADILLO_LIBRARIES})

add_library(PartridgeSchwenkePES PartridgeSchwenke.cpp PartridgeSchwenkeWrappers.hpp)
add_dependencies(PartridgeSchwenkePES Input PartridgeSchwenkeFortran)
target_include_directories(PartridgeSchwenkePES PUBLIC ${ARMADILLO_INCLUDE_DIRS}  )
target_link_libraries(PartridgeSchwenkePES Input PartridgeSchwenkeFortran ${ARMADILLO_LIBRARIES})

add_library(SplinePot SplinePot.cpp)
add_dependencies(SplinePot PartridgeSchwenkePES RingPolymerUtils IO_utils)
target_include_directories(SplinePot PUBLIC ${SPLINTER_INCLUDE_DIRS})
target_link_libraries(SplinePot PartridgeSchwenkePES RingPolymerUtils IO_utils ${SPLINTER_LIBRARIES})


add_library(Potential Potential.cpp GasQTIP4P.cpp SilveraGoldman.cpp Harmonic.cpp)
add_dependencies(Potential Input AmmoniaPES PartridgeSchwenkePES SplinePot RingPolymerUtils)
target_include_directories(Potential PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(Potential Input AmmoniaPES PartridgeSchwenkePES SplinePot RingPolymerUtils ${ARMADILLO_LIBRARIES})
