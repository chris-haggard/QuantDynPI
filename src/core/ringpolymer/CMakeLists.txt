add_library(RingPolymer RingPolymer.cpp AdiabaticRP.cpp QuasiCentroidRP.cpp)
add_dependencies(RingPolymer Input Ensemble NormalModes AtomData RingPolymerUtils)
target_include_directories(RingPolymer PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(RingPolymer Input Ensemble NormalModes AtomData RingPolymerUtils ${ARMADILLO_LIBRARIES})
