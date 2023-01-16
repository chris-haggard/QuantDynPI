add_library(IO_utils IO_utils.cpp)
target_include_directories(IO_utils PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(IO_utils ${ARMADILLO_LIBRARIES})

add_library(unit_conversion unit_conversion.cpp)
add_library(AtomData AtomData.cpp)

add_library(NormalDistribution NormalDistribution.cpp)
target_include_directories(NormalDistribution PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(NormalDistribution Boost::random)
#target_link_libraries(NormalDistribution PUBLIC ${Boost_LIBRARIES})

add_library(rand_utils rand_utils.cpp)
target_include_directories(rand_utils PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(rand_utils Boost::random)
#target_link_libraries(rand_utils PUBLIC ${Boost_LIBRARIES})

add_library(RingPolymerUtils RingPolymerUtils.cpp)
target_include_directories(RingPolymerUtils PUBLIC ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(RingPolymerUtils ${ARMADILLO_LIBRARIES} Boost::random)



