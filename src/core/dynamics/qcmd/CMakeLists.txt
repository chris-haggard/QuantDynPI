add_library(QCMD QCMD.cpp TriatomicQCMD.cpp TetraatomicQCMD.cpp)
add_dependencies(QCMD RingPolymer RingPolymerUtils)
target_include_directories(QCMD PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(QCMD RingPolymer RingPolymerUtils ${ARMADILLO_LIBRARIES})
