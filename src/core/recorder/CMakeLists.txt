add_library(TCF_Traj TCF.cpp TrajectoryRecorder.cpp)
target_include_directories(TCF_Traj PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(TCF_Traj PUBLIC ${ARMADILLO_LIBRARIES})

#add_library(Recorder Recorder.cpp)
#add_dependencies(Recorder TCF_Traj)
#target_include_directories(Recorder PUBLIC ${ARMADILLO_INCLUDE_DIRS})
#target_link_libraries(Recorder TCF_Traj ${ARMADILLO_LIBRARIES})
