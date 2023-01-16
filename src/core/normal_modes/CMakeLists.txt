# find MKL
if(DEFINED ENV{MKLROOT})
  set(MKLROOT $ENV{MKLROOT})
else()
  message( FATAL_ERROR "MKLROOT environment varialbe not set. You might need to load the mkl module." )
endif()
message("MKLROOT = ${MKLROOT}")

set(MKL_LIBRARIES "${MKLROOT}/lib/intel64")
set(MKL_INCLUDE_DIRS "${MKLROOT}/include")
message("MKL_LIBRARIES = ${MKL_LIBRARIES}")

add_library(NormalModes NormalModes.cpp)
target_include_directories(NormalModes PUBLIC ${MKL_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(NormalModes Input ${ARMADILLO_LIBRARIES})
target_link_libraries(NormalModes -lmkl_rt -lpthread -lm -ldl)
target_link_libraries(NormalModes -L${MKL_LIBRARIES})
