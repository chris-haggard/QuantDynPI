cd build
cmake .. -DMPI_CXX_COMPILER=mpiicpc -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build . --target QuantDynPI --parallel 12
cd ..

