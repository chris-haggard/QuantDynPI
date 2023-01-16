## Manual install

- Armadillo linear algebra library (with MKL)
  - [Download latest version](http://arma.sourceforge.net/docs.html)
  - make sure MKL and icc (optimal, gcc works) modules are loaded (or ensure you have BLAS/OpenBLAS and LAPACK)
  - in the armadillo directory run  
  ```cmake . -DCMAKE_INSTALL_PREFIX:PATH=. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc```  
        ```make```  
        ```make install```  

  - if hdf5 is causing issues the change the first command to  
    ```cmake . -D DETECT_HDF5=false -DCMAKE_INSTALL_PREFIX:PATH=. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc```
- Boost libraries
  - ```sudo apt install libboost-all-dev```   
  Or manual install:
  - [Download latest version](https://www.boost.org/)
  - google 'Boost getting started' and follow the simple steps given on the Boost website
- Google test
  - Download googletest from [github repo](https://github.com/google/googletest)
  - copy googletest-master into the project root (no need to compile anything)
- Splinter
  - Download the prebuilt binaries from https://github.com/bgrimstad/splinter
  - CMake is configured to look for this in ~/Software 
- an MPI compiler
- cmake (module load cmake)

#### Running

```shell
    
To build and run the executable

``` shell
mkdir build && cd build  
cmake .. -DMPI_CXX_COMPILER=mpiicpc -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build . --target QuantDynPI --parallel 12
cd ..
mpirun -n <n_proc> bin/QuantDynPI input/test.json  
```

