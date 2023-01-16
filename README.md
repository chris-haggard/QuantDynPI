# QuantDynPI

Path-integral quantum dynamics simulations package. Capable of performing: (Thermostatted) Ring-Polymer Molecular Dynamics, Centroid MD, Quasi-centroid MD.
***

## Background

Nuclear quantum effects, such as zero-point energy and tunnelling, are immportant to account for in many condensed-phase systems. By using the path-integral fomulation of quantum mechanics the quantum Boltzmann operator (the QM operator the governs the thermal statistics of a system) can be discretised and reformulated classically. These classical simulations are carried out in the extended phase space of the ring-polymer, a harmonically connected system of replicas of the original classical system. 

A variety of methods exist in the quantum statistics-classical mechanics space. This package implements the two most popular methods, TRPMD and CMD, and the newer QCMD method. The focus of this software is producing dipole autocorrelation functions, and by extension, infrared spectra. This can be used to probe the dyanamics of system, and whether the chosen methods faithfully replicates the underlying physics.

This package implements the above methods. The only requirement is to specify and input file, examples can be found in the ```input``` directory. The package uses MPI to spawn simulations with different random number seeds. These simultions are then averaged to produce a final TCF with errors.

### Requirements

- CMake  
- Armadillo  
- MKL   
- Boost  
- MPI compiler  
- Spliter (multivariate interpolation, only required for grid based simulations)

### Install & run
```
sudo apt install cmake libarmadillo-dev libboost-all-dev openmpi-bin 
git clone https://github.com/google/googletest.git  
mkdir build  
cd build  
cmake ..  
cmake --build . --target QuantDynPI  
mpirun -n <n_proc> ../bin/QuantDynPI
```   

For a full manual install see [INSTALLING.MD](INSTALLING.md)

available preprocessor directives (requires recompilation)
```shell
SAVE_TRAJECTORY
SKIP_THERMAL
PRINT_STEPS
FROZEN_CENTROID
```

To build and run the tests

```shell
cmake --build . --target QuantDynPI_TEST  
./tests/QuantDynPI_TEST  
```

To debug the program
```shell
mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug -DMPI_CXX_COMPILER=mpiicpc -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc ..
cmake --build . --target QuantDynPI --parallel 12
cd ..
gdb
file bin/QuantDynPI
run input/input_file.json
bt
```

#### Input file
- Gamma must be specified. It has no effect for (T)RPMD/Classical. A gamma of <0.0 will mean CMD/QCMD will not be adaibatically separated (for grid based calculations)

#### Adding a test

All additions must have an associated unit test. These are found in the ```test``` dir which follows the structure of the ```src``` dir.

#### Documentation

Generate docs with ```doxygen Doxyfile```.

#### Common causes of error

- The path to the init file in the input json must be absolute and not include ~ etc.
- Propagation is done entirely in normal modes. The cartesian representations are rarely up to date. The cartesian positions are only guaranteed to be up to date at the end of each step. The momenta are rarely up to date.
- QCMD timestep is dependent on gamma (and therefore the number of beads), and must be quite low
