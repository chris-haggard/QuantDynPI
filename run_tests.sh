#!/bin/bash

cd build
cmake --build . --target QuantDynPI_TEST --parallel 12
./tests/QuantDynPI_TEST

