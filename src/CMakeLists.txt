# compile options for all source files in this dir and below
add_compile_options(-Wall -Wextra -pedantic -O3 -std=c++17)
add_subdirectory(core)
add_subdirectory(util)
