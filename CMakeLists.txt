cmake_minimum_required(VERSION 3.10)
project(MatrixLab3Project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(main main.cpp "Matrix.h")

target_link_libraries(main PRIVATE MPI::MPI_CXX)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	target_compile_options(main PRIVATE /W4 /WX)
else()
	target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()