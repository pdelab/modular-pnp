This repository contains a modularized version of Penn State's CCMA
Newton solver for the Poisson-Nernst-Planck system


TO COMPILE A SPECIFIC PROBLEM:
In order to compile the code, copy the desired main.cpp file to ./main.cpp
from the appropriate directory in ./problems/
and then run ``make`` followed by executing the ``PNP'' file

e.g.
cp ./problems/voltage_benchmark/main.cpp ./main.cpp
make
./PNP




TO CONFIGURE THE LIBRARY:
CMake is used to link the required files and libraries.
To ensure proper linking, create a CMakeLists.txt file in the current directory
and run ``cmake .'' to build the appropriate files.



An example CMakeLists.txt file is below:

## Begin CMakeLists.txt

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME PNP)
project(${PROJECT_NAME})

# Set verbose output while testing CMake
#set(CMAKE_VERBOSE_MAKEFILE 1)

# Set CMake behavior
cmake_policy(SET CMP0004 OLD)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN)

# Need to get VTK config because VTK uses advanced VTK features which
# mean it's not enough to just link to the DOLFIN target. See
# http://www.vtk.org/pipermail/vtk-developers/2013-October/014402.html
find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} NO_MODULE QUIET)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Compiler flags
set(CMAKE_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

# Set FASP directories
set(FASP_DIR /usr/local/faspsolver)     # must be set for each new build
set(FASP4NS_DIR /usr/local/fasp4ns)     # must be set for each new build

# Include directories
set(FASP_INCLUDE_DIR ${FASP_DIR}/base/include)
set(FASP4NS_INCLUDE_DIR ${FASP4NS_DIR}/include)
include_directories(${DOLFIN_INCLUDE_DIRS} ${FASP_INCLUDE_DIR} ${FASP4NS_INCLUDE_DIR} ./include)
include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME} main.cpp ./src/params.cpp)

# Target libraries
set(FASP_LIB ${FASP_DIR}/lib/libfasp.a)
set(FASP4NS_LIB ${FASP4NS_DIR}/lib/libfasp4ns.a)
target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})

## End CMakeLists.txt