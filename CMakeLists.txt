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

# Set FASP directories and libraries
set(FASP_DIR /usr/local/faspsolver)     # must be set for each new build
set(FASP4NS_DIR /usr/local/fasp4ns)     # must be set for each new build
set(FASP_LIB ${FASP_DIR}/lib/libfasp.a)
set(FASP4NS_LIB ${FASP4NS_DIR}/lib/libfasp4ns.a)

# Local src files and include
set(SRC_DIR ./src/fasp_to_fenics.cpp ./src/params.cpp ./src/domains.cpp ./src/boundary_conditions.cpp)

# Include directories
set(FASP_INCLUDE_DIR ${FASP_DIR}/base/include)
set(FASP4NS_INCLUDE_DIR ${FASP4NS_DIR}/include)
include_directories(${DOLFIN_INCLUDE_DIRS} ${FASP_INCLUDE_DIR} ${FASP4NS_INCLUDE_DIR} ./include)
include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Executables
add_executable(aux_test_reading ./aux/test_reading/aux_test_reading.cpp ${SRC_DIR})
add_executable(aux_test_faspfenics ./aux/test_faspfenics/aux_test_faspfenics.cpp ${SRC_DIR})
add_executable(aux_test_eafe ./aux/test_eafe/aux_test_eafe.cpp ${SRC_DIR})
add_executable(aux_test_eafe2 ./aux/test_eafe2/aux_test_eafe2.cpp ${SRC_DIR})
add_executable(aux_test_bc ./aux/test_bc/aux_test_bc.cpp ${SRC_DIR})

add_executable(test_eafe ./tests/eafe_tests/test_eafe.cpp ${SRC_DIR})
add_executable(test_faspfenics ./tests/faspfenics_tests/test_faspfenics.cpp ${SRC_DIR})
add_executable(test_bc ./tests/bc_tests/test_bc.cpp ${SRC_DIR})

# Setup targets
target_link_libraries(aux_test_reading ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
target_link_libraries(aux_test_faspfenics ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
target_link_libraries(aux_test_bc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
#### REGULAR TARGET
target_link_libraries(aux_test_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
target_link_libraries(aux_test_eafe2 ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
target_link_libraries(test_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
target_link_libraries(test_faspfenics ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})
target_link_libraries(test_bc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB})

#### WEIRD OSX TARGET
# target_link_libraries(aux_test_bc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(aux_test_reading ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(aux_test_faspfenics ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(aux_test_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(aux_test_eafe2 ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(test_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(test_faspfenics ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
# target_link_libraries(test_bc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")
