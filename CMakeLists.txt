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

# BOOST lib
find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

# Set FASP directories and libraries
set(FASP_DIR /usr/local/faspsolver)     # must be set for each new build
set(FASP4NS_DIR /usr/local/fasp4ns)     # must be set for each new build
set(FASP_LIB ${FASP_DIR}/lib/libfasp.a)
set(FASP4NS_LIB ${FASP4NS_DIR}/lib/libfasp4ns.a)

# Local src files and include
set(SRC_DIR ./src/fasp_to_fenics.cpp ./src/params.cpp ./src/domains.cpp ./src/boundary_conditions.cpp ./src/funcspace_to_vecspace.cpp ./src/newton_functs.cpp)

# Include directories
set(FASP_INCLUDE_DIR ${FASP_DIR}/base/include)
set(FASP4NS_INCLUDE_DIR ${FASP4NS_DIR}/include)
include_directories(${DOLFIN_INCLUDE_DIRS} ${FASP_INCLUDE_DIR} ${FASP4NS_INCLUDE_DIR} ./include)
include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Awesome OSX TARGET
set(OSX_TARGET "/opt/local/lib/gcc49/gcc/x86_64-apple-darwin14/4.9.3/libgcc.a" "/opt/local/lib/gcc49/libquadmath.a" "/opt/local/lib/gcc49/libgfortran.a")

# Executables
add_executable(aux_test_reading ./aux/test_reading/aux_test_reading.cpp ${SRC_DIR})
add_executable(aux_test_faspfenics ./aux/test_faspfenics/aux_test_faspfenics.cpp ${SRC_DIR})
add_executable(aux_test_eafe ./aux/test_eafe/aux_test_eafe.cpp ${SRC_DIR})
add_executable(aux_test_eafe2 ./aux/test_eafe2/aux_test_eafe2.cpp ${SRC_DIR})
add_executable(aux_test_bc ./aux/test_bc/aux_test_bc.cpp ${SRC_DIR})
add_executable(aux_test_dof ./aux/test_dof/aux_test_dof.cpp ${SRC_DIR})

add_executable(test_eafe ./tests/eafe_tests/test_eafe.cpp ${SRC_DIR})
add_executable(test_faspfenics ./tests/faspfenics_tests/test_faspfenics.cpp ${SRC_DIR})
add_executable(test_faspsolver ./tests/faspsolver_tests/test_faspsolver.cpp ${SRC_DIR})
add_executable(test_bc ./tests/bc_tests/test_bc.cpp ${SRC_DIR})
add_executable(test_add_matrix ./tests/matrices_tests/test_add_matrix.cpp ${SRC_DIR})
add_executable(test_replace_matrix ./tests/matrices_tests/test_replace_matrix.cpp ${SRC_DIR})
add_executable(test_replace_matrix2 ./tests/matrices_tests/test_replace_matrix2.cpp ${SRC_DIR})
add_executable(test_lin_pnp ./tests/linearized_pnp_tests/test_lin_pnp.cpp ${SRC_DIR})
add_executable(test_lin_pnp_eafe ./tests/linearized_pnp_tests/test_lin_pnp_eafe.cpp ${SRC_DIR})
add_executable(test_pnp_eafe ./tests/pnp_tests/test_pnp_eafe.cpp ${SRC_DIR})
add_executable(test_newton_param ./tests/newton_param_tests/test_newton_param.cpp ${SRC_DIR})

add_executable(lin_pnp_eafe ./benchmarks/linear_PNP/lin_pnp_EAFE.cpp ${SRC_DIR})
add_executable(lin_pnp ./benchmarks/linear_PNP/lin_pnp.cpp ${SRC_DIR})
add_executable(pnp_eafe ./benchmarks/PNP/pnp_EAFE.cpp ${SRC_DIR})
add_executable(pnp_eafe2 ./benchmarks/PNP/pnp_EAFE2.cpp ${SRC_DIR})
add_executable(pnp ./benchmarks/PNP/pnp.cpp ${SRC_DIR})
add_executable(pnp_adapt ./benchmarks/PNP/pnp_adaptive.cpp ${SRC_DIR})
add_executable(newton_poisson ./benchmarks/Newton_Poisson/newton_poisson.cpp ${SRC_DIR})
add_executable(pnp_time ./benchmarks/PNP_time/pnp_time.cpp ${SRC_DIR})
add_executable(battery ./benchmarks/battery/battery.cpp ${SRC_DIR})
add_executable(battery_per ./benchmarks/battery/battery_PER.cpp ${SRC_DIR})

# Setup targets
## AUX TESTs
target_link_libraries(aux_test_reading ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(aux_test_faspfenics ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(aux_test_bc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(aux_test_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(aux_test_eafe2 ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(aux_test_dof ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})

## UNIT TESTS
target_link_libraries(test_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_faspfenics ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_faspsolver ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_bc ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_add_matrix ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_replace_matrix ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_replace_matrix2 ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_lin_pnp ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_lin_pnp_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_pnp_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})
target_link_libraries(test_newton_param ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET})

# # ## BENCHMARKS
target_link_libraries(lin_pnp ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(lin_pnp_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(pnp ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(pnp_eafe ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(pnp_eafe2 ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(pnp_adapt ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(newton_poisson ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(pnp_time ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(battery ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
target_link_libraries(battery_per ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${FASP_LIB} ${FASP4NS_LIB} ${OSX_TARGET} ${Boost_LIBRARIES})
