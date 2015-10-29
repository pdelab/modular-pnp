/*! \file test_faspfenics.cpp
 *
 *  \brief Main to test FASP/FENICS interface using the Poisson problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include <vector>
#include "boundary_conditions.h"
#include "Poisson.h"

using namespace dolfin;
//using namespace std;

int main()
{

  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  dolfin::Point p0( -10.0, -10.0, -10.0);
  dolfin::Point p1(  10.0,  10.0,  10.0);
  dolfin::BoxMesh mesh(p0, p1, 10, 10, 5);

  Poisson::FunctionSpace V(mesh);


  // ##############################################################################
  // ### First test to see if the boundary conditions works
  // ##############################################################################
  // dolfin::MeshFunction boundary_parts;
  dolfin::MeshFunction<size_t> boundary_parts(mesh, mesh.topology().dim()-1);
  boundary_parts.set_all(0);


  XBoundaries bdary_x(-10.0);
  bdary_x.mark(boundary_parts, 1);
  XBoundaries bdary_x2(10.0);
  bdary_x.mark(boundary_parts, 2);

  YBoundaries bdary_y(-10.0);
  bdary_y.mark(boundary_parts, 3);
  YBoundaries bdary_y2(10.0);
  bdary_y2.mark(boundary_parts, 4);

  ZBoundaries bdary_z(-10.0);
  bdary_z.mark(boundary_parts, 5);
  ZBoundaries bdary_z2(10.0);
  bdary_z2.mark(boundary_parts, 6);

  File file("aux/test_bc/output/boundary_parts.pvd");
  file << boundary_parts;
  // ##############################################################################
  // ##############################################################################


  return 0;
}
