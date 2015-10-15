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

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
    values[1] = 10*exp(-(dx*dx + dy*dy) / 0.02);
    values[2] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
    values[1] = sin(5*x[1]);
    values[2] = sin(5*x[2]);
  }
};

int main()
{

  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  dolfin::Point p0( -10.0, -10.0, -10.0);
  dolfin::Point p1(  10.0,  10.0,  10.0);
  dolfin::BoxMesh mesh(p0, p1, 10, 10, 5);

  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant ux(0.0);
  XBoundaries bdary_x(10.0);
  DirichletBC bc1(V, ux, bdary_x);

  Constant uy(2.0);
  YBoundaries bdary_y(10.0);
  DirichletBC bc2(V, uy, bdary_y);

  Constant uz(4.0);
  ZBoundaries bdary_z(10.0);
  DirichletBC bc3(V, uz, bdary_z);

  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc1);
  bcs.push_back(&bc2);
  bcs.push_back(&bc3);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);

  dolfin::EigenMatrix EA; assemble(EA,a);

  Source f;
  L.f = f;
  dolfin::EigenVector EV; assemble(EV,L);

  // Compute solution
  Function u(V);
  solve(a == L,u, bcs);

  // Save solution in VTK format
  File file("output/poisson.pvd");
  file << u;


  return 0;
}
