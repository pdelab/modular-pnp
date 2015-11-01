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

  /**
   * First test to see if the boundary conditions works
   */
  // initialize mesh
  printf("Test subdomains: \n");
  printf("\tInitializing mesh: \n");
  double lower_x = -1.0e+1, upper_x =  1.0e+1;
  double lower_y = -1.0e+1, upper_y =  1.0e+1;
  double lower_z = -5.0e+0, upper_z =  5.0e+0;
  printf("\t[%7.2e, %7.2e] x [%7.2e, %7.2e] x [%7.2e, %7.2e]\n",
    lower_x, upper_x, lower_y, upper_y, lower_z, upper_z
  );
  int mesh_x = 10, mesh_y = 10, mesh_z = 5;
  dolfin::Point p0(lower_x, lower_y, lower_z);
  dolfin::Point p1(upper_x, upper_y, upper_z);
  dolfin::BoxMesh mesh(p0, p1, mesh_x, mesh_y, mesh_z);

  // construct boundary subdomains
  printf("\tconstructing boundaries\n");
  XBoundaries bdry_x(lower_x, upper_x);
  YBoundaries bdry_y(lower_y, upper_y);
  ZBoundaries bdry_z(lower_z, upper_z);

  // mark and output mesh
  printf("\tmarking boundaries\n");
  dolfin::FacetFunction<size_t> boundary_parts(mesh);
  boundary_parts.set_all(0);
  bdry_x.mark(boundary_parts, 1);
  bdry_y.mark(boundary_parts, 2);
  bdry_z.mark(boundary_parts, 3);
  printf("\toutput boundaries\n");
  File meshfile("aux/test_bc/output/boundary_parts.pvd");
  meshfile << boundary_parts;

  // test some points in boundary
  double interior_data[3] = {
    0.5*(lower_x + upper_x),
    0.5*(lower_y + upper_y),
    0.5*(lower_z + upper_z)
  };
  dolfin::Array<double> interior(3,interior_data);
  bool interior_in_x = bdry_x.inside(interior, true);
  bool interior_in_y = bdry_y.inside(interior, true);
  bool interior_in_z = bdry_z.inside(interior, true);
  printf("\tIs an interior point in x-boundary: %d\n", interior_in_x);
  printf("\tIs an interior point in y-boundary: %d\n", interior_in_y);
  printf("\tIs an interior point in z-boundary: %d\n", interior_in_z);
  printf("\n"); fflush(stdout);

  double lower_x_data[3] = {
    lower_x,
    0.5*(lower_y + upper_y),
    0.5*(lower_z + upper_z)
  };
  double upper_x_data[3] = {
    upper_x,
    0.5*(lower_y + upper_y),
    0.5*(lower_z + upper_z)
  };
  dolfin::Array<double> lower_x_pt(3,lower_x_data);
  dolfin::Array<double> upper_x_pt(3,upper_x_data);
  bool lower_x_in_x = bdry_x.inside(lower_x_pt, true);
  bool lower_x_in_y = bdry_y.inside(lower_x_pt, true);
  bool lower_x_in_z = bdry_z.inside(lower_x_pt, true);
  printf("\tIs a lower x point in x-boundary: %d\n", lower_x_in_x);
  printf("\tIs a lower x point in y-boundary: %d\n", lower_x_in_y);
  printf("\tIs a lower x point in z-boundary: %d\n", lower_x_in_z);
  bool upper_x_in_x = bdry_x.inside(upper_x_pt, true);
  bool upper_x_in_y = bdry_y.inside(upper_x_pt, true);
  bool upper_x_in_z = bdry_z.inside(upper_x_pt, true);
  printf("\tIs an upper x point in x-boundary: %d\n", upper_x_in_x);
  printf("\tIs an upper x point in y-boundary: %d\n", upper_x_in_y);
  printf("\tIs an upper x point in z-boundary: %d\n", upper_x_in_z);
  printf("\n"); fflush(stdout);

  double lower_y_data[3] = {
    0.5*(lower_x + upper_x),
    lower_y,
    0.5*(lower_z + upper_z)
  };
  double upper_y_data[3] = {
    0.5*(lower_x + upper_x),
    upper_y,
    0.5*(lower_z + upper_z)
  };
  dolfin::Array<double> lower_y_pt(3,lower_y_data);
  dolfin::Array<double> upper_y_pt(3,upper_y_data);
  bool lower_y_in_x = bdry_x.inside(lower_y_pt, true);
  bool lower_y_in_y = bdry_y.inside(lower_y_pt, true);
  bool lower_y_in_z = bdry_z.inside(lower_y_pt, true);
  printf("\tIs a lower y point in x-boundary: %d\n", lower_y_in_x);
  printf("\tIs a lower y point in y-boundary: %d\n", lower_y_in_y);
  printf("\tIs a lower y point in z-boundary: %d\n", lower_y_in_z);
  bool upper_y_in_x = bdry_x.inside(upper_y_pt, true);
  bool upper_y_in_y = bdry_y.inside(upper_y_pt, true);
  bool upper_y_in_z = bdry_z.inside(upper_y_pt, true);
  printf("\tIs an upper y point in x-boundary: %d\n", upper_y_in_x);
  printf("\tIs an upper y point in y-boundary: %d\n", upper_y_in_y);
  printf("\tIs an upper y point in z-boundary: %d\n", upper_y_in_z);
  printf("\n"); fflush(stdout);

  double lower_z_data[3] = {
    0.5*(lower_x + upper_x),
    0.5*(lower_y + upper_y),
    lower_z
  };
  double upper_z_data[3] = {
    0.5*(lower_x + upper_x),
    0.5*(lower_y + upper_y),
    upper_z
  };
  dolfin::Array<double> lower_z_pt(3,lower_z_data);
  dolfin::Array<double> upper_z_pt(3,upper_z_data);
  bool lower_z_in_x = bdry_x.inside(lower_z_pt, true);
  bool lower_z_in_y = bdry_y.inside(lower_z_pt, true);
  bool lower_z_in_z = bdry_z.inside(lower_z_pt, true);
  printf("\tIs a lower z point in x-boundary: %d\n", lower_z_in_x);
  printf("\tIs a lower z point in y-boundary: %d\n", lower_z_in_y);
  printf("\tIs a lower z point in z-boundary: %d\n", lower_z_in_z);
  bool upper_z_in_x = bdry_x.inside(upper_z_pt, true);
  bool upper_z_in_y = bdry_y.inside(upper_z_pt, true);
  bool upper_z_in_z = bdry_z.inside(upper_z_pt, true);
  printf("\tIs an upper z point in x-boundary: %d\n", upper_z_in_x);
  printf("\tIs an upper z point in y-boundary: %d\n", upper_z_in_y);
  printf("\tIs an upper z point in z-boundary: %d\n", upper_z_in_z);
  printf("\n"); fflush(stdout);


  /**
   * Initialize analytic expression for log-charges and voltages
   */
  printf("Test initial expressions: \n");
  Poisson::FunctionSpace CG(mesh);
  LogCharge charge_x_expr(std::exp(1.0), std::exp(10.0), lower_x, upper_x, 0);
  Voltage voltage_x_expr(-1.0, 1.0, lower_x, upper_x, 0);
  dolfin::Function charge_x(CG);
  dolfin::Function voltage_x(CG);
  charge_x.interpolate(charge_x_expr);
  voltage_x.interpolate(voltage_x_expr);
  File charge_x_file("aux/test_bc/output/charge_x.pvd");
  File voltage_x_file("aux/test_bc/output/voltage_x.pvd");
  charge_x_file << charge_x;
  voltage_x_file << voltage_x;


  return 0;
}
