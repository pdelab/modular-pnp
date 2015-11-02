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

bool DEBUG = false;

double lower_x = -1.0e+1;
double upper_x =  1.0e+1;
double lower_y = -1.0e+1;
double upper_y =  1.0e+1;
double lower_z = -5.0e+0;
double upper_z =  5.0e+0;

class LogChargeX : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = 10.0 * (x[0] - lower_x) / (upper_x - lower_x);
    values[0] +=  1.0 * (upper_x - x[0]) / (upper_x - lower_x);
  }
};

class VoltageX : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = +1.0 * (x[0] - lower_x) / (upper_x - lower_x);
    values[0] += -1.0 * (upper_x - x[0]) / (upper_x - lower_x);
  }
};

class LogChargeY : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = std::log(2.0) * (x[1] - lower_y) / (upper_y - lower_y);
    values[0] += std::log(1.0) * (upper_y - x[1]) / (upper_y - lower_y);
  }
};

class VoltageY : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = +10.0 * (x[1] - lower_y) / (upper_y - lower_y);
    values[0] += -10.0 * (upper_y - x[1]) / (upper_y - lower_y);
  }
};

class LogChargeZ : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  =  0.0 * (x[2] - lower_z) / (upper_z - lower_z);
    values[0] += -1.0 * (upper_z - x[2]) / (upper_z - lower_z);
  }
};

class VoltageZ : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = -5.0 * (x[2] - lower_z) / (upper_z - lower_z);
    values[0] += -2.0 * (upper_z - x[2]) / (upper_z - lower_z);
  }
};

int main(int argc, char** argv)
{
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;
  if (argc >1)
  {
    if (std::string(argv[1])=="DEBUG") DEBUG = true;
  }

  /**
   * First test to see if the boundary conditions works
   */
  // initialize mesh
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### Test of boundary_condtions.cpp                          #### \n";
    std::cout << "################################################################# \n";
    printf("Test subdomains: \n");
    printf("\tInitializing mesh: \n");
    printf("\t[%7.2e, %7.2e] x [%7.2e, %7.2e] x [%7.2e, %7.2e]\n",
      lower_x, upper_x, lower_y, upper_y, lower_z, upper_z
    );
  }
  int mesh_x = 10, mesh_y = 10, mesh_z = 5;
  dolfin::Point p0(lower_x, lower_y, lower_z);
  dolfin::Point p1(upper_x, upper_y, upper_z);
  dolfin::BoxMesh mesh(p0, p1, mesh_x, mesh_y, mesh_z);

  // construct boundary subdomains
  if (DEBUG) printf("\tconstructing boundaries\n");
  XBoundaries bdry_x(lower_x, upper_x);
  YBoundaries bdry_y(lower_y, upper_y);
  ZBoundaries bdry_z(lower_z, upper_z);

  // mark and output mesh
  if (DEBUG) printf("\tmarking boundaries\n");
  dolfin::FacetFunction<size_t> boundary_parts(mesh);
  boundary_parts.set_all(0);
  bdry_x.mark(boundary_parts, 1);
  bdry_y.mark(boundary_parts, 2);
  bdry_z.mark(boundary_parts, 3);
  if (DEBUG) {
    printf("\toutput boundaries\n");
    File meshfile("tests/bc_tests/output/boundary_parts.pvd");
    boundary_parts.rename("boundaries (x=1, y=2, z=3)", "boundaries");
    meshfile << boundary_parts;
  }
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
  if (DEBUG) {
    printf("\tIs an interior point in x-boundary: %d\n", interior_in_x);
    printf("\tIs an interior point in y-boundary: %d\n", interior_in_y);
    printf("\tIs an interior point in z-boundary: %d\n", interior_in_z);
    printf("\n"); fflush(stdout);
  }
  if (interior_in_x || interior_in_y || interior_in_z) {
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    printf("***\n***\n***\n");
    printf("***\tSUBDOMAINS TEST:\n");
    printf("***\tThe interior point is on a boundary\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    fflush(stdout);
    return -1;
  }

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
  bool upper_x_in_x = bdry_x.inside(upper_x_pt, true);
  bool upper_x_in_y = bdry_y.inside(upper_x_pt, true);
  bool upper_x_in_z = bdry_z.inside(upper_x_pt, true);
  if (DEBUG) {
    printf("\tIs a lower x point in x-boundary: %d\n", lower_x_in_x);
    printf("\tIs a lower x point in y-boundary: %d\n", lower_x_in_y);
    printf("\tIs a lower x point in z-boundary: %d\n", lower_x_in_z);
    printf("\tIs an upper x point in x-boundary: %d\n", upper_x_in_x);
    printf("\tIs an upper x point in y-boundary: %d\n", upper_x_in_y);
    printf("\tIs an upper x point in z-boundary: %d\n", upper_x_in_z);
    printf("\n"); fflush(stdout);
  }
  if (!lower_x_in_x || lower_x_in_y || lower_x_in_z
    || !upper_x_in_x || upper_x_in_y || upper_x_in_z) {
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    printf("***\n***\n***\n");
    printf("***\tSUBDOMAINS TEST:\n");
    printf("***\tThe x-boundary point is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    fflush(stdout);
    return -1;
  }

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
  bool upper_y_in_x = bdry_x.inside(upper_y_pt, true);
  bool upper_y_in_y = bdry_y.inside(upper_y_pt, true);
  bool upper_y_in_z = bdry_z.inside(upper_y_pt, true);
  if (DEBUG) {
    printf("\tIs a lower y point in x-boundary: %d\n", lower_y_in_x);
    printf("\tIs a lower y point in y-boundary: %d\n", lower_y_in_y);
    printf("\tIs a lower y point in z-boundary: %d\n", lower_y_in_z);
    printf("\tIs an upper y point in x-boundary: %d\n", upper_y_in_x);
    printf("\tIs an upper y point in y-boundary: %d\n", upper_y_in_y);
    printf("\tIs an upper y point in z-boundary: %d\n", upper_y_in_z);
    printf("\n"); fflush(stdout);
  }
  if (lower_y_in_x || !lower_y_in_y || lower_y_in_z
    || upper_y_in_x || !upper_y_in_y || upper_y_in_z) {
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    printf("***\n***\n***\n");
    printf("***\tSUBDOMAINS TEST:\n");
    printf("***\tThe y-boundary point is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    fflush(stdout);
    return -1;
  }

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
  bool upper_z_in_x = bdry_x.inside(upper_z_pt, true);
  bool upper_z_in_y = bdry_y.inside(upper_z_pt, true);
  bool upper_z_in_z = bdry_z.inside(upper_z_pt, true);
  if (DEBUG) {
    printf("\tIs a lower z point in x-boundary: %d\n", lower_z_in_x);
    printf("\tIs a lower z point in y-boundary: %d\n", lower_z_in_y);
    printf("\tIs a lower z point in z-boundary: %d\n", lower_z_in_z);
    printf("\tIs an upper z point in x-boundary: %d\n", upper_z_in_x);
    printf("\tIs an upper z point in y-boundary: %d\n", upper_z_in_y);
    printf("\tIs an upper z point in z-boundary: %d\n", upper_z_in_z);
    printf("\n"); fflush(stdout);
  }
  if (lower_z_in_x || lower_z_in_y || !lower_z_in_z
    || upper_z_in_x || upper_z_in_y || !upper_z_in_z) {
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    printf("***\n***\n***\n");
    printf("***\tSUBDOMAINS TEST:\n");
    printf("***\tThe z-boundary point is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN SUBDOMAINS TEST\n");
    fflush(stdout);
    return -1;
  }

  printf("Success... passed initialization of SubDomians \n");
  fflush(stdout);


  /**
   * Initialize analytic expression for log-charges and voltages
   */
  if (DEBUG) printf("Test initial expressions: \n");
  Poisson::FunctionSpace CG(mesh);

  if (DEBUG) printf("\talong x-coordinate \n");
  LogCharge charge_x_expr(std::exp(1.0), std::exp(10.0), lower_x, upper_x, 0);
  Voltage voltage_x_expr(-1.0, 1.0, lower_x, upper_x, 0);
  dolfin::Function charge_x(CG);
  dolfin::Function voltage_x(CG);
  charge_x.interpolate(charge_x_expr);
  voltage_x.interpolate(voltage_x_expr);
  // set local expressions
  LogChargeX local_charge_x_expr;
  VoltageX local_voltage_x_expr;
  dolfin::Function local_charge_x(CG);
  dolfin::Function local_voltage_x(CG);
  local_charge_x.interpolate(local_charge_x_expr);
  local_voltage_x.interpolate(local_voltage_x_expr);
  // compute error
  *(local_charge_x.vector()) -= *(charge_x.vector());
  *(local_voltage_x.vector()) -= *(voltage_x.vector());
  double charge_x_error = local_charge_x.vector()->norm("l2");
  double voltage_x_error = local_voltage_x.vector()->norm("l2");
  if (DEBUG) {
    printf("\tcharge error:  %e \n", charge_x_error);
    printf("\tvoltage error: %e \n", voltage_x_error);
  }
  if (charge_x_error > 1E-5 || voltage_x_error > 1E-5) {
    printf("***\tERROR IN BC INTERPOLATION TEST\n");
    printf("***\n***\n***\n");
    printf("***\tBC INTERPOLATION TEST:\n");
    printf("***\tThe x-boundary interpolation is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN BC INTERPOLATION TEST\n");
    fflush(stdout);
    return -1;
  }

  if (DEBUG) printf("\talong y-coordinate \n");
  LogCharge charge_y_expr(1.0, 2.0, lower_y, upper_y, 1);
  Voltage voltage_y_expr(-10.0, 10.0, lower_y, upper_y, 1);
  dolfin::Function charge_y(CG);
  dolfin::Function voltage_y(CG);
  charge_y.interpolate(charge_y_expr);
  voltage_y.interpolate(voltage_y_expr);
  // set local expressions
  LogChargeY local_charge_y_expr;
  VoltageY local_voltage_y_expr;
  dolfin::Function local_charge_y(CG);
  dolfin::Function local_voltage_y(CG);
  local_charge_y.interpolate(local_charge_y_expr);
  local_voltage_y.interpolate(local_voltage_y_expr);
  // compute error
  *(local_charge_y.vector()) -= *(charge_y.vector());
  *(local_voltage_y.vector()) -= *(voltage_y.vector());
  double charge_y_error = local_charge_y.vector()->norm("l2");
  double voltage_y_error = local_voltage_y.vector()->norm("l2");
  if (DEBUG) {
    printf("\tcharge error:  %e \n", charge_y_error);
    printf("\tvoltage error: %e \n", voltage_y_error);
  }
  if (charge_y_error > 1E-5 || voltage_y_error > 1E-5) {
    printf("***\tERROR IN BC INTERPOLATION TEST\n");
    printf("***\n***\n***\n");
    printf("***\tBC INTERPOLATION TEST:\n");
    printf("***\tThe y-boundary interpolation is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN BC INTERPOLATION TEST\n");
    fflush(stdout);
    return -1;
  }

  if (DEBUG) printf("\talong z-coordinate \n");
  LogCharge charge_z_expr(std::exp(-1.0), 1.0, lower_z, upper_z, 2);
  Voltage voltage_z_expr(-2.0, -5.0, lower_z, upper_z, 2);
  dolfin::Function charge_z(CG);
  dolfin::Function voltage_z(CG);
  charge_z.interpolate(charge_z_expr);
  voltage_z.interpolate(voltage_z_expr);
  // set local expressions
  LogChargeZ local_charge_z_expr;
  VoltageZ local_voltage_z_expr;
  dolfin::Function local_charge_z(CG);
  dolfin::Function local_voltage_z(CG);
  local_charge_z.interpolate(local_charge_z_expr);
  local_voltage_z.interpolate(local_voltage_z_expr);
  // compute error
  *(local_charge_z.vector()) -= *(charge_z.vector());
  *(local_voltage_z.vector()) -= *(voltage_z.vector());
  double charge_z_error = local_charge_z.vector()->norm("l2");
  double voltage_z_error = local_voltage_z.vector()->norm("l2");
  if (DEBUG) {
    printf("\tcharge error:  %e \n", charge_z_error);
    printf("\tvoltage error: %e \n", voltage_z_error);
  }
  if (charge_z_error > 1E-5 || voltage_z_error > 1E-5) {
    printf("***\tERROR IN BC INTERPOLATION TEST\n");
    printf("***\n***\n***\n");
    printf("***\tBC INTERPOLATION TEST:\n");
    printf("***\tThe z-boundary interpolation is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN BC INTERPOLATION TEST\n");
    fflush(stdout);
    return -1;
  }

  if (DEBUG) {
    File charge_x_file("tests/bc_tests/output/charge_x.pvd");
    File voltage_x_file("tests/bc_tests/output/voltage_x.pvd");
    charge_x_file << charge_x;
    voltage_x_file << voltage_x;

    File charge_y_file("tests/bc_tests/output/charge_y.pvd");
    File voltage_y_file("tests/bc_tests/output/voltage_y.pvd");
    charge_y_file << charge_y;
    voltage_y_file << voltage_y;

    File charge_z_file("tests/bc_tests/output/charge_z.pvd");
    File voltage_z_file("tests/bc_tests/output/voltage_z.pvd");
    charge_z_file << charge_z;
    voltage_z_file << voltage_z;
  }

  printf("Success... passed initializing BC interpolation\n");
  if (DEBUG){
    std::cout << "################################################################# \n";
    std::cout << "#### End of test of boundary_condtions.cpp                   #### \n";
    std::cout << "################################################################# \n";
  }
  return 0;
}
