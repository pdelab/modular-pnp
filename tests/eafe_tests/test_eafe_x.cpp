/*! \file test_eafe_x.cpp
 *
 *  \brief Simple unit test to verify the EAFE discretization functions
 *    function properly.  This test solves a problem in the x-coordinate
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include "Convection.h"
#include "EAFE.h"
#include "L2Error.h"
using namespace dolfin;

double u_L = 0.0;
double u_R = 0.0;
double beta_scale = 1.0e+1;
double lower_boundary = 0.0;
double upper_boundary = 1.0;

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary and (
      x[0] < lower_boundary + DOLFIN_EPS or x[0] > upper_boundary - DOLFIN_EPS
    );
  }
};

// Analytic expression for the advection potential
class AdvectionGiven : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = beta_scale*x[0];
  }
};

class SolutionGiven : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    // homogenous BC component
    double b = beta_scale*x[0];
    values[0] = -0.5*std::exp(-b)*(x[0]-lower_boundary)*(x[0]-upper_boundary);

    // dirichlet component
    double b_L = beta_scale*0.0;
    double b_R = beta_scale*1.0;
    values[0] += u_L*std::exp(b_L-b)*(upper_boundary-x[0]);
    values[0] += u_R*std::exp(b_R-b)*(x[0]-lower_boundary) );
  }
};

int main()
{
  // state problem
  printf("Solving the linear PDE:\n");
  printf("\t-div( alpha*exp(eta)*( grad(u) + <b_x,b_y,b_z>*u ) ) + gamma*exp(eta)u = f\n");
  printf("\n"); fflush(stdout);

  // read in coefficients
  int mesh_size = 100;

  printf("Solving the test problem with f = 1 for a known solution, given b\n");
  printf("\tImposing alpha = 1, eta = b, gamma = 0\n");


  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  printf("Create mesh %d x %d x 3\n",mesh_size,mesh_size); fflush(stdout);
  dolfin::Point p0( lower_boundary, lower_boundary, 0);
  dolfin::Point p1( upper_boundary, upper_boundary, 3.0/((double)mesh_size) );
  dolfin::BoxMesh mesh(p0, p1, mesh_size, mesh_size, 3);
  Convection::FunctionSpace CG(mesh);

  // Define analytic expressions
  SolutionGiven analytic_solution;
  AdvectionGiven betaGiven(alpha_double);

  // Define boundary condition
  printf("Define boundary condition\n"); fflush(stdout);
  dolfin::Function u0(CG);  
  solution.interpolate(analytic_solution);
  DirichletBoundary boundary;
  dolfin::DirichletBC bc(CG, solution, boundary);



  ///////
  /////// remove
  ///////
  printf("\tSave mesh in VTK format\n"); fflush(stdout);
  dolfin::FacetFunction<std::size_t> markedMesh(mesh);
  markedMesh.set_all(1);
  boundary.mark(markedMesh,2);
  dolfin::File fileMesh("./tests/eafe_tests/output/mesh.pvd");
  fileMesh << markedMesh;


  // Define analytic expressions
  printf("Define analytic expressions\n"); fflush(stdout);  
  dolfin::Constant unity(1.0);
  dolfin::Function beta(CG);
  beta.interpolate(betaGiven);
  dolfin::Constant zero(0.0);


  /////
  ///// remove
  /////
  printf("\tSave RHS in VTK format\n"); fflush(stdout);
  dolfin::File fileRHS("./tests/eafe_tests/output/RHS.pvd");
  fileRHS << f;
  printf("\tSave true solution in VTK format\n");
  dolfin::File fileSolution("./tests/eafe_tests/output/solution.pvd");
  fileSolution << solution;
  printf("\n"); fflush(stdout);



  /// Standard convection problem
  printf("Solve convection problem using EAFE formulation\n"); fflush(stdout);
  EAFE::BilinearForm a_eafe(CG,CG);
  Convection::LinearForm L(CG);
  a_eafe.alpha = unity;
  a_eafe.beta = beta;
  a_eafe.gamma = zero;
  a_eafe.eta = beta;
  L.f = unity;
  printf("\n");


  /// Solve for solutions
  // Compute standard solution via linear solver
  printf("Compute solutions\n"); fflush(stdout);
  dolfin::EigenVector b;
  assemble(b,L); bc.apply(b);
  dolfin::EigenMatrix A_eafe;
  dolfin::EigenVector u_eafe_vector;
  assemble(A_eafe,a_eafe); bc.apply(A_eafe); A_eafe.compress();
  solve(A_eafe, u_eafe_vector, b, "bicgstab");
  // convert to Function
  dolfin::Function u_eafe(CG);
  *(u_eafe.vector()) = u_eafe_vector;
  // Save solution in VTK format
  printf("\tSave EAFE solution in VTK format\n"); fflush(stdout);
  dolfin::File file_eafe("./tests/eafe_tests/output/EAFEConvection.pvd");
  file_eafe << u_eafe;


  /// Compare to analytic solution for test_problem == 3
  printf("Compute the L2 error of the computed solutions\n"); fflush(stdout);
  *(u_eafe.vector()) -= *(solution.vector());
  L2Error::Functional error_eafe(mesh,u_eafe);
  double error_norm = assemble(error_eafe);
  printf("\tEAFE computed solution:\t%e\n", error_norm);
  printf("\n"); fflush(stdout);

  // exit successfully
  printf("Done\n"); fflush(stdout);
  return 0;
}
