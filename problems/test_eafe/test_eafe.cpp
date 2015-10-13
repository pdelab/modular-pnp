/*! \file test_faspfenics.cpp
 *
 *  \brief Main to test EAFE functionality on linear convection reaction problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include "Convection.h"
#include "EAFE.h"
using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

class Advection : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double x0 = x[0];
    values[0] = 1.0e+0*x0;
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary and (
      x[0] < DOLFIN_EPS 
      or x[0] > 1.0 - DOLFIN_EPS
      or x[1] < DOLFIN_EPS 
      or x[1] > 1.0 - DOLFIN_EPS
    );
  }
};

int main()
{

  int i;

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  printf("Create mesh\n"); fflush(stdout);
  unsigned int mesh_size = 10;
  bool print_matrices = (mesh_size<5)? true : false;
  dolfin::UnitCubeMesh mesh(mesh_size, mesh_size, mesh_size);
  Convection::FunctionSpace CG(mesh);

  // Define boundary condition
  printf("Define boundary condition\n"); fflush(stdout);
  dolfin::Constant u0(0.0);
  DirichletBoundary boundary;
  dolfin::DirichletBC bc(CG, u0, boundary);

  // Define analytic expressions
  printf("Define analytic expressions\n"); fflush(stdout);
  dolfin::Constant alpha(1.0e-0);
  
  Advection betaExpression;
  dolfin::Function beta(CG);
  beta.interpolate(betaExpression);

  dolfin::Constant gamma(1.0);
  dolfin::Constant eta(0.0);
  dolfin::Constant f(1.0);



  /// Standard convection problem
  printf("Solve convection problem using standard formulation\n"); fflush(stdout);
  // Define variational forms
  printf("\tDefine variational forms\n"); fflush(stdout);
  Convection::BilinearForm a(CG,CG);
  Convection::LinearForm L(CG);
  a.alpha = alpha;
  a.beta = beta;
  a.gamma = gamma;
  a.eta = eta;
  L.f = f;

  // Compute solution
  printf("\tCompute solution\n\t"); fflush(stdout);
  dolfin::Function u(CG);
  solve(a == L, u, bc);

  // Save solution in VTK format
  printf("\tSave solution in VTK format\n"); fflush(stdout);
  dolfin::File file("./problems/test_eafe/output/Convection.pvd");
  file << u;



  /// EAFE convection problem
  printf("Solve convection problem using EAFE formulation\n"); fflush(stdout);
  // Define variational forms
  EAFE::BilinearForm a_eafe(CG,CG);
  a_eafe.alpha = alpha;
  a_eafe.beta = beta;
  a_eafe.gamma = gamma;
  a_eafe.eta = eta;

  // Compute solution
  printf("\tCompute solution\n\t"); fflush(stdout);
  dolfin::Function u_eafe(CG);
  solve(a_eafe == L, u_eafe, bc);

  // Save solution in VTK format
  printf("\tSave solution in VTK format\n"); fflush(stdout);
  dolfin::File file_eafe("./problems/test_eafe/output/EAFEConvection.pvd");
  file_eafe << u_eafe;



  /// Print stiffness matrices
  if (print_matrices) {
    dolfin::EigenMatrix A; 
    assemble(A,a); A.compress();
    std::cout << "There are " << A.nnz() << " nonzero entries in the standard formulation\n";

    dolfin::EigenMatrix A_eafe; 
    assemble(A_eafe,a_eafe); A_eafe.compress();
    std::cout << "There are " << A_eafe.nnz() << " nonzero entries in the EAFE formulation\n\n";

    std::cout << "The standard stiffness matrix is:\n";  
    std::string A_string = A.str(true);
    std::cout << A_string << "\n\n";

    std::cout << "The EAFE stiffness matrix is:\n";  
    std::string A_eafe_string = A_eafe.str(true);  
    std::cout << A_eafe_string;
    printf("\n");
  }

  // exit successfully
  printf("Done\n"); fflush(stdout);
  return 0;
}
