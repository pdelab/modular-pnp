// test_faspsolver solves Poisson's equation with FENiCS and FASP,
//  then compares the two solutions.
//
// The Poisson's equations is as follows
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// and boundary conditions given by
//
//     u(x, y) = 0        for x = 0 or x = 1
// du/dn(x, y) = sin(5*x) for y = 0 or y = 1
//
// It compares the

#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include "fasp_to_fenics.h"
#include "Poisson.h"
#include "L2Error.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
  INT fasp_solver_dcsr_krylov (dCSRmat *A,
   dvector *b,
   dvector *x,
   itsolver_param *itparam);
#define FASP_BSR     ON  /** use BSR format in fasp */
}

using namespace dolfin;
// using namespace std;
bool DEBUG = false;

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

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main(int argc, char** argv)
{
  // state problem
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### Test of the FASP Solver with DEBUG=TRUE                 #### \n";
    std::cout << "################################################################# \n";
  }

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  UnitSquareMesh mesh(32, 32);
  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);

  Source f;
  dUdN g;
  L.f = f;
  L.g = g;

  // Assembl Matrix and RHS
  EigenMatrix A;
  assemble(A,a); bc.apply(A);
  EigenVector b;
  assemble(b,L); bc.apply(b);
  EigenVector Solu_vec;


  /// FENiCS Solver
  printf("Solve the Poisson's equation using FENiCS..."); fflush(stdout);
  solve(A, Solu_vec, b, "bicgstab");
  printf("done\n"); fflush(stdout);

  /// FASP Solver
  dCSRmat A_fasp;
  dvector b_fasp;
  dvector Solu_fasp;
  EigenVector_to_dvector(&b,&b_fasp);
  EigenMatrix_to_dCSRmat(&A,&A_fasp);
  fasp_dvec_alloc(b_fasp.row, &Solu_fasp);
  fasp_dvec_set(b_fasp.row, &Solu_fasp, 0.0);
  printf("Solve the Poisson's equation using FASP..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/faspsolver_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  status = fasp_solver_dcsr_krylov(&A_fasp, &b_fasp, &Solu_fasp, &itpar);
  printf("done\n"); fflush(stdout);

  Function Error(V);
  double error_norm = 0.0;
  copy_dvector_to_Function(&Solu_fasp,&Error);
  *(Error.vector())-= Solu_vec;

  error_norm = Error.vector()->norm("linf");
  printf("L2 Error is:\t%e\n", error_norm);

  L2Error::Functional error_l2(mesh,Error);
  error_norm = assemble(error_l2);

  return 0;
}
