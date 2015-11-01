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
double pi=DOLFIN_PI;


// Exact solution
class Solution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2*pi*x[0])*cos(2*pi*x[1]);
  }
};

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] =  8*pow(pi,2)*sin(2*pi*x[0])*cos(2*pi*x[1]);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN1 : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 2*pi*sin(2*pi*x[0])*sin(2*pi*x[1]);
  }
};
class dUdN2 : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -2*pi*sin(2*pi*x[0])*sin(2*pi*x[1]);
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

class N1Boundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[1] < DOLFIN_EPS && on_boundary;
  }
};

class N2Boundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && x[1] > 1.0 - DOLFIN_EPS;
  }
};

int main(int argc, char** argv)
{
  if (argc >1)
  {
    if (std::string(argv[1])=="DEBUG") DEBUG = true;
  }

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
  UnitSquareMesh mesh(200, 200);
  Poisson::FunctionSpace V(mesh);

  FacetFunction<std::size_t> markers(mesh, 1);
  N1Boundary N1B;
  N2Boundary N2B;
  markers.set_all(0);
  N1B.mark(markers, 1);
  N2B.mark(markers, 2);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);

  Source f;
  dUdN1 g1;
  dUdN2 g2;
  L.ds = markers;
  L.f = f;
  L.g1 = g1;
  L.g2 = g2;

  // Assembl Matrix and RHS
  EigenMatrix A;
  assemble(A,a); bc.apply(A);
  EigenVector b;
  assemble(b,L); bc.apply(b);
  EigenVector Solu_vec;


  /// FENiCS Solver
  if (DEBUG) printf("Solve the Poisson's equation using FENiCS..."); fflush(stdout);
  solve(A, Solu_vec, b, "bicgstab");
  if (DEBUG) printf("done\n"); fflush(stdout);

  /// FASP Solver
  dCSRmat A_fasp;
  dvector b_fasp;
  dvector Solu_fasp;
  EigenVector_to_dvector(&b,&b_fasp);
  EigenMatrix_to_dCSRmat(&A,&A_fasp);
  fasp_dvec_alloc(b_fasp.row, &Solu_fasp);
  fasp_dvec_set(b_fasp.row, &Solu_fasp, 0.0);
  if (DEBUG) printf("Solve the Poisson's equation using FASP..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/faspsolver_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  status = fasp_solver_dcsr_krylov(&A_fasp, &b_fasp, &Solu_fasp, &itpar);
  if (DEBUG)  printf("done\n"); fflush(stdout);

  Solution ExactSolu;
  dolfin::Function solu_ex(V);
  solu_ex.interpolate(ExactSolu);

  dolfin::Function Error1(V);
  dolfin::Function Error2(V);
  dolfin::Function Error3(V);
  double error_norm1 = 0.0;
  double error_norm2 = 0.0;
  double error_norm3 = 0.0;
  copy_dvector_to_Function(&Solu_fasp,&Error1);
  copy_dvector_to_Function(&Solu_fasp,&Error2);
  *(Error3.vector())=Solu_vec;

  if (DEBUG){
    printf("Saving reulsts in tests/faspsolver_tests/output/\n"); fflush(stdout);
    dolfin::File file_markers("./tests/faspsolver_tests/output/markers.pvd");
    file_markers << markers;
    dolfin::File file_fasp("./tests/faspsolver_tests/output/FASPPoisson.pvd");
    file_fasp << Error1;
    dolfin::File file_fenics("./tests/faspsolver_tests/output/FENiCSPoisson.pvd");
    file_fenics << Error3;
    dolfin::File file_exact("./tests/faspsolver_tests/output/ExactPoisson.pvd");
    file_exact << solu_ex;
  }


  *(Error1.vector())-=*(solu_ex.vector());
  *(Error2.vector())-=*(solu_ex.vector());
  *(Error3.vector())-=*(solu_ex.vector());

  L2Error::Form_M L2error1(mesh,Error1);
  error_norm1 = assemble(L2error1);
  if (DEBUG) printf("FASP/Exact Solution L2 Error is:\t%e\n", error_norm1);

  L2Error::Form_M L2error2(mesh,Error2);
  error_norm2 = assemble(L2error2);
  if (DEBUG) printf("FASP/FENiCS L2 Error is:\t%e\n", error_norm2);

  L2Error::Form_M L2error3(mesh,Error3);
  error_norm3 = assemble(L2error3);
  if (DEBUG) printf("FENiCS/Exact Soltuion L2 Error is:\t%e\n", error_norm3);

  if ((error_norm1 < 1E-7) && (error_norm1 < 1E-7))
  {
    std::cout << "Success... the fasp solver is working\n";
  }
  else {
    std::cout << "ERROR...the fasp solver is not working\n";
  }

  // state problem
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of the test of the FASP Solver with DEBUG=TRUE      #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;
}
