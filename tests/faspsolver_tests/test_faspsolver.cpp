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
  L2Error::Form_M L2error2(mesh,Error2);
  error_norm2 = assemble(L2error2);
  L2Error::Form_M L2error3(mesh,Error3);
  error_norm3 = assemble(L2error3);


  // On the 100x100 mesh
  UnitSquareMesh mesh1(150, 150);
  Poisson::FunctionSpace V1(mesh1);
  FacetFunction<std::size_t> markers1(mesh1, 1);
  markers1.set_all(0);
  N1B.mark(markers1, 1);
  N2B.mark(markers1, 2);
  DirichletBC bc1(V1, u0, boundary);
  Poisson::BilinearForm a1(V1, V1);
  Poisson::LinearForm L1(V1);
  L1.ds = markers1;
  L1.f = f;
  L1.g1 = g1;
  L1.g2 = g2;
  EigenMatrix A1;
  assemble(A1,a1); bc1.apply(A1);
  EigenVector b1;
  assemble(b1,L1); bc1.apply(b1);
  dCSRmat A1_fasp;
  dvector b1_fasp;
  dvector Solu1_fasp;
  EigenVector_to_dvector(&b1,&b1_fasp);
  EigenMatrix_to_dCSRmat(&A1,&A1_fasp);
  fasp_dvec_alloc(b1_fasp.row, &Solu1_fasp);
  fasp_dvec_set(b1_fasp.row, &Solu1_fasp, 0.0);
  status = fasp_solver_dcsr_krylov(&A1_fasp, &b1_fasp, &Solu1_fasp, &itpar);
  dolfin::Function solu1_ex(V1);
  solu1_ex.interpolate(ExactSolu);
  dolfin::Function Error4(V1);
  copy_dvector_to_Function(&Solu1_fasp,&Error4);
  *(Error4.vector())-=*(solu1_ex.vector());
  double error_norm4 = 0.0;
  L2Error::Form_M L2error4(mesh1,Error4);
  error_norm4 = assemble(L2error4);

  // On the 300x300 mesh
  UnitSquareMesh mesh3(250, 250);
  Poisson::FunctionSpace V3(mesh3);
  FacetFunction<std::size_t> markers3(mesh3, 1);
  markers3.set_all(0);
  N1B.mark(markers3, 1);
  N2B.mark(markers3, 2);
  DirichletBC bc3(V3, u0, boundary);
  Poisson::BilinearForm a3(V3, V3);
  Poisson::LinearForm L3(V3);
  L3.ds = markers3;
  L3.f = f;
  L3.g1 = g1;
  L3.g2 = g2;
  EigenMatrix A3;
  assemble(A3,a3); bc3.apply(A3);
  EigenVector b3;
  assemble(b3,L3); bc3.apply(b3);
  dCSRmat A3_fasp;
  dvector b3_fasp;
  dvector Solu3_fasp;
  EigenVector_to_dvector(&b3,&b3_fasp);
  EigenMatrix_to_dCSRmat(&A3,&A3_fasp);
  fasp_dvec_alloc(b3_fasp.row, &Solu3_fasp);
  fasp_dvec_set(b3_fasp.row, &Solu3_fasp, 0.0);
  status = fasp_solver_dcsr_krylov(&A3_fasp, &b3_fasp, &Solu3_fasp, &itpar);
  dolfin::Function solu3_ex(V3);
  solu3_ex.interpolate(ExactSolu);
  dolfin::Function Error5(V3);
  copy_dvector_to_Function(&Solu3_fasp,&Error5);
  *(Error5.vector())-=*(solu3_ex.vector());
  double error_norm5 = 0.0;
  L2Error::Form_M L2error5(mesh3,Error5);
  error_norm5 = assemble(L2error5);


  if (DEBUG){
    printf("On the 150x150 mesh:\n");
    printf("\tFASP/Exact Solution L2 Error is:\t%e\n", error_norm4);
    printf("On the 200x200 mesh:\n");
    printf("\tFASP/Exact Solution L2 Error is:\t%e\n", error_norm1);
    printf("\tFASP/FENiCS L2 Error is:\t\t%e\n", error_norm2);
    printf("\tFENiCS/Exact Soltuion L2 Error is:\t%e\n", error_norm3);
    printf("On the 250x250 mesh:\n");
    printf("\tFASP/Exact Solution L2 Error is:\t%e\n", error_norm5);
    printf("The slopes for the FASP/Exact Solution L2 Error are:\n");
    printf("\t%e\n", std::fabs(log(error_norm4)-log(error_norm1)/(log(150.0)-log(200.0))));
    printf("\t%e\n", std::fabs(log(error_norm5)-log(error_norm1)/(log(250.0)-log(200.0))));
  }


  if ((error_norm1 < 1E-7) && (error_norm2 < 1E-7) && (error_norm1 < error_norm4) && (error_norm5 < error_norm1))
  {
    std::cout << "Success... the fasp solver is working\n";
  }
  else {
    printf("***\tERROR IN FASP SOLVER TEST\n");
    printf("***\n***\n***\n");
    printf("***\tFASP SOLVER TEST:\n");
    printf("***\tThe computed solution is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN FASP SOLVER TEST\n");
    fflush(stdout);
  }

  // state problem
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of the test of the FASP Solver with DEBUG=TRUE      #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;
}
