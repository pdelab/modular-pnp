/*! \file lin_pnp.cpp
 *
 *  \brief Setup and solve the linearized PNP equation using FASP
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "poisson.h"
#include "newton.h"
#include "newton_functs.h"
#include "L2Error.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
#define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

double pi=DOLFIN_PI;


class InitialExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = 0.9*sin(2*pi*x[0]);
  }
};

class analyticExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = sin(2*pi*x[0]);
  }
};

class SourceExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = -8*pow(pi,2)*cos(4*pi*x[0]);
  }
};

class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && ( x[0] < -1.0+DOLFIN_EPS || x[0] > 1.0 -DOLFIN_EPS );
  }
};

int main()
{

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n Solving the linearized Poisson-Nernst-Planck system           "); fflush(stdout);
  printf("\n of a single cation and anion                                  "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  //*************************************************************
  //  Initialization
  //*************************************************************

  IntervalMesh mesh(100, -1.0, 1.0);

  // open files for outputting solutions
  File FileSolu("./benchmarks/Newton_Poisson/output/solution.pvd");
  File FileExact("./benchmarks/Newton_Poisson/output/exact.pvd");
  File FileError("./benchmarks/Newton_Poisson/output/error.pvd");

  // Initialize variational forms
  printf("\tvariational forms...\n"); fflush(stdout);
  poisson::FunctionSpace V(mesh);
  poisson::BilinearForm a(V,V);
  poisson::LinearForm L(V);


  // analytic solution
  SourceExpression f;
  analyticExpression u;
  Function analyticSolution(V);
  analyticSolution.interpolate(u);

  L.f=f;

  // Set Dirichlet boundaries
  printf("\tboundary conditions...\n"); fflush(stdout);
  DirichletBoundary boundary;
  Constant zero(0.0);
  dolfin::DirichletBC bc(V, zero, boundary);

  // Interpolate analytic expressions
  InitialExpression InitialGuess;
  Function solution(V);

  solution.interpolate(InitialGuess);

  // print to file
  FileSolu << solution;
  FileExact << analyticSolution;

  // initialize linear system
  printf("\tlinear algebraic objects...\n"); fflush(stdout);
  EigenMatrix A;
  EigenVector b;
  dCSRmat A_fasp;
  dvector b_fasp, solu_fasp;


  // Setup FASP solver
  printf("\tsetup FASP solver...\n"); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./benchmarks/Newton_Poisson/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  //*************************************************************
  //  Initialize Newton solver
  //*************************************************************
  // Setup newton parameters and compute initial residual
  printf("\tnewton solver setup...\n"); fflush(stdout);
  Function dsolution(V);
  dsolution.interpolate(zero);
  unsigned int newton_iteration = 0;

  // compute initial residual and Jacobian
  printf("\tconstruct residual...\n"); fflush(stdout);
  L.u = solution;
  assemble(b, L);
  bc.apply(b);
  double initial_residual = b.norm("l2");
  double relative_residual = initial_residual;//1.0;
  printf("\tinitial nonlinear residual has l2-norm of %e\n", initial_residual);

  printf("\tinitialized succesfully!\n\n"); fflush(stdout);

  fasp_dvec_alloc(b.size(), &solu_fasp);

  //*************************************************************
  //  Solve : this will be inside Newton loop
  //*************************************************************
  double tol=1E-8;
  while (relative_residual>tol)
  {
      printf("Solve the system\n"); fflush(stdout);
      newton_iteration++;

      // Construct stiffness matrix
      printf("\tconstruct stiffness matrix...\n"); fflush(stdout);
      a.u = solution;
      assemble(A, a);
      bc.apply(A);

      // Convert to fasp
      printf("\tconvert to FASP and solve...\n"); fflush(stdout);
      EigenVector_to_dvector(&b,&b_fasp);
      EigenMatrix_to_dCSRmat(&A,&A_fasp);
      fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
      status = fasp_solver_dcsr_krylov(&A_fasp, &b_fasp, &solu_fasp, &itpar);

      // map solu_fasp into solutionUpdate
      printf("\tconvert FASP solution to function...\n"); fflush(stdout);
      copy_dvector_to_Function(&solu_fasp, &dsolution);
      update_solution(&solution, &dsolution);

      // *(solution.vector())+=*(dsolution.vector());
      // compute residual
      L.u = solution;
      assemble(b,L);
      bc.apply(b);
      relative_residual = b.norm("l2") ;// initial_residual;
      if (newton_iteration == 1)
        printf("\trelative nonlinear residual after 1 iteration has l2-norm of %e\n", relative_residual);
      else
        printf("\trelative nonlinear residual after %d iterations has l2-norm of %e\n", newton_iteration, relative_residual);

      // write computed solution to file
      printf("\tsolved successfully!\n"); fflush(stdout);
      FileSolu << solution;

      // compute solution error
      printf("\nCompute the error\n"); fflush(stdout);
      Function Error(analyticSolution);
      *(Error.vector())-=*(solution.vector());
      double L2_Error = 0.0;
      L2Error::Form_M L2error(mesh,Error);
      L2_Error = assemble(L2error);
      printf("\tL2 error is:\t%e\n", L2_Error);

      // print error
      // cationFile << cationSolution;
      // anionFile << anionSolution;
      // potentialFile << potentialSolution;
  }

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
