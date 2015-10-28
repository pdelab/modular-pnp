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
#include "fasp_to_fenics.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
}
using namespace dolfin;

double U_L = 0.0;
double U_R = 1.0;
double BETA_SCALE = 1.0e+0;
double LOWER_BOUNDARY = 0.0;
double UPPER_BOUNDARY = 1.0;

// Sub domain for Dirichlet boundary condition
class DirichletBoundary_x : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary and (
      x[0] < LOWER_BOUNDARY + DOLFIN_EPS or x[0] > UPPER_BOUNDARY - DOLFIN_EPS
    );
  }
};

class SolutionGiven_x : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    // homogenous BC component
    double b = BETA_SCALE*x[0];
    values[0] = -0.5*std::exp(-b)*(x[0]-LOWER_BOUNDARY)*(x[0]-UPPER_BOUNDARY);

    // dirichlet component
    double b_L = BETA_SCALE*0.0;
    double b_R = BETA_SCALE*1.0;
    values[0] += U_L*std::exp(b_L-b)*(UPPER_BOUNDARY-x[0]);
    values[0] += U_R*std::exp(b_R-b)*(x[0]-LOWER_BOUNDARY);
  }
};

class AdvectionGiven_x : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = BETA_SCALE*x[0];
  }
};

int main()
{
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  int mesh_size = 100;
  dolfin::Point p0( LOWER_BOUNDARY, LOWER_BOUNDARY, 0);
  dolfin::Point p1( UPPER_BOUNDARY, UPPER_BOUNDARY, 3.0/((double)mesh_size) );
  dolfin::BoxMesh mesh(p0, p1, mesh_size, mesh_size, 3);
  Convection::FunctionSpace CG(mesh);

  // Define analytic expressions
  SolutionGiven_x analytic_solution_x;
  AdvectionGiven_x betaGiven_x;

  // Define boundary condition
  dolfin::Function solution_x(CG);  
  solution_x.interpolate(analytic_solution_x);
  DirichletBoundary_x boundary_x;
  dolfin::DirichletBC bc_x(CG, solution_x, boundary_x);

  // Define analytic expressions
  dolfin::Constant unity(1.0);
  dolfin::Function beta(CG);
  beta.interpolate(betaGiven_x);
  dolfin::Constant zero(0.0);

  // Standard convection problem
  EAFE::BilinearForm a_x(CG,CG);
  Convection::LinearForm L(CG);
  a_x.alpha = unity;
  a_x.beta = beta;
  a_x.gamma = zero;
  a_x.eta = beta;
  L.f = unity;

  /// Solve for solution
  dolfin::EigenMatrix A_x;
  dolfin::EigenVector u_vector;
  dolfin::EigenVector b_x;
  assemble(A_x,a_x); 
  bc_x.apply(A_x);
  assemble(b_x,L);
  bc_x.apply(b_x);

  /// solve using FASP
  dCSRmat adaptA_fasp;
  dvector adaptsoluvec;
  dvector adaptb_fasp;
  EigenMatrix_to_dCSRmat(&A_x, &adaptA_fasp);
  EigenVector_to_dvector(&b_x, &adaptb_fasp);
  fasp_dvec_alloc(adaptb_fasp.row, &adaptsoluvec);
  fasp_dvec_set(adaptb_fasp.row, &adaptsoluvec, 0.0);

  // setup solver
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/eafe_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  status = fasp_solver_dcsr_krylov(&adaptA_fasp, &adaptb_fasp, &adaptsoluvec, &itpar);
  dolfin::Function u_fasp(CG);
  copy_dvector_to_Function(&adaptsoluvec, &u_fasp);

  /**
   * Unit test: compare to analytic solution
   */
  *(u_fasp.vector()) -= *(solution_x.vector());
  L2Error::Functional solution_norm(mesh,solution_x);
  L2Error::Functional error(mesh,u_fasp);
  double error_norm = assemble(error);
  double true_norm = assemble(solution_norm);
  double relative_error = error_norm / true_norm;
  // Print if relative error is not sufficiently small
  if (relative_error > 1.0e-6) {
    printf("***\tERROR IN EAFE TEST ON X\n");
    printf("***\n***\n***\n");
    printf("***\tEAFE TEST ON X:\n");
    printf("***\terror is %e > 1.0e-6\n", relative_error);
    printf("***\n***\n***\n");
    printf("***\tERROR IN EAFE TEST ON X\n");
    fflush(stdout);
    return -1;
  } else {
    printf("Success... ");
    printf("passed EAFE test on x\n");
    fflush(stdout);
  }


  return 0;
}
