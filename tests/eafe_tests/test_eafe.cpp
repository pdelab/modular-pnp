/*! \file test_eafe.cpp
 *
 *  \brief Simple unit test to verify the EAFE discretization functions
 *    function properly.
 *    This test solves a problem in the x-coordinate and y-coordinate.
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

bool DEBUG = false;

double BETA_SCALE = 1.0e+0;
double LOWER_BOUNDARY = 0.0;
double UPPER_BOUNDARY = 1.0;

/*** test problem in x-coordinate ***/
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
    double b_L = BETA_SCALE*LOWER_BOUNDARY;
    double b_R = BETA_SCALE*UPPER_BOUNDARY;
    values[0] += 2.0*std::exp(b_L-b)*(UPPER_BOUNDARY-x[0]);
    values[0] += 3.0*std::exp(b_R-b)*(x[0]-LOWER_BOUNDARY);
  }
};

class AdvectionGiven_x : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = BETA_SCALE*x[0];
  }
};


/*** test problem in y-coordinate ***/
class DirichletBoundary_y : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary and (
      x[1] < LOWER_BOUNDARY + DOLFIN_EPS or x[1] > UPPER_BOUNDARY - DOLFIN_EPS
    );
  }
};

class SolutionGiven_y : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    // homogenous BC component
    double b = -BETA_SCALE*x[1]*x[1];
    values[0] = -0.5*std::exp(-b)*(x[1]-LOWER_BOUNDARY)*(x[1]-UPPER_BOUNDARY);

    // dirichlet component
    double b_L = -BETA_SCALE*LOWER_BOUNDARY*LOWER_BOUNDARY;
    double b_R = -BETA_SCALE*UPPER_BOUNDARY*UPPER_BOUNDARY;
    values[0] += 4.0*std::exp(b_L-b)*(UPPER_BOUNDARY-x[1]);
    values[0] += 2.0*std::exp(b_R-b)*(x[1]-LOWER_BOUNDARY);
  }
};

class AdvectionGiven_y : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -BETA_SCALE*x[1]*x[1];
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

  // Create mesh and function space
  int mesh_size = 100;
  dolfin::Point p0( LOWER_BOUNDARY, LOWER_BOUNDARY, 0);
  dolfin::Point p1( UPPER_BOUNDARY, UPPER_BOUNDARY, 3.0/((double)mesh_size) );
  dolfin::BoxMesh mesh(p0, p1, mesh_size, mesh_size, 3);
  Convection::FunctionSpace CG(mesh);
  dolfin::Constant unity(1.0);
  dolfin::Constant zero(0.0);
  dolfin::Function beta(CG);

  // Setup FASP
  dCSRmat adaptA_fasp;
  dvector adaptsoluvec;
  dvector adaptb_fasp;
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/eafe_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  if (DEBUG) inpar.print_level = 1;
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  /**
   * Unit test: test EAFE on a problem for the x-coordinate
   */
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### Test of EAFE.h with DEBUG=TRUE                          #### \n";
    std::cout << "################################################################# \n";
    printf("Solving the unit test for EAFE in the x-coordinate\n");
  }
  // Define analytic expressions
  SolutionGiven_x analytic_solution_x;
  AdvectionGiven_x betaGiven_x;
  // Define boundary condition, expressions, and forms
  dolfin::Function solution_x(CG);
  solution_x.interpolate(analytic_solution_x);
  DirichletBoundary_x boundary_x;
  dolfin::DirichletBC bc_x(CG, solution_x, boundary_x);
  beta.interpolate(betaGiven_x);
  EAFE::BilinearForm a_x(CG,CG);
  Convection::LinearForm L_x(CG);
  a_x.alpha = unity;
  a_x.beta = beta;
  a_x.gamma = zero;
  a_x.eta = beta;
  L_x.f = unity;
  /// Setup linear algebra objects
  dolfin::EigenMatrix A_x;
  dolfin::EigenVector b_x;
  assemble(A_x,a_x);
  bc_x.apply(A_x);
  assemble(b_x,L_x);
  bc_x.apply(b_x);
  /// Solve using FASP
  EigenMatrix_to_dCSRmat(&A_x, &adaptA_fasp);
  EigenVector_to_dvector(&b_x, &adaptb_fasp);
  fasp_dvec_alloc(adaptb_fasp.row, &adaptsoluvec);
  fasp_dvec_set(adaptb_fasp.row, &adaptsoluvec, 0.0);
  status = fasp_solver_dcsr_krylov(&adaptA_fasp, &adaptb_fasp, &adaptsoluvec, &itpar);
  dolfin::Function u_fasp_x(CG);
  copy_dvector_to_Function(&adaptsoluvec, &u_fasp_x);
  if (DEBUG) {
    printf("\tSave analytic and computed solution in VTK format\n"); fflush(stdout);
    dolfin::File file_analytic_x("./tests/eafe_tests/output/analytic_x.pvd");
    file_analytic_x << solution_x;
    dolfin::File file_computed_x("./tests/eafe_tests/output/computed_x.pvd");
    file_computed_x << u_fasp_x;
  }
  // Compute relative error
  *(u_fasp_x.vector()) -= *(solution_x.vector());
  L2Error::Functional solution_norm_x(mesh,solution_x);
  L2Error::Functional error_x(mesh,u_fasp_x);
  double error_norm_x = assemble(error_x);
  double true_norm_x = assemble(solution_norm_x);
  double relative_error_x = error_norm_x / true_norm_x;
  if (DEBUG) {
    printf("\t|| u ||_0               = %e\n", true_norm_x);
    printf("\t|| u-u_h ||_0/|| u ||_0 = %e\n", relative_error_x);
    printf("\tSave computed solution error in VTK format\n"); fflush(stdout);
    dolfin::File file_computed_x("./tests/eafe_tests/output/error_x.pvd");
    file_computed_x << u_fasp_x;
  }
  // Print if relative error is not sufficiently small
  if (relative_error_x > 1.0e-6) {
    printf("***\tERROR IN EAFE TEST ON X\n");
    printf("***\n***\n***\n");
    printf("***\tEAFE TEST ON X:\n");
    printf("***\trelative error is %e > 1.0e-6\n", relative_error_x);
    printf("***\n***\n***\n");
    printf("***\tERROR IN EAFE TEST ON X\n");
    fflush(stdout);
    return -1;
  } else {
    printf("Success... ");
    printf("passed EAFE test on x\n");
    fflush(stdout);
  }




  /**
   * Unit test: test EAFE on a problem for the y-coordinate
   */
  if (DEBUG) {
    printf("\n\n\n");
    printf("Solving the unit test for EAFE in the y-coordinate\n");
  }
  // Define analytic expressions
  SolutionGiven_y analytic_solution_y;
  AdvectionGiven_y betaGiven_y;
  // Define boundary condition, expressions, and forms
  dolfin::Function solution_y(CG);
  solution_y.interpolate(analytic_solution_y);
  DirichletBoundary_y boundary_y;
  dolfin::DirichletBC bc_y(CG, solution_y, boundary_y);
  beta.interpolate(betaGiven_y);
  EAFE::BilinearForm a_y(CG,CG);
  Convection::LinearForm L_y(CG);
  a_y.alpha = unity;
  a_y.beta = beta;
  a_y.gamma = zero;
  a_y.eta = beta;
  L_y.f = unity;
  /// Setup linear algebra objects
  dolfin::EigenMatrix A_y;
  dolfin::EigenVector b_y;
  assemble(A_y,a_y);
  bc_y.apply(A_y);
  assemble(b_y,L_y);
  bc_y.apply(b_y);
  /// Solve using FASP
  EigenMatrix_to_dCSRmat(&A_y, &adaptA_fasp);
  EigenVector_to_dvector(&b_y, &adaptb_fasp);
  fasp_dvec_set(adaptb_fasp.row, &adaptsoluvec, 0.0);
  status = fasp_solver_dcsr_krylov(&adaptA_fasp, &adaptb_fasp, &adaptsoluvec, &itpar);
  dolfin::Function u_fasp_y(CG);
  copy_dvector_to_Function(&adaptsoluvec, &u_fasp_y);
  if (DEBUG) {
    printf("\tSave analytic and computed solution in VTK format\n"); fflush(stdout);
    dolfin::File file_analytic_y("./tests/eafe_tests/output/analytic_y.pvd");
    file_analytic_y << solution_y;
    dolfin::File file_computed_y("./tests/eafe_tests/output/computed_y.pvd");
    file_computed_y << u_fasp_y;
  }
  // Compute relative error
  *(u_fasp_y.vector()) -= *(solution_y.vector());
  L2Error::Functional solution_norm_y(mesh,solution_y);
  L2Error::Functional error_y(mesh,u_fasp_y);
  double error_norm_y = assemble(error_y);
  double true_norm_y = assemble(solution_norm_y);
  double relative_error_y = error_norm_y / true_norm_y;
  if (DEBUG) {
    printf("\t|| u ||_0               = %e\n", true_norm_y);
    printf("\t|| u-u_h ||_0/|| u ||_0 = %e\n", relative_error_y);
    printf("\tSave computed solution error in VTK format\n"); fflush(stdout);
    dolfin::File file_computed_y("./tests/eafe_tests/output/error_y.pvd");
    file_computed_y << u_fasp_y;
  }
  // Print if relative error is not sufficiently small
  if (relative_error_y > 1.0e-6) {
    printf("***\tERROR IN EAFE TEST ON Y\n");
    printf("***\n***\n***\n");
    printf("***\tEAFE TEST ON Y:\n");
    printf("***\trelative error is %e > 1.0e-6\n", relative_error_y);
    printf("***\n***\n***\n");
    printf("***\tERROR IN EAFE TEST ON Y\n");
    fflush(stdout);
    return -1;
  } else {
    printf("Success... ");
    printf("passed EAFE test on y\n");
    fflush(stdout);
  }

  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of the test of EAFE                                 #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;
}
