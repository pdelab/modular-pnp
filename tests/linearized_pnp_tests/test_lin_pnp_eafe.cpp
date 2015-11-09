/*! \file test_lin_pnp.cpp
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
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "linear_pnp.h"
#include "newton.h"
#include "newton_functs.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
  INT fasp_solver_dcsr_krylov (dCSRmat *A,
   dvector *b,
   dvector *x,
   itsolver_param *itparam
  );
#define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

bool DEBUG = false;

double lower_cation_val = 0.1;  // 1 / m^3
double upper_cation_val = 1.0;  // 1 / m^3
double lower_anion_val = 1.0;  // 1 / m^3
double upper_anion_val = 0.1;  // 1 / m^3
double lower_potential_val = -1.0;  // V
double upper_potential_val = 1.0;  // V

class analyticCationExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_cation_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_cation_val * (x[0] + 5.0) / 10.0;
    values[0] += 2.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
    values[0]  = std::log(values[0]);
  }
};

class analyticAnionExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_anion_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_anion_val * (x[0] + 5.0) / 10.0;
    values[0] += 1.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
    values[0]  = std::log(values[0]);
  }
};

class analyticPotentialExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_potential_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_potential_val * (x[0] + 5.0) / 10.0;
    values[0] += -2.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
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
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  //*************************************************************
  //  Initialization
  //*************************************************************
  if (DEBUG) printf("Initialize the problem\n");
  // read domain parameters
  if (DEBUG) printf("\tdomain...\n");
  domain_param domain_par;
  char domain_param_filename[] = "./tests/linearized_pnp_tests/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  // print_domain_param(&domain_par);

  // build mesh
  if (DEBUG) printf("\tmesh...\n");
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);

  // read coefficients and boundary values
  if (DEBUG) printf("\tcoefficients...\n");
  coeff_param coeff_par;
  char coeff_param_filename[] = "./tests/linearized_pnp_tests/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  // print_coeff_param(&coeff_par);

  // Initialize variational forms
  if (DEBUG) printf("\tvariational forms...\n");
  linear_pnp::FunctionSpace V(mesh);
  linear_pnp::BilinearForm a_pnp(V,V);
  linear_pnp::LinearForm L_pnp(V);
  Constant eps(coeff_par.relative_permittivity);
  Constant Dp(coeff_par.cation_diffusivity);
  Constant Dn(coeff_par.anion_diffusivity);
  Constant qn(coeff_par.cation_mobility);
  Constant qp(coeff_par.anion_mobility);
  Constant zero(0.0);
  a_pnp.eps = eps; L_pnp.eps = eps;
  a_pnp.Dp = Dp; L_pnp.Dp = Dp;
  a_pnp.Dn = Dn; L_pnp.Dn = Dn;
  a_pnp.qp = qp; L_pnp.qp = qp;
  a_pnp.qn = qn; L_pnp.qn = qn;

  //EAFE Formulation
  Constant gamma(0.0);
  EAFE::FunctionSpace V_cat(mesh);
  EAFE::BilinearForm a_cat(V_cat,V_cat);
  a_cat.alpha = Dp;
  a_cat.gamma = gamma;
  EAFE::FunctionSpace V_an(mesh);
  EAFE::BilinearForm a_an(V_an,V_an);
  a_an.alpha = Dn;
  a_an.gamma = gamma;

  // analytic solution
  Function analyticSolutionFunction(V);
  Function analyticCation(analyticSolutionFunction[0]);
  Function analyticAnion(analyticSolutionFunction[1]);
  Function analyticPotential(analyticSolutionFunction[2]);
  analyticCationExpression cationExpression;
  analyticAnionExpression anionExpression;
  analyticPotentialExpression potentialExpression;
  analyticCation.interpolate(cationExpression);
  analyticAnion.interpolate(anionExpression);
  analyticPotential.interpolate(potentialExpression);
  L_pnp.cation = analyticCation;
  L_pnp.anion = analyticAnion;
  L_pnp.potential = analyticPotential;


  // Set Dirichlet boundaries
  if (DEBUG) printf("\tboundary conditions...\n");
  unsigned int dirichlet_coord = 0;
  Constant zero_vec(0.0, 0.0, 0.0);
  SymmBoundaries boundary(dirichlet_coord, -domain_par.length_x/2.0, domain_par.length_x/2.0);
  dolfin::DirichletBC bc(V, zero_vec, boundary);

  // Initialize analytic expressions
  if (DEBUG) printf("\tanalytic expressions...\n");
  LogCharge Cation(lower_cation_val, upper_cation_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);
  LogCharge Anion(lower_anion_val, upper_anion_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);
  Voltage Volt(lower_potential_val, upper_potential_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);

  // Interpolate analytic expressions
  Function solutionFunction(V);
  Function cationSolution(solutionFunction[0]);
  cationSolution.interpolate(Cation);
  Function anionSolution(solutionFunction[1]);
  anionSolution.interpolate(Anion);
  ivector cation_dofs;
  ivector anion_dofs;
  ivector potential_dofs;
  get_dofs(&solutionFunction, &cation_dofs, 0);
  get_dofs(&solutionFunction, &anion_dofs, 1);
  get_dofs(&solutionFunction, &potential_dofs, 2);

  // Solve for consistent voltage : not yet implemented
  Function potentialSolution(solutionFunction[2]);
  potentialSolution.interpolate(Volt);

  // Interpolate analytic expressions for EAFE
  Function CatCatFunction(V_cat);
  CatCatFunction.interpolate(Cation);
  Function CatBetaFunction(V_cat);
  CatBetaFunction.interpolate(Volt);
  *(CatBetaFunction.vector())*=coeff_par.cation_mobility;
  *(CatBetaFunction.vector())+=*(CatCatFunction.vector());
  Function AnAnFunction(V_an);
  AnAnFunction.interpolate(Anion);
  Function AnBetaFunction(V_an);
  AnBetaFunction.interpolate(Volt);
  *(AnBetaFunction.vector())*=coeff_par.anion_mobility;
  *(AnBetaFunction.vector())+=*(AnAnFunction.vector());

  // initialize linear system
  if (DEBUG) printf("\tlinear algebraic objects...\n");
  EigenMatrix A_pnp;
  EigenMatrix A_cat;
  EigenMatrix A_an;
  EigenVector b_pnp;
  dCSRmat A_fasp;
  dBSRmat A_fasp_bsr;
  dvector b_fasp, solu_fasp;

  // Setup FASP solver
  if (DEBUG) printf("\tsetup FASP solver...\n");
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/linearized_pnp_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  //*************************************************************
  //  Initialize Newton solver
  //*************************************************************
  // Setup newton parameters and compute initial residual
  if (DEBUG) printf("\tnewton solver setup...\n");
  Function solutionUpdate(V);
  unsigned int newton_iteration = 0;

  // compute initial residual and Jacobian
  if (DEBUG) printf("\tconstruct linear system...\n");
  a_pnp.CatCat = cationSolution;
  a_pnp.AnAn = anionSolution;
  a_pnp.EsEs = potentialSolution;
  a_cat.eta = CatCatFunction;
  a_cat.beta = CatBetaFunction;
  a_an.eta = AnAnFunction;
  a_an.beta = AnBetaFunction;
  assemble(A_pnp, a_pnp);
  assemble(A_cat, a_cat);
  assemble(A_an, a_an);
  replace_matrix(3,0, &V, &V_cat, &A_pnp, &A_cat);
  // replace_matrix(3,1, &V, &V_an , &A_pnp, &A_an );
  bc.apply(A_pnp);

  L_pnp.CatCat = cationSolution;
  L_pnp.AnAn = anionSolution;
  L_pnp.EsEs = potentialSolution;
  assemble(b_pnp, L_pnp);
  bc.apply(b_pnp);
  double initial_residual = b_pnp.norm("l2");
  double relative_residual = 1.0;
  if (DEBUG) printf("\tinitial nonlinear residual has l2-norm of %e\n", initial_residual);


  if (DEBUG) printf("\tinitialized succesfully!\n\n");



  //*************************************************************
  //  Solve : this will be inside Newton loop
  //*************************************************************
  if (DEBUG) printf("Solve the system\n");
  newton_iteration++;

  // Convert to fasp
  if (DEBUG) printf("\tconvert to FASP and solve...\n");
  EigenVector_to_dvector(&b_pnp,&b_fasp);
  EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
  A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
  fasp_dvec_alloc(b_fasp.row, &solu_fasp);
  fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
  status = fasp_solver_dbsr_krylov_amg(&A_fasp_bsr, &b_fasp, &solu_fasp, &itpar, &amgpar);

  // map solu_fasp into solutionUpdate
  if (DEBUG) printf("\tconvert FASP solution to function...\n");
  copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &cation_dofs, &cation_dofs);
  copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &anion_dofs, &anion_dofs);
  copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &potential_dofs, &potential_dofs);

  // update solution and reset solutionUpdate
  if (DEBUG) printf("\tupdate solution...\n");
  Function update(V);
  dolfin::Function cat(update[0]); cat.interpolate(solutionUpdate[0]);
  dolfin::Function an(update[1]); an.interpolate(solutionUpdate[1]);
  dolfin::Function pot(update[2]); pot.interpolate(solutionUpdate[2]);
  *(cationSolution.vector()) += *(cat.vector());
  *(anionSolution.vector()) += *(an.vector());
  *(potentialSolution.vector()) += *(pot.vector());

  // compute residual
  L_pnp.CatCat = cationSolution;
  L_pnp.AnAn = anionSolution;
  L_pnp.EsEs = potentialSolution;
  assemble(b_pnp, L_pnp);
  bc.apply(b_pnp);
  relative_residual = b_pnp.norm("l2") / initial_residual;
  if (DEBUG) {
    if (newton_iteration == 1)
      printf("\trelative nonlinear residual after 1 iteration has l2-norm of %e\n", relative_residual);
    else
      printf("\trelative nonlinear residual after %d iterations has l2-norm of %e\n", newton_iteration, relative_residual);
    printf("\tsolved successfully!\n");
  }

  // compute solution error
  if (DEBUG) printf("\nCompute the error\n");
  *(cationSolution.vector()) -= *(analyticCation.vector());
  *(anionSolution.vector()) -= *(analyticAnion.vector());
  *(potentialSolution.vector()) -= *(analyticPotential.vector());
  double cationError = cationSolution.vector()->norm("l2");
  double anionError = anionSolution.vector()->norm("l2");
  double potentialError = potentialSolution.vector()->norm("l2");
  if (DEBUG) {
    printf("\tcation l2 error is:     %e\n", cationError);
    printf("\tanion l2 error is:      %e\n", anionError);
    printf("\tpotential l2 error is:  %e\n", potentialError);
  }

  if ( (cationError < 1E-7) && (anionError < 1E-7) && (potentialError < 1E-7) )
    std::cout << "Success... the linearized pnp solver is working\n";
  else {
    printf("***\tERROR IN LINEARIZED PNP SOLVER TEST\n");
    printf("***\n***\n***\n");
    printf("***\tLINEARIZED PNP SOLVER TEST:\n");
    printf("***\tThe computed solution is wrong\n");
    printf("***\n***\n***\n");
    printf("***\tERROR IN LINEARIZED PNP SOLVER TEST\n");
    fflush(stdout);
  }

  // state problem
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of test for LINEARIZED PNP Solver with DEBUG=TRUE   #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;


  return 0;
}
