/*! \file test_pnp_eafe.cpp
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
#include "pnp_and_source.h"
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
bool DEBUG = false;
bool eafe_switch = false;

double lower_cation_val = 0.1;  // 1 / m^3
double upper_cation_val = 1.0;  // 1 / m^3
double lower_anion_val = 1.0;  // 1 / m^3
double upper_anion_val = 0.1;  // 1 / m^3
double lower_potential_val = -1.0;  // V
double upper_potential_val = 1.0;  // V

double update_solution_pnp (
  dolfin::Function* iterate0,
  dolfin::Function* iterate1,
  dolfin::Function* iterate2,
  dolfin::Function* update0,
  dolfin::Function* update1,
  dolfin::Function* update2,
  double relative_residual,
  double initial_residual,
  pnp_and_source::LinearForm* L,
  const dolfin::DirichletBC* bc,
  newton_param* params )
{
  // compute residual
  dolfin::Function _iterate0(*iterate0);
  dolfin::Function _iterate1(*iterate1);
  dolfin::Function _iterate2(*iterate2);
  dolfin::Function _update0(*update0);
  dolfin::Function _update1(*update1);
  dolfin::Function _update2(*update2);
  update_solution(&_iterate0, &_update0);
  update_solution(&_iterate1, &_update1);
  update_solution(&_iterate2, &_update2);
  L->CatCat = _iterate0;
  L->AnAn = _iterate1;
  L->EsEs = _iterate2;
  dolfin::EigenVector b;
  assemble(b, *L);
  bc->apply(b);
  double new_relative_residual = b.norm("l2") / initial_residual;

  // backtrack loop
  unsigned int damp_iters = 1;
  if (DEBUG) printf("\t\trel_res after damping %d times: %e\n", damp_iters, new_relative_residual);

  while (
    new_relative_residual > relative_residual && damp_iters < params->damp_it )
  {
    damp_iters++;
    *(_iterate0.vector()) = *(iterate0->vector());
    *(_iterate1.vector()) = *(iterate1->vector());
    *(_iterate2.vector()) = *(iterate2->vector());
    *(_update0.vector()) *= params->damp_factor;
    *(_update1.vector()) *= params->damp_factor;
    *(_update2.vector()) *= params->damp_factor;
    update_solution(&_iterate0, &_update0);
    update_solution(&_iterate1, &_update1);
    update_solution(&_iterate2, &_update2);
    L->CatCat = _iterate0;
    L->AnAn = _iterate1;
    L->EsEs = _iterate2;
    assemble(b, *L);
    bc->apply(b);
    new_relative_residual = b.norm("l2") / initial_residual;
    if (DEBUG) printf("\t\trel_res after damping %d times: %e\n", damp_iters, new_relative_residual);
  }

  // check for decrease
  if ( new_relative_residual > relative_residual )
    return -new_relative_residual;

  // update iterates
  if (DEBUG) printf("\taccepted update after damping %d times\n", damp_iters);
  *(iterate0->vector()) = *(_iterate0.vector());
  *(iterate1->vector()) = *(_iterate1.vector());
  *(iterate2->vector()) = *(_iterate2.vector());
  return new_relative_residual;
}

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
    values[0]  =std::log(values[0]);
  }
};

class analyticPotentialExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_potential_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_potential_val * (x[0] + 5.0) / 10.0;
    values[0] += -20.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
  }
};


int main(int argc, char** argv)
{
  if (argc == 2) {
    if (std::string(argv[1])=="EAFE")
      eafe_switch = true;
    else if (std::string(argv[1])=="DEBUG")
      DEBUG = true;
  }
  else if (argc > 2)
  {
    if (std::string(argv[1])=="EAFE" || std::string(argv[2])=="EAFE")
      eafe_switch = true;
    if (std::string(argv[1])=="DEBUG" || std::string(argv[2])=="DEBUG")
      DEBUG = true;
  }

  // state problem
  if (DEBUG && eafe_switch) {
    std::cout << "################################################################# \n";
    std::cout << "#### Test of nonlinear PNP/EAFE DEBUG=TRUE                   #### \n";
    std::cout << "################################################################# \n";
  }
  else if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### Test of nonlinear PNP DEBUG=TRUE                        #### \n";
    std::cout << "################################################################# \n";
  }

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  //*************************************************************
  //  Initialization
  //*************************************************************
  if (DEBUG) printf("Initialize the problem\n"); fflush(stdout);
  // read domain parameters
  if (DEBUG) printf("\tdomain...\n"); fflush(stdout);
  domain_param domain_par;
  char domain_param_filename[] = "./tests/pnp_tests/domain_params2.dat";
  domain_param_input(domain_param_filename, &domain_par);
  // print_domain_param(&domain_par);

  // build mesh
  if (DEBUG) printf("\tmesh...\n"); fflush(stdout);
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces);

  // read coefficients and boundary values
  if (DEBUG) printf("\tcoefficients...\n"); fflush(stdout);
  coeff_param coeff_par;
  char coeff_param_filename[] = "./tests/pnp_tests/coeff_params2.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);

  // Initialize variational forms
  if (DEBUG) printf("\tvariational forms...\n"); fflush(stdout);
  pnp_and_source::FunctionSpace V(mesh);
  pnp_and_source::BilinearForm a_pnp(V,V);
  pnp_and_source::LinearForm L_pnp(V);
  Constant eps(coeff_par.relative_permittivity);
  Constant Dp(coeff_par.cation_diffusivity);
  Constant Dn(coeff_par.anion_diffusivity);
  Constant qp(coeff_par.cation_valency);
  Constant qn(coeff_par.anion_valency);
  Constant zero(0.0);
  a_pnp.eps = eps; L_pnp.eps = eps;
  a_pnp.Dp = Dp; L_pnp.Dp = Dp;
  a_pnp.Dn = Dn; L_pnp.Dn = Dn;
  a_pnp.qp = qp; L_pnp.qp = qp;
  a_pnp.qn = qn; L_pnp.qn = qn;

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
  if (DEBUG) printf("\tboundary conditions...\n"); fflush(stdout);
  unsigned int dirichlet_coord = 0;
  Constant zero_vec(0.0, 0.0, 0.0);
  SymmBoundaries boundary(dirichlet_coord, -domain_par.length_x/2.0, domain_par.length_x/2.0);
  dolfin::DirichletBC bc(V, zero_vec, boundary);

  // Initialize analytic expressions
  if (DEBUG) printf("\tanalytic expressions...\n"); fflush(stdout);
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

  //EAFE Formulation
  if (DEBUG && eafe_switch)
    printf("\tinitialize EAFE forms...\n");
  EAFE::FunctionSpace V_cat(mesh);
  EAFE::BilinearForm a_cat(V_cat,V_cat);
  a_cat.alpha = Dp;
  a_cat.gamma = zero;
  EAFE::FunctionSpace V_an(mesh);
  EAFE::BilinearForm a_an(V_an,V_an);
  a_an.alpha = Dn;
  a_an.gamma = zero;

  // Initialize functions for EAFE
  if (DEBUG && eafe_switch)
    printf("\tinitialize EAFE functions...\n");
  Function CatCatFunction(V_cat);
  Function CatBetaFunction(V_cat);
  Function AnAnFunction(V_an);
  Function AnBetaFunction(V_an);

  // initialize linear system
  if (DEBUG) printf("\tlinear algebraic objects...\n"); fflush(stdout);
  EigenMatrix A_pnp, A_cat, A_an;
  EigenVector b_pnp;
  dCSRmat A_fasp;
  dBSRmat A_fasp_bsr;
  dvector b_fasp, solu_fasp;

  // Setup FASP solver
  if (DEBUG) printf("\tsetup FASP solver...\n"); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/pnp_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  //*************************************************************
  //  Initialize Newton solver
  //*************************************************************
  // Setup newton parameters and compute initial residual
  if (DEBUG) printf("\tNewton solver setup...\n"); fflush(stdout);
  Function solutionUpdate(V);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/PNP/newton_param.dat";
  newton_param_input (newton_param_file,&newtparam);
  if (DEBUG) print_newton_param(&newtparam);
  unsigned int newton_iteration = 0;

  // compute initial residual and Jacobian
  if (DEBUG) printf("\tconstruct residual...\n"); fflush(stdout);
  L_pnp.CatCat = cationSolution;
  L_pnp.AnAn = anionSolution;
  L_pnp.EsEs = potentialSolution;
  assemble(b_pnp, L_pnp);
  bc.apply(b_pnp);
  double initial_residual = b_pnp.norm("l2");
  double relative_residual = 1.0;
  if (DEBUG) printf("\tinitial nonlinear residual has l2-norm of %e\n", initial_residual);

  fasp_dvec_alloc(b_pnp.size(), &solu_fasp);
  if (DEBUG) printf("\tinitialized succesfully!\n\n"); fflush(stdout);

  //*************************************************************
  //  Newton solver
  //*************************************************************
  if (DEBUG) printf("solve the nonlinear system\n"); fflush(stdout);
  double cationError = 0.0;
  double anionError = 0.0;
  double potentialError = 0.0;

  double nonlinear_tol = newtparam.tol;
  unsigned int max_newton_iters = newtparam.max_it;
  while (relative_residual > nonlinear_tol && newton_iteration < max_newton_iters)
  {
    newton_iteration++;
    if (DEBUG) printf("\nNewton iteration: %d\n", newton_iteration); fflush(stdout);

    // Construct stiffness matrix
    if (DEBUG) printf("\tconstruct stiffness matrix...\n"); fflush(stdout);
    a_pnp.CatCat = cationSolution;
    a_pnp.AnAn = anionSolution;
    a_pnp.EsEs = potentialSolution;
    assemble(A_pnp, a_pnp);

    // EAFE expressions
    if (eafe_switch) {
      if (DEBUG) printf("\tcompute EAFE expressions...\n");
      CatCatFunction.interpolate(cationSolution);
      CatBetaFunction.interpolate(potentialSolution);
      *(CatBetaFunction.vector()) *= coeff_par.cation_valency;
      *(CatBetaFunction.vector()) += *(CatCatFunction.vector());
      AnAnFunction.interpolate(anionSolution);
      AnBetaFunction.interpolate(potentialSolution);
      *(AnBetaFunction.vector()) *= coeff_par.anion_valency;
      *(AnBetaFunction.vector()) += *(AnAnFunction.vector());

      // Construct EAFE approximations to Jacobian
      if (DEBUG) printf("\tcompute EAFE approximation...\n");
      a_cat.eta = CatCatFunction;
      a_cat.beta = CatBetaFunction;
      a_an.eta = AnAnFunction;
      a_an.beta = AnBetaFunction;
      assemble(A_cat, a_cat);
      assemble(A_an, a_an);

      // Modify Jacobian
      if (DEBUG) printf("\treplace Jacobian with EAFE expressions...\n");
      replace_matrix(3,0, &V, &V_cat, &A_pnp, &A_cat);
      replace_matrix(3,1, &V, &V_an , &A_pnp, &A_an );
    }
    bc.apply(A_pnp);

    // Convert to fasp
    if (DEBUG) printf("\tconvert to FASP and solve...\n"); fflush(stdout);
    EigenVector_to_dvector(&b_pnp,&b_fasp);
    EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
    A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
    fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
    status = fasp_solver_dbsr_krylov_amg(&A_fasp_bsr, &b_fasp, &solu_fasp, &itpar, &amgpar);

    // map solu_fasp into solutionUpdate
    if (DEBUG) printf("\tconvert FASP solution to function...\n"); fflush(stdout);
    copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &cation_dofs, &cation_dofs);
    copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &anion_dofs, &anion_dofs);
    copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &potential_dofs, &potential_dofs);

    // update solution using backtracking and reset solutionUpdate
    if (DEBUG) printf("\tupdate solution...\n"); fflush(stdout);
    relative_residual = update_solution_pnp(
      &cationSolution,
      &anionSolution,
      &potentialSolution,
      &(solutionUpdate[0]),
      &(solutionUpdate[1]),
      &(solutionUpdate[2]),
      relative_residual,
      initial_residual,
      &L_pnp,
      &bc,
      &newtparam
    );
    if (relative_residual < 0.0) {
      printf("Newton backtracking failed!\n");
      printf("\tresidual has not decreased after damping %d times\n", newtparam.damp_it);
      printf("\tthe relative residual is %e\n", relative_residual);
      relative_residual *= -1.0;
    }

    // compute residual
    L_pnp.CatCat = cationSolution;
    L_pnp.AnAn = anionSolution;
    L_pnp.EsEs = potentialSolution;
    assemble(b_pnp, L_pnp);
    bc.apply(b_pnp);
    relative_residual = b_pnp.norm("l2");/// initial_residual;
    if (DEBUG) {
      if (newton_iteration == 1)
        printf("\trelative nonlinear residual after 1 iteration has l2-norm of %e\n", relative_residual);
      else
        printf("\trelative nonlinear residual after %d iterations has l2-norm of %e\n", newton_iteration, relative_residual);
    }

    // compute solution error
    if (DEBUG) printf("\nCompute the error\n"); fflush(stdout);
    Function Error1(analyticCation);
    Function Error2(analyticAnion);
    Function Error3(analyticPotential);
    *(Error1.vector()) -= *(cationSolution.vector());
    *(Error2.vector()) -= *(anionSolution.vector());
    *(Error3.vector()) -= *(potentialSolution.vector());
    L2Error::Form_M L2error1(mesh,Error1);
    cationError = assemble(L2error1);
    L2Error::Form_M L2error2(mesh,Error2);
    anionError = assemble(L2error2);
    L2Error::Form_M L2error3(mesh,Error3);
    potentialError = assemble(L2error3);
    if (DEBUG) {
      printf("\tcation l2 error is:     %e\n", cationError);
      printf("\tanion l2 error is:      %e\n", anionError);
      printf("\tpotential l2 error is:  %e\n\n", potentialError);
    }

  }

  if ( (relative_residual < nonlinear_tol) && (anionError < 1E-16) && (cationError < 1E-16) && (potentialError < 1E-16) && (newton_iteration <max_newton_iters)) {
    if (eafe_switch) printf("Success... solved the nonlinear PNP/EAFE in %d steps!\n", newton_iteration);
    else printf("Success... solved the nonlinear PNP in %d steps!\n", newton_iteration);
  }
  else {
    if (eafe_switch)
      printf("***\tERROR IN PNP/EAFE SOLVER TEST\n");
    else
      printf("***\tERROR IN PNP SOLVER TEST\n");
    printf("***\n***\n***\n");
    printf("***\tDid not converge in %d Newton iterations...\n", max_newton_iters);
    printf("***\tcurrent relative residual is %e > %e\n", relative_residual, nonlinear_tol);
    printf("***\n***\n***\n");
    if (eafe_switch)
      printf("***\tERROR IN PNP/EAFE SOLVER TEST\n");
    else
      printf("***\tERROR IN PNP SOLVER TEST\n");
    fflush(stdout);
  }

  // state problem
  if (DEBUG && eafe_switch) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of Test of nonlinear PNP/EAFE DEBUG=TRUE            #### \n";
    std::cout << "################################################################# \n";
  }
  else if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of Test of nonlinear PNP DEBUG=TRUE                 #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;
}
