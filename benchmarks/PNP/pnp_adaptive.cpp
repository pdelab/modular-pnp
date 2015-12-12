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
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "pnp_and_source.h"
#include "newton.h"
#include "newton_functs.h"
#include "L2Error.h"
#include "L2Error_3.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
  #define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

double lower_cation_val = 0.1;  // 1 / m^3
double upper_cation_val = 1.0;  // 1 / m^3
double lower_anion_val = 1.0;  // 1 / m^3
double upper_anion_val = 0.1;  // 1 / m^3
double lower_potential_val = -1.0;  // V
double upper_potential_val = 1.0;  // V

double get_initial_residual (
  pnp_and_source::LinearForm* L,
  const dolfin::DirichletBC* bc,
  dolfin::Function* cation,
  dolfin::Function* anion,
  dolfin::Function* potential
);

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
  newton_param* params
);

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
    values[0] += -20.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
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
  printf("Initialize the problem\n"); fflush(stdout);
  // read domain parameters
  printf("domain...\n"); fflush(stdout);
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/PNP/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  dolfin::Mesh mesh0, mesh_init;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh0, &subdomains, &surfaces);
  domain_build(&domain_par, &mesh_init, &subdomains, &surfaces);
  print_domain_param(&domain_par);

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par;
  char coeff_param_filename[] = "./benchmarks/PNP/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/PNP/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  // Setup FASP solver
  printf("FASP solver parameters...\n"); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/PNP/bsr.dat";
  fasp_param_input(fasp_params, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  // open files for outputting solutions
  File cationFile("./benchmarks/PNP/output/cation.pvd");
  File anionFile("./benchmarks/PNP/output/anion.pvd");
  File potentialFile("./benchmarks/PNP/output/potential.pvd");

  // Initialize guess
  printf("intial guess...\n"); fflush(stdout);
  unsigned int dirichlet_coord = 0;
  LogCharge Cation(lower_cation_val, upper_cation_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);
  LogCharge Anion(lower_anion_val, upper_anion_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);

  // not ideal implementation: replace by a solve for voltage below
  Voltage Volt(lower_potential_val, upper_potential_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);

  // interpolate
  pnp_and_source::FunctionSpace V_init(mesh_init);
  dolfin::Function initialGuessFunction(V_init);
  dolfin::Function initialCation(initialGuessFunction[0]);
  dolfin::Function initialAnion(initialGuessFunction[1]);
  dolfin::Function initialPotential(initialGuessFunction[2]);
  initialCation.interpolate(Cation);
  initialAnion.interpolate(Anion);
  initialPotential.interpolate(Volt);

  //*************************************************************
  //  Mesh adaptivity
  //*************************************************************
  // interpolate analytic expressions
  printf("interpolate analytic expressions onto initial mesh...\n\n"); fflush(stdout);
  pnp_and_source::FunctionSpace V0(mesh0);
  dolfin::Function initialGuessFunction0(V0);
  dolfin::Function solutionFunction0(V0);
  analyticCationExpression cationExpression0;
  analyticAnionExpression anionExpression0;
  analyticPotentialExpression potentialExpression0;
  dolfin::Function cation0(solutionFunction0[0]);
  dolfin::Function anion0(solutionFunction0[1]);
  dolfin::Function potential0(solutionFunction0[2]);
  cation0.interpolate(Cation);
  anion0.interpolate(Anion);
  potential0.interpolate(Volt);
  // cation0.interpolate(cationExpression0);
  // anion0.interpolate(anionExpression0);
  // potential0.interpolate(potentialExpression0);

  // set adaptivity parameters
  dolfin::Mesh mesh(mesh0);
  double entropy_tol = 1.0e-4;
  unsigned int num_adapts = 0, max_adapts = 5;
  bool adaptive_convergence = false;

  printf("Adaptivity loop\n"); fflush(stdout);

  // adaptivity loop
  while (!adaptive_convergence)
  {
    // output mesh
    meshOut << mesh;

    // Initialize variational forms
    printf("\tvariational forms...\n"); fflush(stdout);
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

    //EAFE Formulation
    EAFE::FunctionSpace V_cat(mesh);
    EAFE::BilinearForm a_cat(V_cat,V_cat);
    a_cat.alpha = Dp;
    a_cat.gamma = zero;
    EAFE::FunctionSpace V_an(mesh);
    EAFE::BilinearForm a_an(V_an,V_an);
    a_an.alpha = Dn;
    a_an.gamma = zero;

    // analytic solution
    Function analyticSolutionFunction(V);
    Function analyticCation(analyticSolutionFunction[0]);
    Function analyticAnion(analyticSolutionFunction[1]);
    Function analyticPotential(analyticSolutionFunction[2]);
    analyticCationExpression cationExpression;
    analyticAnionExpression anionExpression;
    analyticPotentialExpression potentialExpression;
    // analyticCation.interpolate(cationExpression);
    // analyticAnion.interpolate(anionExpression);
    // analyticPotential.interpolate(potentialExpression);
    analyticCation.interpolate(zero);
    analyticAnion.interpolate(zero);
    analyticPotential.interpolate(zero);
    L_pnp.cation = analyticCation;
    L_pnp.anion = analyticAnion;
    L_pnp.potential = analyticPotential;

    File EXcationFile("./benchmarks/PNP/output/Ex_cation.pvd");
    File EXanionFile("./benchmarks/PNP/output/Ex_anion.pvd");
    File EXpotentialFile("./benchmarks/PNP/output/Ex_potential.pvd");
    EXcationFile << analyticCation;
    EXanionFile << analyticAnion;
    EXpotentialFile << analyticPotential;

    // Set Dirichlet boundaries
    printf("\tboundary conditions...\n"); fflush(stdout);
    Constant zero_vec(0.0, 0.0, 0.0);
    SymmBoundaries boundary(dirichlet_coord, -domain_par.length_x/2.0, domain_par.length_x/2.0);
    dolfin::DirichletBC bc(V, zero_vec, boundary);

    // Interpolate analytic expressions
    printf("\tinterpolate solution onto new mesh...\n"); fflush(stdout);
    dolfin::Function solutionFunction(V);
    dolfin::Function cationSolution(solutionFunction[0]);
    cationSolution.interpolate(cation0);
    dolfin::Function anionSolution(solutionFunction[1]);
    anionSolution.interpolate(anion0);

    // solve for voltage
    dolfin::Function potentialSolution(solutionFunction[2]);
    potentialSolution.interpolate(potential0);

    // write computed solution to file
    printf("\toutput projected solution to file\n"); fflush(stdout);
    cationFile << cationSolution;
    anionFile << anionSolution;
    potentialFile << potentialSolution;

    // map dofs
    ivector cation_dofs;
    ivector anion_dofs;
    ivector potential_dofs;
    get_dofs(&solutionFunction, &cation_dofs, 0);
    get_dofs(&solutionFunction, &anion_dofs, 1);
    get_dofs(&solutionFunction, &potential_dofs, 2);

    // Initialize functions for EAFE
    dolfin::Function CatCatFunction(V_cat);
    dolfin::Function CatBetaFunction(V_cat);
    dolfin::Function AnAnFunction(V_an);
    dolfin::Function AnBetaFunction(V_an);

    // initialize linear system
    printf("\tlinear algebraic objects...\n"); fflush(stdout);
    EigenMatrix A_pnp, A_cat, A_an;
    EigenVector b_pnp;
    dCSRmat A_fasp;
    dBSRmat A_fasp_bsr;
    dvector b_fasp, solu_fasp;

    //*************************************************************
    //  Initialize Newton solver
    //*************************************************************
    // Setup newton parameters and compute initial residual
    printf("\tNewton solver initialization...\n"); fflush(stdout);
    dolfin::Function solutionUpdate(V);
    unsigned int newton_iteration = 0;

    // set initial residual
    printf("\tupdate initial residual...\n"); fflush(stdout);
    initial_residual = get_initial_residual(&L_pnp, &bc, &initialCation, &initialAnion, &initialPotential);

    printf("\tcompute relative residual...\n"); fflush(stdout);
    L_pnp.CatCat = cationSolution;
    L_pnp.AnAn = anionSolution;
    L_pnp.EsEs = potentialSolution;
    assemble(b_pnp, L_pnp);
    bc.apply(b_pnp);
    relative_residual = b_pnp.norm("l2") / initial_residual;
    if (num_adapts == 0)
      printf("\tinitial nonlinear residual has l2-norm of %e\n", initial_residual);
    else
      printf("\tadapted relative nonlinear residual is %e\n", relative_residual);

    fasp_dvec_alloc(b_pnp.size(), &solu_fasp);
    printf("\tinitialized succesfully...\n\n"); fflush(stdout);

    //*************************************************************
    //  Newton solver
    //*************************************************************
    printf("Solve the nonlinear system\n"); fflush(stdout);

    double nonlinear_tol = newtparam.tol;
    unsigned int max_newton_iters = newtparam.max_it;
    while (relative_residual > nonlinear_tol && newton_iteration < max_newton_iters)
    {
      printf("\nNewton iteration: %d\n", ++newton_iteration); fflush(stdout);

      // Construct stiffness matrix
      printf("\tconstruct stiffness matrix...\n"); fflush(stdout);
      a_pnp.CatCat = cationSolution;
      a_pnp.AnAn = anionSolution;
      a_pnp.EsEs = potentialSolution;
      assemble(A_pnp, a_pnp);

      // EAFE expressions
      CatCatFunction.interpolate(cationSolution);
      CatBetaFunction.interpolate(potentialSolution);
      *(CatBetaFunction.vector()) *= coeff_par.cation_valency;
      *(CatBetaFunction.vector()) += *(CatCatFunction.vector());
      AnAnFunction.interpolate(anionSolution);
      AnBetaFunction.interpolate(potentialSolution);
      *(AnBetaFunction.vector()) *= coeff_par.anion_valency;
      *(AnBetaFunction.vector()) += *(AnAnFunction.vector());

      // Construct EAFE approximations to Jacobian
      a_cat.eta = CatCatFunction;
      a_cat.beta = CatBetaFunction;
      a_an.eta = AnAnFunction;
      a_an.beta = AnBetaFunction;
      assemble(A_cat, a_cat);
      assemble(A_an, a_an);

      // Modify Jacobian
      replace_matrix(3,0, &V, &V_cat, &A_pnp, &A_cat);
      replace_matrix(3,1, &V, &V_an , &A_pnp, &A_an );
      bc.apply(A_pnp);

      // Convert to fasp
      printf("\tconvert to FASP and solve...\n"); fflush(stdout);
      EigenVector_to_dvector(&b_pnp,&b_fasp);
      EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
      A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
      fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
      status = fasp_solver_dbsr_krylov_amg(&A_fasp_bsr, &b_fasp, &solu_fasp, &itpar, &amgpar);
      if (status < 0)
        printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", status);
      else
        printf("\tsolved linear system successfully!\n");

      // map solu_fasp into solutionUpdate
      printf("\tconvert FASP solution to function...\n"); fflush(stdout);
      copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &cation_dofs, &cation_dofs);
      copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &anion_dofs, &anion_dofs);
      copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &potential_dofs, &potential_dofs);

      // update solution and reset solutionUpdate
      printf("\tupdate solution...\n"); fflush(stdout);
      relative_residual = update_solution_pnp (
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

      // update nonlinear residual
      L_pnp.CatCat = cationSolution;
      L_pnp.AnAn = anionSolution;
      L_pnp.EsEs = potentialSolution;
      assemble(b_pnp, L_pnp);
      bc.apply(b_pnp);
      // if (newton_iteration == 1)
      //   printf("\trelative nonlinear residual after 1 iteration has l2-norm of %e\n", relative_residual);
      // else
      //   printf("\trelative nonlinear residual after %d iterations has l2-norm of %e\n",
      //     newton_iteration,
      //     relative_residual
      //   );

      // write computed solution to file
      // printf("\toutput computed solution to file\n"); fflush(stdout);
      // cationFile << cationSolution;
      // anionFile << anionSolution;
      // potentialFile << potentialSolution;

      // compute solution error
      // printf("\nCompute the error\n"); fflush(stdout);
      // dolfin::Function Error1(analyticCation);
      // dolfin::Function Error2(analyticAnion);
      // dolfin::Function Error3(analyticPotential);
      // *(Error1.vector()) -= *(cationSolution.vector());
      // *(Error2.vector()) -= *(anionSolution.vector());
      // *(Error3.vector()) -= *(potentialSolution.vector());
      // double cationError = 0.0;
      // double anionError = 0.0;
      // double potentialError = 0.0;
      // L2Error::Form_M L2error1(mesh,Error1);
      // cationError = assemble(L2error1);
      // L2Error::Form_M L2error2(mesh,Error2);
      // anionError = assemble(L2error2);
      // L2Error::Form_M L2error3(mesh,Error3);
      // potentialError = assemble(L2error3);
      // printf("\tcation l2 error is:     %e\n", cationError);
      // printf("\tanion l2 error is:      %e\n", anionError);
      // printf("\tpotential l2 error is:  %e\n", potentialError);

      // print error
      // cationFile << cationSolution;
      // anionFile << anionSolution;
      // potentialFile << potentialSolution;
    }

    if (relative_residual < nonlinear_tol)
      printf("\nSuccessfully solved the system below desired residual in %d steps!\n\n", newton_iteration);
    else {
      printf("\nDid not converge in %d Newton iterations...\n", max_newton_iters);
      printf("\tcurrent relative residual is %e > %e\n\n", relative_residual, nonlinear_tol);
    }

    cationFile << cationSolution;
    anionFile << anionSolution;
    potentialFile << potentialSolution;

    // compute local entropy and refine mesh
    printf("Computing local entropy for refinement\n");
    unsigned int num_refines;
    num_refines = check_local_entropy (
      &cationSolution,
      coeff_par.cation_valency,
      &anionSolution,
      coeff_par.anion_valency,
      &potentialSolution,
      &mesh0,
      entropy_tol
    );

    if (num_refines == 0) {
      // successful solve
      printf("\n\nSuccessfully distributed entropy below desired entropy in %d adapts!\n\n", num_adapts);
      adaptive_convergence = true;
      break;
    }
    else if ( ++num_adapts > max_adapts ) {
      // failed adaptivity
      printf("\nDid not adapt mesh to entropy in %d adapts...\n", max_adapts);
      adaptive_convergence = true;
      break;
    }

    // adapt solutions to refined mesh
    if (num_refines == 1)
      printf("\tadapting the mesh using one level of local refinement...\n");
    else
      printf("\tadapting the mesh using %d levels of local refinement...\n", num_refines);

    std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(mesh0) );
    cation0 = adapt(cationSolution, mesh_ptr);
    anion0 = adapt(anionSolution, mesh_ptr);
    potential0 = adapt(potentialSolution, mesh_ptr);
    mesh = mesh0;
    mesh.bounding_box_tree()->build(mesh); // to ensure the building_box_tree is correctly indexed

  }

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}

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
  EigenVector b;
  assemble(b, *L);
  bc->apply(b);
  double new_relative_residual = b.norm("l2") / initial_residual;

  // backtrack loop
  unsigned int damp_iters = 0;
  printf("\t\trel_res after damping %d times: %e\n", damp_iters, new_relative_residual);

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
    printf("\t\trel_res after damping %d times: %e\n", damp_iters, new_relative_residual);
  }

  // check for decrease
  if ( new_relative_residual > relative_residual )
    return -new_relative_residual;

  // update iterates
  *(iterate0->vector()) = *(_iterate0.vector());
  *(iterate1->vector()) = *(_iterate1.vector());
  *(iterate2->vector()) = *(_iterate2.vector());
  return new_relative_residual;
}

double get_initial_residual (
  pnp_and_source::LinearForm* L,
  const dolfin::DirichletBC* bc,
  dolfin::Function* cation,
  dolfin::Function* anion,
  dolfin::Function* potential)
{
  pnp_and_source::FunctionSpace V( *(cation->function_space()->mesh()) );
  dolfin::Function adapt_func(V);
  dolfin::Function adapt_cation(adapt_func[0]);
  dolfin::Function adapt_anion(adapt_func[1]);
  dolfin::Function adapt_potential(adapt_func[2]);
  adapt_cation.interpolate(*cation);
  adapt_anion.interpolate(*anion);
  adapt_potential.interpolate(*potential);
  L->CatCat = adapt_cation;
  L->AnAn = adapt_anion;
  L->EsEs = adapt_potential;
  EigenVector b;
  assemble(b, *L);
  bc->apply(b);
  return b.norm("l2");
}
