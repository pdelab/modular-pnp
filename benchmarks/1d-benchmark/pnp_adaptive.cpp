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
#include "pnp.h"
#include "L2Error.h"
#include "energy.h"
#include "newton.h"
#include "newton_functs.h"
#include <ctime>
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
  #define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

bool eafe_switch = false;

double lower_cation_val = 0.1;  // 1 / m^3
double upper_cation_val = 1.0;  // 1 / m^3
double lower_anion_val = 1.0;  // 1 / m^3
double upper_anion_val = 0.1;  // 1 / m^3
double lower_potential_val = -1.0;  // V
double upper_potential_val = 1.0;  // V


double time_step_size = 0.02;
double final_time = 1.0;

double get_initial_residual (
  pnp::LinearForm* L,
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
  pnp::LinearForm* L,
  const dolfin::DirichletBC* bc,
  newton_param* params
);


int main(int argc, char** argv)
{
  if (argc > 1)
    if (std::string(argv[1])=="EAFE")
      eafe_switch = true;

  printf("\n-----------------------------------------------------------    ");
  printf("\n Solving the linearized Poisson-Nernst-Planck system           ");
  printf("\n of a single cation and anion ");
  if (eafe_switch)
    printf("using EAFE approximations \n to the Jacobians");
  printf("\n-----------------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  // File
  std::ofstream ofs;
  ofs.open ("./benchmarks/1d-benchmark/data.txt", std::ofstream::out);
  ofs << "t" << "\t" << "NewtonIteration" << "\t" << "RelativeResidual" << "\t" << "Cation" << "\t" << "Anion" << "\t" << "Potential" << "\t" << "Energy" << "\t"<< "TimeElaspsed" << "\t" << "MeshSize" << "\n";
  ofs.close();

  //*************************************************************
  //  Initialization
  //*************************************************************
  printf("Initialize the problem\n"); fflush(stdout);
  // read domain parameters
  printf("domain...\n"); fflush(stdout);
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/1d-benchmark/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  dolfin::Mesh mesh_adapt;
  dolfin::MeshFunction<std::size_t> subdomains_init;
  dolfin::MeshFunction<std::size_t> surfaces_init;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh_adapt, &subdomains_init, &surfaces_init);
  print_domain_param(&domain_par);

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par;
  char coeff_param_filename[] = "./benchmarks/1d-benchmark/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/1d-benchmark/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  // Setup FASP solver
  printf("FASP solver parameters...\n"); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/1d-benchmark/bsr.dat";
  fasp_param_input(fasp_params, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  // open files for outputting solutions
  File cationFile("./benchmarks/1d-benchmark/output/cation.pvd");
  File anionFile("./benchmarks/1d-benchmark/output/anion.pvd");
  File potentialFile("./benchmarks/1d-benchmark/output/potential.pvd");

  // PREVIOUS ITERATE
  pnp::FunctionSpace V_init(mesh_adapt);
  dolfin::Function initial_soln(V_init);
  dolfin::Function initial_cation(initial_soln[0]);
  dolfin::Function initial_anion(initial_soln[1]);
  dolfin::Function initial_potential(initial_soln[2]);
  unsigned int dirichlet_coord = 0;
  LogCharge Cation(
    lower_cation_val,
    upper_cation_val,
    -domain_par.length_x/2.0,
    domain_par.length_x/2.0,
    dirichlet_coord
  );
  LogCharge Anion(
    lower_anion_val,
    upper_anion_val,
    -domain_par.length_x/2.0,
    domain_par.length_x/2.0,
    dirichlet_coord
  );
  Voltage Volt(
    lower_potential_val,
    upper_potential_val,
    -domain_par.length_x/2.0,
    domain_par.length_x/2.0,
    dirichlet_coord
  );
  initial_cation.interpolate(Cation);
  initial_anion.interpolate(Anion);
  initial_potential.interpolate(Volt);
  
  // output solution after solved for timestep
  cationFile << initial_cation;
  anionFile << initial_anion;
  potentialFile << initial_potential;

  // initialize error
  double cationError = 0.0;
  double anionError = 0.0;
  double potentialError = 0.0;
  double energy = 0.0;

  // Time
  std::clock_t begin = std::clock();
  std::clock_t end;
  double timeElaspsed;

  // Fasp matrices and vectors
  dCSRmat A_fasp;
  dBSRmat A_fasp_bsr;
  dvector b_fasp, solu_fasp;

  // Constants
  Constant eps(coeff_par.relative_permittivity);
  Constant Dp(coeff_par.cation_diffusivity);
  Constant Dn(coeff_par.anion_diffusivity);
  Constant qp(coeff_par.cation_valency);
  Constant qn(coeff_par.anion_valency);
  Constant C_dt(time_step_size);
  Constant cat_alpha(coeff_par.cation_diffusivity*time_step_size);
  Constant an_alpha(coeff_par.anion_diffusivity*time_step_size);
  Constant C1(1.0);
  Constant zero(0.0);

  for (double t = 0; t < final_time; t += time_step_size) {
    // printf("\nSet voltage to %e...\n", volt); fflush(stdout);

    //*************************************************************
    //  Mesh adaptivity
    //*************************************************************


    // set adaptivity parameters
    dolfin::Mesh mesh(mesh_adapt);
    double entropy_tol = 1.0e-5;
    unsigned int num_adapts = 0, max_adapts = 1;
    bool adaptive_convergence = false;

    // initialize storage functions for adaptivity
    printf("store previous solution and initialize solution functions\n"); fflush(stdout);
    pnp::FunctionSpace V_adapt(mesh_adapt);
    dolfin::Function prev_soln_adapt(V_adapt);
    dolfin::Function prev_cation_adapt(prev_soln_adapt[0]);
    dolfin::Function prev_anion_adapt(prev_soln_adapt[1]);
    dolfin::Function prev_potential_adapt(prev_soln_adapt[2]);
    prev_cation_adapt.interpolate(initial_cation);
    prev_anion_adapt.interpolate(initial_anion);
    prev_potential_adapt.interpolate(initial_potential);

    dolfin::Function soln_adapt(V_adapt);
    dolfin::Function cation_adapt(soln_adapt[0]);
    dolfin::Function anion_adapt(soln_adapt[1]);
    dolfin::Function potential_adapt(soln_adapt[2]);
    cation_adapt.interpolate(initial_cation);
    anion_adapt.interpolate(initial_anion);
    potential_adapt.interpolate(initial_potential);

    // adaptivity loop
    printf("Adaptivity loop\n"); fflush(stdout);
    while (!adaptive_convergence)
    {
      // output mesh
      meshOut << mesh;

      // Initialize variational forms
      printf("\tvariational forms...\n"); fflush(stdout);
      pnp::FunctionSpace V(mesh);
      pnp::BilinearForm a_pnp(V,V);
      pnp::LinearForm L_pnp(V);
      a_pnp.eps = eps; L_pnp.eps = eps;
      a_pnp.Dp = Dp; L_pnp.Dp = Dp;
      a_pnp.Dn = Dn; L_pnp.Dn = Dn;
      a_pnp.qp = qp; L_pnp.qp = qp;
      a_pnp.qn = qn; L_pnp.qn = qn;
      a_pnp.dt = C_dt; L_pnp.dt = C_dt;

      // Interpolate previous solutions analytic expressions
      printf("\tinterpolate previous step solution onto new mesh...\n"); fflush(stdout);
      dolfin::Function prev_soln(V);
      dolfin::Function previous_cation(prev_soln[0]);
      previous_cation.interpolate(prev_cation_adapt);
      dolfin::Function previous_anion(prev_soln[1]);
      previous_anion.interpolate(prev_anion_adapt);
      dolfin::Function previous_potential(prev_soln[2]);
      previous_potential.interpolate(prev_potential_adapt);

      printf("\tinterpolate solution onto new mesh...\n"); fflush(stdout);
      dolfin::Function solutionFunction(V);
      dolfin::Function cationSolution(solutionFunction[0]);
      cationSolution.interpolate(cation_adapt);
      dolfin::Function anionSolution(solutionFunction[1]);
      anionSolution.interpolate(anion_adapt);
      dolfin::Function potentialSolution(solutionFunction[2]);
      potentialSolution.interpolate(potential_adapt);

      // Set Dirichlet boundaries
      printf("\tboundary conditions...\n"); fflush(stdout);
      Constant zero_vec(0.0, 0.0, 0.0);
      SymmBoundaries boundary(dirichlet_coord, -domain_par.length_x/2.0, domain_par.length_x/2.0);
      dolfin::DirichletBC bc(V, zero_vec, boundary);

      // map dofs
      ivector cation_dofs;
      ivector anion_dofs;
      ivector potential_dofs;
      get_dofs(&solutionFunction, &cation_dofs, 0);
      get_dofs(&solutionFunction, &anion_dofs, 1);
      get_dofs(&solutionFunction, &potential_dofs, 2);

      //EAFE Formulation
      if (eafe_switch)
        printf("\tEAFE initialization...\n");
      EAFE::FunctionSpace V_cat(mesh);
      EAFE::BilinearForm a_cat(V_cat,V_cat);
      a_cat.alpha = an_alpha;
      a_cat.gamma = C1;
      EAFE::FunctionSpace V_an(mesh);
      EAFE::BilinearForm a_an(V_an,V_an);
      a_an.alpha = cat_alpha;
      a_an.gamma = C1;
      dolfin::Function CatCatFunction(V_cat);
      dolfin::Function CatBetaFunction(V_cat);
      dolfin::Function AnAnFunction(V_an);
      dolfin::Function AnBetaFunction(V_an);

      // initialize linear system
      printf("\tlinear algebraic objects...\n"); fflush(stdout);
      EigenMatrix A_pnp, A_cat, A_an;
      EigenVector b_pnp;

      //*************************************************************
      //  Initialize Newton solver
      //*************************************************************
      // Setup newton parameters and compute initial residual
      printf("\tNewton solver initialization...\n"); fflush(stdout);
      dolfin::Function solutionUpdate(V);
      unsigned int newton_iteration = 0;

      // set initial residual
      printf("\tupdate initial residual...\n"); fflush(stdout);
      initial_residual = get_initial_residual(&L_pnp, &bc, &previous_cation, &previous_anion, &previous_potential);

      printf("\tcompute relative residual...\n"); fflush(stdout);
      L_pnp.CatCat = cationSolution;
      L_pnp.AnAn = anionSolution;
      L_pnp.EsEs = potentialSolution;
      L_pnp.CatCat_t0 = previous_cation;
      L_pnp.AnAn_t0 = previous_anion;
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
        printf("\nNewton iteration: %d at t=%f\n", ++newton_iteration,t); fflush(stdout);

        // Construct stiffness matrix
        printf("\tconstruct stiffness matrix...\n"); fflush(stdout);
        a_pnp.CatCat = cationSolution;
        a_pnp.AnAn = anionSolution;
        a_pnp.EsEs = potentialSolution;
        assemble(A_pnp, a_pnp);

        // EAFE expressions
        if (eafe_switch) {
          printf("\tcompute EAFE expressions...\n");
          CatCatFunction.interpolate(cationSolution);
          CatBetaFunction.interpolate(potentialSolution);
          *(CatBetaFunction.vector()) *= coeff_par.cation_valency;
          *(CatBetaFunction.vector()) += *(CatCatFunction.vector());
          AnAnFunction.interpolate(anionSolution);
          AnBetaFunction.interpolate(potentialSolution);
          *(AnBetaFunction.vector()) *= coeff_par.anion_valency;
          *(AnBetaFunction.vector()) += *(AnAnFunction.vector());

          // Construct EAFE approximations to Jacobian
          printf("\tconstruct EAFE modifications...\n"); fflush(stdout);
          a_cat.eta = CatCatFunction;
          a_cat.beta = CatBetaFunction;
          a_an.eta = AnAnFunction;
          a_an.beta = AnBetaFunction;
          assemble(A_cat, a_cat);
          assemble(A_an, a_an);

          // Modify Jacobian
          printf("\treplace Jacobian with EAFE approximations...\n"); fflush(stdout);
          replace_matrix(3,0, &V, &V_cat, &A_pnp, &A_cat);
          replace_matrix(3,1, &V, &V_an , &A_pnp, &A_an );
        }
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
          printf("\tsolved linear system successfully...\n");

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
        L_pnp.CatCat_t0 = previous_cation;
        L_pnp.AnAn_t0 = previous_cation;
        assemble(b_pnp, L_pnp);
        bc.apply(b_pnp);

        //fasp_dbsr_free(&A_fasp_bsr);

      }

      if (relative_residual < nonlinear_tol)
        printf("\nSuccessfully solved the system below desired residual in %d steps!\n\n", newton_iteration);
      else {
        printf("\nDid not converge in %d Newton iterations at t=%e...\n", max_newton_iters,t);
        printf("\tcurrent relative residual is %e > %e\n\n", relative_residual, nonlinear_tol);
      }

      // compute local entropy and refine mesh
      printf("Computing local entropy for refinement\n");
      unsigned int num_refines;
      num_refines = check_local_entropy (
        &cationSolution,
        coeff_par.cation_valency,
        &anionSolution,
        coeff_par.anion_valency,
        &potentialSolution,
        &mesh_adapt,
        entropy_tol
      );

      // free fasp solution
      fasp_dvec_free(&solu_fasp);

      if ( (num_refines == 0) || ( ++num_adapts > max_adapts ) ){
        // successful solve
          if (num_refines == 0) printf("\tsuccessfully distributed entropy below desired entropy in %d adapts!\n\n", num_adapts);
          else printf("\nDid not adapt mesh to entropy in %d adapts...\n", max_adapts);
          adaptive_convergence = true;
          dolfin::Function Er_cat(previous_cation);
          dolfin::Function Er_an(previous_anion);
          dolfin::Function Er_es(previous_potential);
          *(Er_cat.vector()) -= *(cationSolution.vector());
          *(Er_an.vector()) -= *(anionSolution.vector());
          *(Er_es.vector()) -= *(potentialSolution.vector());
          *(Er_cat.vector()) /= time_step_size;
          *(Er_an.vector()) /= time_step_size;
          *(Er_es.vector()) /= time_step_size;
          L2Error::Form_M L2error1(mesh,Er_cat);
          cationError = assemble(L2error1);
          L2Error::Form_M L2error2(mesh,Er_an);
          anionError = assemble(L2error2);
          L2Error::Form_M L2error3(mesh,Er_es);
          potentialError = assemble(L2error3);
          energy::Form_M EN(mesh,cationSolution,anionSolution,potentialSolution,eps);
          energy = assemble(EN);

          printf("***********************************************\n");
          printf("***********************************************\n");
          printf("Difference at t=%e...\n",t);
          printf("\tcation l2 error is:     %e\n", cationError);
          printf("\tanion l2 error is:      %e\n", anionError);
          printf("\tpotential l2 error is:  %e\n", potentialError);
          printf("\tEnergy is:  %e\n", energy);
          printf("***********************************************\n");
          printf("***********************************************\n");
          end = clock();

          ofs.open("./benchmarks/1d-benchmark/data.txt", std::ofstream::out | std::ofstream::app);
          timeElaspsed = double(end - begin) / CLOCKS_PER_SEC;
          ofs << t << "\t" << newton_iteration << "\t" << relative_residual << "\t" << cationError << "\t" << anionError << "\t" << potentialError << "\t" << energy << "\t"<< timeElaspsed << "\t" << mesh.num_cells() << "\n";
          ofs.close();

          // store solution as solution from previous step
          std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(mesh_adapt) );
          initial_cation = adapt(cationSolution, mesh_ptr);
          initial_anion = adapt(anionSolution, mesh_ptr);
          initial_potential = adapt(potentialSolution, mesh_ptr);

          // to ensure the building_box_tree is correctly indexed
          mesh = mesh_adapt;
          mesh.bounding_box_tree()->build(mesh);
          mesh_adapt.bounding_box_tree()->build(mesh_adapt);

          // output solution after solved for timestep
          cationFile << initial_cation;
          anionFile << initial_anion;
          potentialFile << initial_potential;

        break;
      }

      // adapt solutions to refined mesh
      if (num_refines == 1)
        printf("\tadapting the mesh using one level of local refinement...\n");
      else
        printf("\tadapting the mesh using %d levels of local refinement...\n", num_refines);

      std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(mesh_adapt) );
      cation_adapt = adapt(cationSolution, mesh_ptr);
      anion_adapt = adapt(anionSolution, mesh_ptr);
      potential_adapt = adapt(potentialSolution, mesh_ptr);

      prev_cation_adapt = adapt(previous_cation, mesh_ptr);
      prev_anion_adapt = adapt(previous_anion, mesh_ptr);
      prev_potential_adapt = adapt(previous_potential, mesh_ptr);
      mesh = mesh_adapt;
      mesh.bounding_box_tree()->build(mesh);  // to ensure the building_box_tree is correctly indexed


    }
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
  pnp::LinearForm* L,
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
  dolfin::Constant C_dt(time_step_size);
  L->CatCat = _iterate0;
  L->AnAn = _iterate1;
  L->EsEs = _iterate2;
  EigenVector b;
  assemble(b, *L);
  bc->apply(b);
  double new_relative_residual = b.norm("l2") / initial_residual;

  // backtrack loop
  unsigned int damp_iters = 0;
  printf("\t\trelative residual after damping %d times: %e\n", damp_iters, new_relative_residual);

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
  pnp::LinearForm* L,
  const dolfin::DirichletBC* bc,
  dolfin::Function* cation,
  dolfin::Function* anion,
  dolfin::Function* potential)
{
  pnp::FunctionSpace V( *(cation->function_space()->mesh()) );
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
  L->CatCat_t0 = adapt_cation;
  L->AnAn_t0 = adapt_anion;
  EigenVector b;
  assemble(b, *L);
  bc->apply(b);
  return b.norm("l2");
}
