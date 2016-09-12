/*! \file pnp_adaptive.cpp
 *
 *  \brief Setup and solve the PNP equations using adaptive meshing and FASP
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

bool eafe_switch = false;

double get_initial_residual (
  pnp::LinearForm* L,
  const dolfin::DirichletBC* bc,
  std::shared_ptr<dolfin::Function> cation,
  std::shared_ptr<dolfin::Function> anion,
  std::shared_ptr<dolfin::Function> potential
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

class PeriodicBoundary : public SubDomain
{
  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (
      x[0] < Lx / 2.0 + 5.0 * DOLFIN_EPS
      || (x[1]) < Ly / 2.0 + 5.0 * DOLFIN_EPS
      || (x[2]) < Lz / 2.0 + 5.0 * DOLFIN_EPS
    // return on_boundary;
    );
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {
      if (std::abs(x[0]) < Lx / 2.0 + 5.0 * DOLFIN_EPS){
          y[0] = -x[0];
          y[1] = x[1];
          y[2] = x[2];
        }
      else if (std::abs(x[1]) < Ly / 2.0 + 5.0 * DOLFIN_EPS){
        y[0] = x[0];
        y[1] = -x[1];
        y[2] = x[2];
      }
      else (std::abs(x[2]) < Lz / 2.0 + 5.0 * DOLFIN_EPS){
          y[0] = x[0];
          y[1] = x[1];
          y[2] = -x[2];
        }
  }
};

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
  char domain_param_filename[] = "./benchmarks/battery/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  // std::shared_ptr<dolfin::Mesh> mesh0;
  // std::shared_ptr<dolfin::Mesh> mesh_init;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut("./benchmarks/battery/output/mesh.pvd");
  // domain_build(&domain_par, mesh0, &subdomains, &surfaces);
  // domain_build(&domain_par, mesh_init, &subdomains, &surfaces);
  // mesh
  // dolfin::Point p0( -domain_par.length_x/2, -domain_par.length_y/2, -domain_par.length_z/2);
  // dolfin::Point p1(  domain_par.length_x/2,  domain_par.length_y/2,  domain_par.length_z/2);
  // auto mesh0 = std::make_shared<dolfin::BoxMesh>(p0, p1, domain_par.grid_x, domain_par.grid_y, domain_par.grid_z);
  auto mesh0 = std::make_shared<dolfin::Mesh>("./benchmarks/battery/cheese.xml");
  auto mesh_init = std::make_shared<dolfin::Mesh>(*mesh0);
  print_domain_param(&domain_par);

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par, non_dim_coeff_par;
  char coeff_param_filename[] = "./benchmarks/battery/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  // non_dimesionalize_coefficients(&domain_par, &coeff_par, &non_dim_coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/battery/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  // Setup FASP solver
  printf("FASP solver parameters...\n"); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/battery/bsr.dat";
  fasp_param_input(fasp_params, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;

  // open files for outputting solutions
  File cationFile("./benchmarks/battery/output/cation.pvd");
  File anionFile("./benchmarks/battery/output/anion.pvd");
  File potentialFile("./benchmarks/battery/output/potential.pvd");

  // Initialize guess
  printf("intial guess...\n"); fflush(stdout);
  LogCharge Cation(
    coeff_par.cation_lower_val,
    coeff_par.anion_lower_val,
    -domain_par.length_x/2.0,
    domain_par.length_x/2.0,
    coeff_par.bc_coordinate
  );
  LogCharge Anion(
    coeff_par.anion_lower_val,
    coeff_par.anion_upper_val,
    -domain_par.length_x/2.0,
    domain_par.length_x/2.0,
    coeff_par.bc_coordinate
  );

  // not ideal implementation: replace by a solve for voltage below
  Voltage Volt(
    coeff_par.potential_lower_val,
    coeff_par.potential_upper_val,
    -domain_par.length_x/2.0,
    domain_par.length_x/2.0,
    coeff_par.bc_coordinate
  );

// meshOut << *mesh0;

  auto periodic_boundary = std::make_shared<PeriodicBoundary>();

  // interpolate
  auto V_init = std::make_shared<pnp::FunctionSpace>(mesh_init,periodic_boundary);
  auto initialGuessFunction = std::make_shared<Function>(V_init);
  auto initialCation = std::make_shared<Function>((*initialGuessFunction)[0]);
  auto initialAnion = std::make_shared<Function>((*initialGuessFunction)[1]);
  auto initialPotential = std::make_shared<Function>((*initialGuessFunction)[2]);
  initialCation->interpolate(Cation);
  initialAnion->interpolate(Anion);
  initialPotential->interpolate(Volt);

  //*************************************************************
  //  Mesh adaptivity
  //*************************************************************
  // interpolate analytic expressions
  printf("interpolate analytic expressions onto initial mesh...\n\n"); fflush(stdout);
  auto V0 = std::make_shared<pnp::FunctionSpace>(mesh0);
  // auto initialGuessFunction0 = std::make_shared<Function>(V0);
  auto solutionFunction0 = std::make_shared<Function>(V0);
  auto cation0 = std::make_shared<Function>((*solutionFunction0)[0]);
  auto anion0 = std::make_shared<Function>((*solutionFunction0)[1]);
  auto potential0 = std::make_shared<Function>((*solutionFunction0)[2]);
  cation0->interpolate(Cation);
  anion0->interpolate(Anion);
  potential0->interpolate(Volt);

  // set adaptivity parameters
  auto mesh = std::make_shared<Mesh>(*mesh0);
  double entropy_tol = newtparam.adapt_tol;
  unsigned int num_adapts = 0, max_adapts = 5;
  bool adaptive_convergence = false;


  // adaptivity loop
  printf("Adaptivity loop\n"); fflush(stdout);
  while (!adaptive_convergence)
  {
    // output mesh
    meshOut << *mesh;

    // Initialize variational forms
    printf("\tvariational forms...\n"); fflush(stdout);
    auto V = std::make_shared<pnp::FunctionSpace>(mesh);
    pnp::BilinearForm a_pnp(V,V);
    pnp::LinearForm L_pnp(V);
    auto eps = std::make_shared<Constant>(coeff_par.relative_permittivity);
    auto Dp = std::make_shared<Constant>(coeff_par.cation_diffusivity);
    auto Dn = std::make_shared<Constant>(coeff_par.anion_diffusivity);
    auto qp = std::make_shared<Constant>(coeff_par.cation_valency);
    auto qn = std::make_shared<Constant>(coeff_par.anion_valency);
    auto zero = std::make_shared<Constant>(0.0);
    a_pnp.eps = eps; L_pnp.eps = eps;
    a_pnp.Dp = Dp; L_pnp.Dp = Dp;
    a_pnp.Dn = Dn; L_pnp.Dn = Dn;
    a_pnp.qp = qp; L_pnp.qp = qp;
    a_pnp.qn = qn; L_pnp.qn = qn;

    // analytic solution
    auto analyticSolutionFunction = std::make_shared<Function>(V);
    auto analyticCation= std::make_shared<Function>((*analyticSolutionFunction)[0]);
    auto analyticAnion= std::make_shared<Function>((*analyticSolutionFunction)[1]);
    auto analyticPotential= std::make_shared<Function>((*analyticSolutionFunction)[2]);
    analyticCation->interpolate(*zero);
    analyticAnion->interpolate(*zero);
    analyticPotential->interpolate(*zero);
    L_pnp.cation = analyticCation;
    L_pnp.anion = analyticAnion;
    L_pnp.potential = analyticPotential;

    File EXcationFile("./benchmarks/battery/output/Ex_cation.pvd");
    File EXanionFile("./benchmarks/battery/output/Ex_anion.pvd");
    File EXpotentialFile("./benchmarks/battery/output/Ex_potential.pvd");
    EXcationFile << *analyticCation;
    EXanionFile << *analyticAnion;
    EXpotentialFile << *analyticPotential;

    // Set Dirichlet boundaries
    printf("\tboundary conditions...\n"); fflush(stdout);
    auto zero_vec = std::make_shared<Constant>(0.0, 0.0, 0.0);
    auto boundary = std::make_shared<SymmBoundaries>(coeff_par.bc_coordinate, -domain_par.length_x/2.0, domain_par.length_x/2.0);
    dolfin::DirichletBC bc(V, zero_vec, boundary);

    // Interpolate analytic expressions
    printf("\tinterpolate solution onto new mesh...\n"); fflush(stdout);
    auto solutionFunction = std::make_shared<Function>(V);
    auto cationSolution = std::make_shared<Function>((*solutionFunction)[0]);
    auto anionSolution = std::make_shared<Function>((*solutionFunction)[1]);
    auto potentialSolution = std::make_shared<Function>((*solutionFunction)[2]);
    // cationSolution.interpolate(initialCation);
    cationSolution->interpolate(*cation0);
    // anionSolution.interpolate(initialAnion);
    anionSolution->interpolate(*anion0);
    // solve for voltage
    // potentialSolution.interpolate(initialPotential);
    potentialSolution->interpolate(*potential0);

    // write computed solution to file
    printf("\toutput projected solution to file\n"); fflush(stdout);
    cationFile << *cationSolution;
    anionFile << *anionSolution;
    potentialFile << *potentialSolution;

    // map dofs
    ivector cation_dofs;
    ivector anion_dofs;
    ivector potential_dofs;
    get_dofs(solutionFunction.get(), &cation_dofs, 0);
    get_dofs(solutionFunction.get(), &anion_dofs, 1);
    get_dofs(solutionFunction.get(), &potential_dofs, 2);

    //EAFE Formulation
    if (eafe_switch)
      printf("\tEAFE initialization...\n");
    auto V_cat = std::make_shared<EAFE::FunctionSpace>(mesh);
    EAFE::BilinearForm a_cat(V_cat,V_cat);
    a_cat.alpha = Dp;
    a_cat.gamma = zero;
    auto V_an = std::make_shared<EAFE::FunctionSpace>(mesh);
    EAFE::BilinearForm a_an(V_an,V_an);
    a_an.alpha = Dn;
    a_an.gamma = zero;
    auto CatCatFunction = std::make_shared<Function>(V_cat);
    auto CatBetaFunction = std::make_shared<Function>(V_cat);
    auto AnAnFunction = std::make_shared<Function>(V_an);
    auto AnBetaFunction = std::make_shared<Function>(V_an);

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
    // auto solutionUpdate = std::make_shared<Function>(V);
    Function solutionUpdate(V);
    unsigned int newton_iteration = 0;

    // set initial residual
    printf("\tupdate initial residual...\n"); fflush(stdout);
    initial_residual = get_initial_residual(
      &L_pnp,
      &bc,
      initialCation,
      initialAnion,
      initialPotential
    );

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
      if (eafe_switch) {
        printf("\tcompute EAFE expressions...\n");
        CatCatFunction->interpolate(*cationSolution);
        CatBetaFunction->interpolate(*potentialSolution);
        *(CatBetaFunction->vector()) *= coeff_par.cation_valency;
        *(CatBetaFunction->vector()) += *(CatCatFunction->vector());
        AnAnFunction->interpolate(*anionSolution);
        AnBetaFunction->interpolate(*potentialSolution);
        *(AnBetaFunction->vector()) *= coeff_par.anion_valency;
        *(AnBetaFunction->vector()) += *(AnAnFunction->vector());
        a_cat.eta = CatCatFunction;
        a_cat.beta = CatBetaFunction;
        a_an.eta = AnAnFunction;
        a_an.beta = AnBetaFunction;
        assemble(A_cat, a_cat);
        assemble(A_an, a_an);

        // Modify Jacobian
        printf("\treplace Jacobian with EAFE approximations...\n"); fflush(stdout);
        replace_matrix(3,0, V.get(), V_cat.get(), &A_pnp, &A_cat);
        replace_matrix(3,1, V.get(), V_an.get() , &A_pnp, &A_an );
      }
      bc.apply(A_pnp);

      // Convert to fasp
      printf("\tconvert to FASP...\n"); fflush(stdout);
      EigenVector_to_dvector(&b_pnp,&b_fasp);
      EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
      A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
      fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);

      // solve the linear system using FASP solver and verify success
      printf("\tsolve linear system using FASP solver...\n"); fflush(stdout);
      status = fasp_solver_dbsr_krylov_amg(&A_fasp_bsr, &b_fasp, &solu_fasp, &itpar, &amgpar);
      if (status < 0) {
        printf("\n### WARNING: FASP solver failed! Exit status = %d.\n", status);
        newton_iteration = max_newton_iters;
      }
      else {
        printf("\tsolved linear system successfully...\n");

        // map solu_fasp into solutionUpdate
        printf("\tconvert FASP solution to function...\n"); fflush(stdout);
        copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &cation_dofs, &cation_dofs);
        copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &anion_dofs, &anion_dofs);
        copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &potential_dofs, &potential_dofs);

        Function dAnion = solutionUpdate[0];
        Function dCation = solutionUpdate[1];
        Function dPotential = solutionUpdate[2];
        *(cationSolution->vector())+=*(dAnion.vector());
        *(anionSolution->vector())+=*(dCation.vector());
        *(potentialSolution->vector())+=*(dPotential.vector());


        // update nonlinear residual
        L_pnp.CatCat = cationSolution;
        L_pnp.AnAn = anionSolution;
        L_pnp.EsEs = potentialSolution;
        assemble(b_pnp, L_pnp);
        bc.apply(b_pnp);
        relative_residual = b_pnp.norm("l2") / initial_residual;
        printf("\t\trelative residual : %e\n",relative_residual);

        if (relative_residual < 0.0) {
          printf("Newton backtracking failed!\n");
          printf("\tresidual has not decreased after damping %d times\n", newtparam.damp_it);
          printf("\tthe relative residual is %e\n", relative_residual);
          relative_residual *= -1.0;
        }

        // update nonlinear residual
        // L_pnp.CatCat = cationSolution;
        // L_pnp.AnAn = anionSolution;
        // L_pnp.EsEs = potentialSolution;
        // assemble(b_pnp, L_pnp);
        // bc.apply(b_pnp);

        // print
        cationFile << *cationSolution;
        anionFile << *anionSolution;
        potentialFile << *potentialSolution;
      }
    }

    // check status of Newton solver
    if (std::isnan(relative_residual)) {
      printf("\n### WARNING: Newton solver failed...\n");
      printf("\trelative residual is NaN!!\n\n");
      adaptive_convergence = true;
      break;
    }
    else if (status < 0) {
      entropy_tol *= 0.1;
      printf("\tdrop adaptivity tolerance to %e\n\n", entropy_tol);
    }
    else if (relative_residual < nonlinear_tol) {
      printf("\nSuccessfully solved the system below desired residual in %d steps!\n\n", newton_iteration);
    }
    else {
      printf("\n### WARNING: Newton solver failed...\n");
      printf("\tDid not converge in %d Newton iterations...\n", max_newton_iters);
      printf("\tcurrent relative residual is %e > %e\n\n", relative_residual, nonlinear_tol);
    }

    // cationFile << cationSolution;
    // anionFile << anionSolution;
    // potentialFile << potentialSolution;

    // compute local entropy and refine mesh
    // printf("Computing local entropy for refinement\n");
    // unsigned int num_refines;
    // std::shared_ptr<Mesh> mesh_ptr;
    // num_refines = check_local_entropy (
    //   cationSolution,
    //   coeff_par.cation_valency,
    //   anionSolution,
    //   coeff_par.anion_valency,
    //   potentialSolution,
    //   mesh_ptr,
    //   entropy_tol,
    //   newtparam.max_cells
    // );
    //
    // if (num_refines == 0) {
    //   // successful solve
    //   printf("\tsuccessfully distributed entropy below desired entropy in %d adapts!\n\n", num_adapts);
    //   adaptive_convergence = true;
    //   break;
    // }
    // else if ( ++num_adapts > max_adapts ) {
    //   // failed adaptivity
    //   printf("\nDid not adapt mesh to entropy in %d adapts...\n", max_adapts);
    //   adaptive_convergence = true;
    //   break;
    // }
    //
    // // adapt solutions to refined mesh
    // if (num_refines == 1)
    //   printf("\tadapting the mesh using one level of local refinement...\n");
    // else
    //   printf("\tadapting the mesh using %d levels of local refinement...\n", num_refines);
    //
    // // std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(*mesh00) );
    // std::shared_ptr<GenericFunction> cation0 = adapt(cationSolution, mesh_ptr);
    // std::shared_ptr<GenericFunction> anion0 = adapt(anionSolution, mesh_ptr);
    // std::shared_ptr<GenericFunction> potential0 = adapt(potentialSolution, mesh_ptr);
    // *mesh = *mesh_ptr;
    // mesh->bounding_box_tree()->build(*mesh); // to ensure the building_box_tree is correctly indexed

  }

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}

// double update_solution_pnp (
//   dolfin::Function* iterate0,
//   dolfin::Function* iterate1,
//   dolfin::Function* iterate2,
//   dolfin::Function* update0,
//   dolfin::Function* update1,
//   dolfin::Function* update2,
//   double relative_residual,
//   double initial_residual,
//   pnp::LinearForm* L,
//   const dolfin::DirichletBC* bc,
//   newton_param* params )
// {
//   // compute residual
//   dolfin::Function _iterate0(*iterate0);
//   dolfin::Function _iterate1(*iterate1);
//   dolfin::Function _iterate2(*iterate2);
//   dolfin::Function _update0(*update0);
//   dolfin::Function _update1(*update1);
//   dolfin::Function _update2(*update2);
//   update_solution(&_iterate0, &_update0);
//   update_solution(&_iterate1, &_update1);
//   update_solution(&_iterate2, &_update2);
//   L->CatCat = _iterate0;
//   L->AnAn = _iterate1;
//   L->EsEs = _iterate2;
//   EigenVector b;
//   assemble(b, *L);
//   bc->apply(b);
//   double new_relative_residual = b.norm("l2") / initial_residual;
//
//   // backtrack loop
//   unsigned int damp_iters = 0;
//   printf("\t\trelative residual after damping %d times: %e\n", damp_iters, new_relative_residual);
//
//   while ( new_relative_residual > relative_residual && damp_iters < params->damp_it )
//   {
//     damp_iters++;
//     *(_iterate0.vector()) = *(iterate0->vector());
//     *(_iterate1.vector()) = *(iterate1->vector());
//     *(_iterate2.vector()) = *(iterate2->vector());
//     *(_update0.vector()) *= params->damp_factor;
//     *(_update1.vector()) *= params->damp_factor;
//     *(_update2.vector()) *= params->damp_factor;
//     update_solution(&_iterate0, &_update0);
//     update_solution(&_iterate1, &_update1);
//     update_solution(&_iterate2, &_update2);
//     L->CatCat = _iterate0;
//     L->AnAn = _iterate1;
//     L->EsEs = _iterate2;
//     assemble(b, *L);
//     bc->apply(b);
//     new_relative_residual = b.norm("l2") / initial_residual;
//     printf("\t\trelative residual after damping %d times: %e\n", damp_iters, new_relative_residual);
//   }
//
//   // check for decrease
//   if ( new_relative_residual > relative_residual )
//     return -new_relative_residual;
//
//   // update iterates
//   *(iterate0->vector()) = *(_iterate0.vector());
//   *(iterate1->vector()) = *(_iterate1.vector());
//   *(iterate2->vector()) = *(_iterate2.vector());
//   return new_relative_residual;
// }

double get_initial_residual (
  pnp::LinearForm* L,
  const dolfin::DirichletBC* bc,
  std::shared_ptr<dolfin::Function> cation,
  std::shared_ptr<dolfin::Function> anion,
  std::shared_ptr<dolfin::Function> potential)
{
  // auto V = std::make_shared<pnp::FunctionSpace>( *(cation->function_space()->mesh()) );
  // dolfin::Function adapt_func(V);
  // dolfin::Function adapt_cation(adapt_func[0]);
  // dolfin::Function adapt_anion(adapt_func[1]);
  // dolfin::Function adapt_potential(adapt_func[2]);
  // adapt_cation.interpolate(*cation);
  // adapt_anion.interpolate(*anion);
  // adapt_potential.interpolate(*potential);
  L->CatCat = cation;
  L->AnAn = anion;
  L->EsEs = potential;
  EigenVector b;
  assemble(b, *L);
  bc->apply(b);
  return b.norm("l2");
}
