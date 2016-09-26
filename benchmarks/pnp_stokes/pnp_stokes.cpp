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
#include "pnp_with_stokes.h"
#include "stokes_with_pnp.h"
#include "newton.h"
#include "newton_functs.h"
#include "L2Error.h"
#include "GS.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
  #define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

bool eafe_switch = false;

double get_initial_residual (
  pnp_with_stokes::LinearForm* L,
  stokes_with_pnp::LinearForm* Ls,
  const dolfin::DirichletBC* bc,
  const dolfin::DirichletBC* bc_stokes,
  std::shared_ptr<dolfin::Function> cation,
  std::shared_ptr<dolfin::Function> anion,
  std::shared_ptr<dolfin::Function> potential,
  std::shared_ptr<dolfin::Function> velocity,
  std::shared_ptr<dolfin::Function> pressure);

  class Bd_all : public dolfin::SubDomain
  {
      bool inside(const dolfin::Array<double>& x, bool on_boundary) const
      {
          return on_boundary;
      }

  };

  class zerovec4 : public dolfin::Expression
  {
  public:

    zerovec4() : Expression(4) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
      values[3] = 0.0;
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
  char domain_param_filename[] = "./benchmarks/pnp_stokes/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  // mesh
  dolfin::Point p0( -domain_par.length_x/2, -domain_par.length_y/2, -domain_par.length_z/2);
  dolfin::Point p1(  domain_par.length_x/2,  domain_par.length_y/2,  domain_par.length_z/2);
  auto mesh0 = std::make_shared<dolfin::BoxMesh>(p0, p1, domain_par.grid_x, domain_par.grid_y, domain_par.grid_z);
  auto mesh_init = std::make_shared<dolfin::Mesh>(*mesh0);
  print_domain_param(&domain_par);

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par, non_dim_coeff_par;
  char coeff_param_filename[] = "./benchmarks/pnp_stokes/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  // non_dimesionalize_coefficients(&domain_par, &coeff_par, &non_dim_coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/pnp_stokes/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  // Setup FASP solver
  printf("FASP solver parameters..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/pnp_stokes/bsr.dat";
  fasp_param_input(fasp_params, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  printf("done\n"); fflush(stdout);

  // Setup FASP solver
  printf("FASP solver parameters for stokes..."); fflush(stdout);
  input_ns_param stokes_inpar;
  itsolver_ns_param stokes_itpar;
  AMG_ns_param  stokes_amgpar;
  ILU_param stokes_ilupar;
  Schwarz_param stokes_schpar;
  char fasp_ns_params[] = "./benchmarks/pnp_stokes/ns.dat";
  fasp_ns_param_input(fasp_ns_params, &stokes_inpar);
  fasp_ns_param_init(&stokes_inpar, &stokes_itpar, &stokes_amgpar, &stokes_ilupar, &stokes_schpar);
  printf("done\n"); fflush(stdout);

  // open files for outputting solutions
  File cationFile("./benchmarks/pnp_stokes/output/cation.pvd");
  File anionFile("./benchmarks/pnp_stokes/output/anion.pvd");
  File potentialFile("./benchmarks/pnp_stokes/output/potential.pvd");
  File velocityFile("./benchmarks/pnp_stokes/output/velocity.pvd");
  File pressureFile("./benchmarks/pnp_stokes/output/pressure.pvd");

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

  FluidVelocity Velocity(1.0,-1.0,domain_par.length_x,coeff_par.bc_coordinate);

  // Constants
  auto eps = std::make_shared<Constant>(coeff_par.relative_permittivity);
  auto Dp = std::make_shared<Constant>(coeff_par.cation_diffusivity);
  auto Dn = std::make_shared<Constant>(coeff_par.anion_diffusivity);
  auto qp = std::make_shared<Constant>(coeff_par.cation_valency);
  auto qn = std::make_shared<Constant>(coeff_par.anion_valency);
  auto cat_alpha=std::make_shared<Constant>(coeff_par.cation_diffusivity);
  auto an_alpha=std::make_shared<Constant>(coeff_par.anion_diffusivity);
  auto one=std::make_shared<Constant>(1.0);
  auto zero=std::make_shared<Constant>(0.0);
  auto zero_vec3=std::make_shared<Constant>(0.0, 0.0, 0.0);
  auto zero_vec2=std::make_shared<Constant>(0.0, 0.0);
  auto zero_vec4=std::make_shared<zerovec4>();
  auto CU_init=std::make_shared<Constant>(0.1);
  auto mu=std::make_shared<Constant>(0.1);
  auto penalty1=std::make_shared<Constant>(1.0e-3);
  auto penalty2=std::make_shared<Constant>(1.0e-6);

// meshOut << *mesh0;

  // interpolate
  auto V_init = std::make_shared<pnp_with_stokes::FunctionSpace>(mesh_init);
  auto initialGuessFunction = std::make_shared<Function>(V_init);
  auto initialCation = std::make_shared<Function>((*initialGuessFunction)[0]);
  auto initialAnion = std::make_shared<Function>((*initialGuessFunction)[1]);
  auto initialPotential = std::make_shared<Function>((*initialGuessFunction)[2]);
  auto V_init_stokes = std::make_shared<stokes_with_pnp::FunctionSpace>(mesh_init);
  auto initial_soln_stokes = std::make_shared<Function>(V_init_stokes );
  auto initialVelocity = std::make_shared<Function>((*initial_soln_stokes)[0]);
  auto initialPressure = std::make_shared<Function>((*initial_soln_stokes)[1]);
  auto one_vec3 = std::make_shared<Constant>(1.0,1.0,1.0);
  auto vec_vel = std::make_shared<Constant>(1.0,0.0,0.0);

  initialCation->interpolate(Cation);
  initialAnion->interpolate(Anion);
  initialPotential->interpolate(Volt);
  initialVelocity->interpolate(*vec_vel);
  initialPressure->interpolate(*zero);

  //*************************************************************
  //  Mesh adaptivity
  //*************************************************************
  // interpolate analytic expressions
  printf("interpolate analytic expressions onto initial mesh...\n\n"); fflush(stdout);
  auto V0 = std::make_shared<pnp_with_stokes::FunctionSpace>(mesh0);
  auto solutionFunction0 = std::make_shared<Function>(V0);
  auto cation0 = std::make_shared<Function>((*solutionFunction0)[0]);
  auto anion0 = std::make_shared<Function>((*solutionFunction0)[1]);
  auto potential0 = std::make_shared<Function>((*solutionFunction0)[2]);
  auto V0_stokes = std::make_shared<stokes_with_pnp::FunctionSpace>(mesh0);
  auto solutionFunction0_stokes = std::make_shared<Function>(V0_stokes );
  auto velocity0 = std::make_shared<Function>((*solutionFunction0_stokes)[0]);
  auto pressure0 = std::make_shared<Function>((*solutionFunction0_stokes)[1]);
  cation0->interpolate(Cation);
  anion0->interpolate(Anion);
  potential0->interpolate(Volt);
  // velocity0->interpolate(Velocity);
  velocity0->interpolate(*one_vec3);
  pressure0->interpolate(*zero);

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
    auto V = std::make_shared<pnp_with_stokes::FunctionSpace>(mesh);
    pnp_with_stokes::BilinearForm a_pnp(V,V);
    pnp_with_stokes::LinearForm L_pnp(V);
    a_pnp.eps = eps; L_pnp.eps = eps;
    a_pnp.Dp = Dp; L_pnp.Dp = Dp;
    a_pnp.Dn = Dn; L_pnp.Dn = Dn;
    a_pnp.qp = qp; L_pnp.qp = qp;
    a_pnp.qn = qn; L_pnp.qn = qn;
    //Stokes
    auto Vs = std::make_shared<stokes_with_pnp::FunctionSpace>(mesh);
    stokes_with_pnp::BilinearForm a_s(Vs,Vs);
    stokes_with_pnp::LinearForm L_s(Vs);
    L_s.qp = qp; L_s.qn = qn;
    a_s.mu = mu; L_s.mu = mu;
    a_s.alpha1 = penalty1; L_s.alpha1 = penalty1;
    a_s.alpha2 = penalty2; L_s.alpha2 = penalty2;

    // Updates
    L_pnp.du = zero_vec3;
    L_pnp.dCat = zero;
    L_pnp.dAn = zero;
    L_pnp.dPhi = zero;
    L_s.du      = zero_vec3;
    L_s.dPress  = zero;
    L_s.dPhi    = zero;
    L_s.dCat    = zero;
    L_s.dAn     = zero;


    // Set Dirichlet boundaries
    printf("\tboundary conditions...\n"); fflush(stdout);
    auto boundary = std::make_shared<SymmBoundaries>(coeff_par.bc_coordinate, -domain_par.length_x/2.0, domain_par.length_x/2.0);
    auto bddd = std::make_shared<Bd_all>();
    dolfin::DirichletBC bc(V, zero_vec3, boundary);
    dolfin::DirichletBC bc_stokes(Vs->sub(0), zero_vec3, bddd);
    // dolfin::DirichletBC bc_stokes(Vs, zero_vec4, bddd);

    // Interpolate analytic expressions
    printf("\tinterpolate solution onto new mesh...\n"); fflush(stdout);
    auto solutionFunction = std::make_shared<Function>(V);
    auto cationSolution = std::make_shared<Function>((*solutionFunction)[0]);
    auto anionSolution = std::make_shared<Function>((*solutionFunction)[1]);
    auto potentialSolution = std::make_shared<Function>((*solutionFunction)[2]);
    auto solutionStokesFunction = std::make_shared<Function>(Vs);
    auto VelocitySolution = std::make_shared<Function>((*solutionStokesFunction)[0]);
    auto PressureSolution = std::make_shared<Function>((*solutionStokesFunction)[1]);

    cationSolution->interpolate(*cation0);
    anionSolution->interpolate(*anion0);
    potentialSolution->interpolate(*potential0);
    VelocitySolution->interpolate(*velocity0);
    PressureSolution->interpolate(*pressure0);

    // write computed solution to file
    printf("\toutput projected solution to file\n"); fflush(stdout);
    cationFile << *cationSolution;
    anionFile << *anionSolution;
    potentialFile << *potentialSolution;
    velocityFile << *VelocitySolution;
    pressureFile << *PressureSolution;

    // map dofs
    ivector cation_dofs;
    ivector anion_dofs;
    ivector potential_dofs;
    get_dofs(solutionFunction.get(), &cation_dofs, 0);
    get_dofs(solutionFunction.get(), &anion_dofs, 1);
    get_dofs(solutionFunction.get(), &potential_dofs, 2);
    ivector velocity_dofs;
    ivector pressure_dofs;
    get_dofs(solutionStokesFunction.get(), &velocity_dofs, 0);
    get_dofs(solutionStokesFunction.get(), &pressure_dofs, 1);

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
    EigenMatrix A_stokes;
    EigenVector b_stokes;
    // Fasp matrices and vectors
    dCSRmat A_fasp;
    dBSRmat A_fasp_bsr;
    dvector b_fasp, solu_fasp;
    dvector b_stokes_fasp, solu_stokes_fasp;
    block_dCSRmat A_stokes_fasp;

    //*************************************************************
    //  Initialize Newton solver
    //*************************************************************
    // Setup newton parameters and compute initial residual
    printf("\tNewton solver initialization...\n"); fflush(stdout);
    // auto solutionUpdate = std::make_shared<Function>(V);
    auto solutionUpdate = std::make_shared<dolfin::Function>(V);
    auto StokessolutionUpdate = std::make_shared<dolfin::Function>(Vs);
    unsigned int newton_iteration = 0;

    // set initial residual
    printf("\tupdate initial residual...\n"); fflush(stdout);
    initial_residual = get_initial_residual(
      &L_pnp,
      &L_s,
      &bc,
      &bc_stokes,
      initialCation,
      initialAnion,
      initialPotential,
      initialVelocity,
      initialPressure
    );

    printf("\tcompute relative residual...\n"); fflush(stdout);
    L_pnp.CatCat = cationSolution;
    L_pnp.AnAn = anionSolution;
    L_pnp.EsEs = potentialSolution;
    L_pnp.uu = VelocitySolution;
    assemble(b_pnp, L_pnp);
    bc.apply(b_pnp);

    L_s.cation = cationSolution;
    L_s.anion = anionSolution;
    L_s.phi = potentialSolution;
    L_s.uu = VelocitySolution;
    L_s.pp = PressureSolution;
    assemble(b_stokes, L_s);
    bc_stokes.apply(b_stokes);

    relative_residual = ( b_pnp.norm("l2") + b_stokes.norm("l2") ) / initial_residual;
    if (num_adapts == 0)
      printf("\tinitial nonlinear residual has l2-norm of %e\n", initial_residual);
    else
      printf("\tadapted relative nonlinear residual is %e\n", relative_residual);

    // fasp_dvec_alloc(b_pnp.size(), &solu_fasp);
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
      a_pnp.uu = VelocitySolution;
      assemble(A_pnp, a_pnp);
      assemble(A_stokes, a_s);

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
      bc_stokes.apply(A_stokes);
      int index = pressure_dofs.val[0];
      replace_row(index, &A_stokes);

      // Convert to fasp
      printf("\tconvert to FASP...\n"); fflush(stdout);
      EigenVector_to_dvector(&b_pnp,&b_fasp);
      EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
      A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
      // fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
      copy_EigenMatrix_to_block_dCSRmat(&A_stokes, &A_stokes_fasp, &velocity_dofs, &pressure_dofs);

      /******************* TEST *******************/
      // L_pnp.CatCat = cationSolution;
      // L_pnp.AnAn = anionSolution;
      // L_pnp.EsEs = potentialSolution;
      // L_pnp.uu = VelocitySolution;
      //
      // L_s.cation = cationSolution;
      // L_s.anion = anionSolution;
      // L_s.phi = potentialSolution;
      // L_s.uu = VelocitySolution;
      // L_s.pp = PressureSolution;
      /*******************************************/

      double relative_residual_tol = 1E-6;
      unsigned int max_bgs_it = 10;
      status  = electrokinetic_block_guass_seidel (
        &A_fasp_bsr,
        &A_stokes_fasp,
        &L_pnp,
        &L_s,
        &bc,
        &bc_stokes,

        solutionUpdate,
        StokessolutionUpdate,
        &velocity_dofs,
        &pressure_dofs,

        relative_residual_tol,
        max_bgs_it,

        &itpar,
        &amgpar,
        &stokes_itpar,
        &stokes_amgpar,
        &stokes_ilupar,
        &stokes_schpar

      );

      // Updates
      L_pnp.du = zero_vec3;
      L_pnp.dCat = zero;
      L_pnp.dAn = zero;
      L_pnp.dPhi = zero;
      L_s.du      = zero_vec3;
      L_s.dPress  = zero;
      L_s.dPhi    = zero;
      L_s.dCat    = zero;
      L_s.dAn     = zero;

      printf("\tconvert FASP solution to function...\n"); fflush(stdout);

      printf("\tupdate solution...\n"); fflush(stdout);
      Function dAnion = (*solutionUpdate)[0];
      Function dCation = (*solutionUpdate)[1];
      Function dPotential = (*solutionUpdate)[2];
      *(cationSolution->vector())+=*(dAnion.vector());
      *(anionSolution->vector())+=*(dCation.vector());
      *(potentialSolution->vector())+=*(dPotential.vector());

      Function dU = (*StokessolutionUpdate)[0];
      Function dPressure = (*StokessolutionUpdate)[1];
      *(VelocitySolution->vector())+=*(dU.vector());
      *(PressureSolution->vector())+=*(dPressure.vector());
      // update nonlinear residual
      L_pnp.CatCat = cationSolution;
      L_pnp.AnAn = anionSolution;
      L_pnp.EsEs = potentialSolution;
      L_pnp.uu = VelocitySolution;
      assemble(b_pnp, L_pnp);
      bc.apply(b_pnp);

      L_s.cation = cationSolution;
      L_s.anion = anionSolution;
      L_s.phi = potentialSolution;
      L_s.uu = VelocitySolution;
      L_s.pp = PressureSolution;
      assemble(b_stokes, L_s);
      bc_stokes.apply(b_stokes);

      double relative_residual = (b_pnp.norm("l2") + b_stokes.norm("l2") )/ initial_residual;
      printf("\t\trel_res after: %e\n", relative_residual);
      if (relative_residual < 0.0) {
        printf("Newton backtracking failed!\n");
        printf("\tresidual has not decreased after damping %d times\n", newtparam.damp_it);
        printf("\tthe relative residual is %e\n", relative_residual);
        relative_residual *= -1.0;
      }
        cationFile << *cationSolution;
        anionFile << *anionSolution;
        potentialFile << *potentialSolution;
        velocityFile << *VelocitySolution;
        pressureFile << *PressureSolution;
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
    // velocityFile << *VelocitySolution;
    // pressureFile << *PressureSolution;

    // compute local entropy and refine mesh
    printf("Computing local entropy for refinement\n");
    unsigned int num_refines;
    std::shared_ptr<Mesh> mesh_ptr;
    num_refines = check_local_entropy (
      cationSolution,
      coeff_par.cation_valency,
      anionSolution,
      coeff_par.anion_valency,
      potentialSolution,
      mesh_ptr,
      entropy_tol,
      newtparam.max_cells
    );

    if (num_refines == 0) {
      // successful solve
      printf("\tsuccessfully distributed entropy below desired entropy in %d adapts!\n\n", num_adapts);
      adaptive_convergence = true;
      break;
    }
    else if (num_refines == -1) {
      // failed adaptivity
      printf("\nDid not adapt mesh to entropy in %d adapts...\n", max_adapts);
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

    // std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(*mesh00) );
    std::shared_ptr<GenericFunction> cation0 = adapt(cationSolution, mesh_ptr);
    std::shared_ptr<GenericFunction> anion0 = adapt(anionSolution, mesh_ptr);
    std::shared_ptr<GenericFunction> potential0 = adapt(potentialSolution, mesh_ptr);
    std::shared_ptr<GenericFunction> velocity0 = adapt(VelocitySolution, mesh_ptr);
    std::shared_ptr<GenericFunction> pressure0 = adapt(PressureSolution, mesh_ptr);
    *mesh = *mesh_ptr;
    mesh->bounding_box_tree()->build(*mesh); // to ensure the building_box_tree is correctly indexed

  }

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}

double get_initial_residual (
  pnp_with_stokes::LinearForm* L,
  stokes_with_pnp::LinearForm* Ls,
  const dolfin::DirichletBC* bc,
  const dolfin::DirichletBC* bc_stokes,
  std::shared_ptr<dolfin::Function> cation,
  std::shared_ptr<dolfin::Function> anion,
  std::shared_ptr<dolfin::Function> potential,
  std::shared_ptr<dolfin::Function> velocity,
  std::shared_ptr<dolfin::Function> pressure)
{
  L->CatCat = cation;
  L->AnAn = anion;
  L->EsEs = potential;
  L->uu = velocity;

  Ls->cation = cation;
  Ls->anion = anion;
  Ls->phi = potential;
  Ls->uu = velocity;
  Ls->pp = pressure;

  EigenVector b, bs;
  assemble(b, *L);
  bc->apply(b);
  assemble(bs, *Ls);
  bc_stokes->apply(bs);
  return b.norm("l2")+bs.norm("l2");
}
