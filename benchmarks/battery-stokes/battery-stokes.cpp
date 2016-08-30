/*! \file lin_pnp.cpp
 *
 *  \brief Setup and solve the linearized PNP equation using FASP
 *
 *  \note Currently initializes the problem based on specification
 */
#include <boost/filesystem.hpp>
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
#include "L2Error.h"
#include "energy.h"
#include "newton.h"
#include "newton_functs.h"
#include "spheres.h"
#include <ctime>

using namespace dolfin;
// using namespace std;

bool eafe_switch = false;

double lower_cation_val = 1.0;  // 1 / m^3
double upper_cation_val = 1.0;  // 1 / m^3
double lower_anion_val = 1.0;  // 1 / m^3
double upper_anion_val = 1.0;  // 1 / m^3
double lower_potential_val = -1.0e-0;  // V
double upper_potential_val = +1.0e-0;  // V
double Lx = 72.0;
double Ly = 72.0;
double Lz = 72.0;

unsigned int dirichlet_coord = 0;

double time_step_size = 1.0;
double final_time = 10.0;

double get_initial_residual (
  pnp_with_stokes::LinearForm* L,
  stokes_with_pnp::LinearForm* Ls,
  const dolfin::DirichletBC* bc,
  const dolfin::DirichletBC* bc_stokes,
  dolfin::Function* cation,
  dolfin::Function* anion,
  dolfin::Function* potential,
  dolfin::Function* velocity,
  dolfin::Function* pressure);

double update_solution_pnp_stokes (
  dolfin::Function* iterate0,
  dolfin::Function* iterate1,
  dolfin::Function* iterate2,
  dolfin::Function* iterate3,
  dolfin::Function* iterate4,
  dolfin::Function* update0,
  dolfin::Function* update1,
  dolfin::Function* update2,
  dolfin::Function* update3,
  dolfin::Function* update4,
  double relative_residual,
  double initial_residual,
  pnp_with_stokes::LinearForm* L,
  stokes_with_pnp::LinearForm* Ls,
  const dolfin::DirichletBC* bc,
  const dolfin::DirichletBC* bc_stokes,
  newton_param* params);

// Sub domain for Periodic boundary condition
class PeriodicBoundary : public SubDomain
{
  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (
      std::abs(x[0]) < Lx / 2.0 + 5.0 * DOLFIN_EPS
      || std::abs(x[1]) < Ly / 2.0 + 5.0 * DOLFIN_EPS
      || std::abs(x[2]) < Lz / 2.0 + 5.0 * DOLFIN_EPS
    );
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {
    y[0] = -x[0];
    y[1] = -x[1];
    y[2] = -x[2];
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
  printf("\n-----------------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;
  parameters["refinement_algorithm"] = "plaza_with_parent_facets";

  //*************************************************************
  //  Initialization
  //*************************************************************
  printf("Initialize the problem\n"); fflush(stdout);

  // Deleting the folders:
  boost::filesystem::remove_all("./benchmarks/battery-stokes/output");
  boost::filesystem::remove_all("./benchmarks/battery-stokes/meshOut");

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  dolfin::Mesh mesh_adapt("./benchmarks/battery-stokes/mesh.xml.gz");
  // MeshFunction<std::size_t> sub_domains_adapt(mesh_adapt, "./benchmarks/battery-stokes/boundary_parts.xml.gz");
  // dolfin::MeshFunction<std::size_t> subdomains_init;
  // dolfin::MeshFunction<std::size_t> surfaces_init;
  dolfin::File meshOut("./benchmarks/battery-stokes/meshOut/mesh.pvd");
  // domain_build(&domain_par, &mesh_adapt, &subdomains_init, &surfaces_init);


  // dolfin::File BoundaryFile("./benchmarks/battery-stokes/meshOut/boundary.pvd");
  // BoundaryFile << sub_domains_adapt;
  meshOut << mesh_adapt;
  // return 0;

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par;
  char coeff_param_filename[] = "./benchmarks/battery-stokes/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/battery-stokes/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  // Setup FASP solver
  printf("FASP solver parameters..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/battery-stokes/bsr.dat";
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
  char fasp_ns_params[] = "./benchmarks/battery-stokes/nsbcsr.dat";
  fasp_ns_param_input(fasp_ns_params, &stokes_inpar);
  fasp_ns_param_init(&stokes_inpar, &stokes_itpar, &stokes_amgpar, &stokes_ilupar, &stokes_schpar);
  printf("done\n"); fflush(stdout);

  // File
  std::ofstream ofs;
  ofs.open ("./benchmarks/battery-stokes/data.txt", std::ofstream::out);
  ofs << "starting mesh size =" << mesh_adapt.num_cells() << "\n";
  ofs << "t" << "\t" << "NewtonIteration" << "\t" << "RelativeResidual" << "\t" << "Cation" << "\t" << "Anion" << "\t" << "Potential" << "\t" << "Energy" << "\t"<< "TimeElaspsed" << "\t" << "MeshSize" << "\n";
  ofs.close();

  // open files for outputting solutions
  File cationFile("./benchmarks/battery-stokes/output/cation.pvd");
  File anionFile("./benchmarks/battery-stokes/output/anion.pvd");
  File potentialFile("./benchmarks/battery-stokes/output/potential.pvd");
  File velocityFile("./benchmarks/battery-stokes/output/velocity.pvd");
  File pressureFile("./benchmarks/battery-stokes/output/pressure.pvd");


  // PeriodicBoundary periodic_boundary;

  // PREVIOUS ITERATE
  pnp_with_stokes::FunctionSpace V_init(mesh_adapt /*,periodic_boundary*/);
  dolfin::Function initial_soln(V_init);
  dolfin::Function initial_cation(initial_soln[0]);
  dolfin::Function initial_anion(initial_soln[1]);
  dolfin::Function initial_potential(initial_soln[2]);
  stokes_with_pnp::FunctionSpace V_init_stokes(mesh_adapt /*,periodic_boundary*/);
  dolfin::Function initial_soln_stokes(V_init_stokes);
  dolfin::Function initial_velocity(initial_soln_stokes[0]);
  dolfin::Function initial_pressure(initial_soln_stokes[1]);

  // Constants
  Constant eps(coeff_par.relative_permittivity);
  Constant Dp(coeff_par.cation_diffusivity);
  Constant Dn(coeff_par.anion_diffusivity);
  Constant qp(coeff_par.cation_valency);
  Constant qn(coeff_par.anion_valency);
  Constant C_dt(time_step_size);
  Constant cat_alpha(coeff_par.cation_diffusivity*time_step_size);
  Constant an_alpha(coeff_par.anion_diffusivity*time_step_size);
  Constant one(1.0);
  Constant zero(0.0);
  Constant zero_vec3(0.0, 0.0, 0.0);
  Constant CU_init(0.1);
  Constant mu(0.1);
  Constant penalty(1.0e+2);

  LogCharge_SPH Cation(
    lower_cation_val,
    upper_cation_val,
    -Lx / 2.0,
    Lx / 2.0,
    dirichlet_coord
  );

  LogCharge_SPH Anion(
    lower_anion_val,
    upper_anion_val,
    -Lx / 2.0,
    Lx / 2.0,
    dirichlet_coord
  );

  Potential_SPH Volt(
    lower_potential_val,
    upper_potential_val,
    -Lx / 2.0,
    Lx / 2.0,
    dirichlet_coord
  );

  FluidVelocity Velocity(1.0,1.0,Lx,dirichlet_coord);


  printf("Interpolating intial gueses..");fflush(stdout);
  initial_cation.interpolate(Cation);
  initial_anion.interpolate(Anion);
  initial_potential.interpolate(Volt);
  initial_velocity.interpolate(Velocity);
  initial_pressure.interpolate(zero);
  printf("done\n");fflush(stdout);

  // output solution after solved for timestep
  cationFile << initial_cation;
  anionFile << initial_anion;
  potentialFile << initial_potential;
  velocityFile << initial_velocity;
  pressureFile << initial_pressure;

  // initialize error
  double cationError = 0.0;
  double anionError = 0.0;
  double potentialError = 0.0;
  double velocityError = 0.0;
  double pressureError = 0.0;
  double energy = 0.0;

  // Time
  std::clock_t begin = std::clock();
  std::clock_t end;
  double timeElaspsed;

  // Fasp matrices and vectors
  dCSRmat A_fasp;
  dBSRmat A_fasp_bsr;
  dvector b_fasp, solu_fasp;
  dvector b_stokes_fasp, solu_stokes_fasp;
  block_dCSRmat A_stokes_fasp;

  SpheresSubDomain SPS;

  for (double t = time_step_size; t < final_time; t += time_step_size) {
    // printf("\nSet voltage to %e...\n", volt); fflush(stdout);

    //*************************************************************
    //  Mesh adaptivity
    //*************************************************************


    // set adaptivity parameters
    dolfin::Mesh mesh(mesh_adapt);
    // dolfin::MeshFunction<std::size_t> sub_domains(sub_domains_adapt);
    // dolfin::MeshFunction<std::size_t> sub_domains = adapt(sub_domains_adapt, mesh);
    double entropy_tol = newtparam.adapt_tol;
    unsigned int num_adapts = 0, max_adapts = 5;
    bool adaptive_convergence = false;

    // initialize storage functions for adaptivity
    printf("store previous solution and initialize solution functions..."); fflush(stdout);
    // PNP
    pnp_with_stokes::FunctionSpace V_adapt(mesh_adapt /*,periodic_boundary*/);
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

    dolfin::Function dsoln_adapt(V_adapt);
    dolfin::Function dcation_adapt(dsoln_adapt[0]);
    dolfin::Function danion_adapt(dsoln_adapt[1]);
    dolfin::Function dpotential_adapt(dsoln_adapt[2]);
    dcation_adapt.interpolate(zero);
    danion_adapt.interpolate(zero);
    dpotential_adapt.interpolate(zero);

    // Stokes
    stokes_with_pnp::FunctionSpace VS_adapt(mesh_adapt /*,periodic_boundary*/);
    dolfin::Function prev_stokes_soln_adapt(VS_adapt);
    dolfin::Function prev_velocity_adapt(prev_stokes_soln_adapt[0]);
    dolfin::Function prev_pressure_adapt(prev_stokes_soln_adapt[1]);
    prev_velocity_adapt.interpolate(initial_velocity);
    prev_pressure_adapt.interpolate(initial_pressure);

    dolfin::Function stokes_soln_adapt(VS_adapt);
    dolfin::Function velocity_adapt(stokes_soln_adapt[0]);
    dolfin::Function pressure_adapt(stokes_soln_adapt[1]);
    velocity_adapt.interpolate(initial_velocity);
    pressure_adapt.interpolate(initial_pressure);

    dolfin::Function dstokes_soln_adapt(VS_adapt);
    dolfin::Function dvelocity_adapt(dstokes_soln_adapt[0]);
    dolfin::Function dpressure_adapt(dstokes_soln_adapt[1]);
    dvelocity_adapt.interpolate(zero_vec3);
    dpressure_adapt.interpolate(zero);
    printf("done\n");fflush(stdout);

    // adaptivity loop
    printf("Adaptivity loop\n"); fflush(stdout);
    while (!adaptive_convergence)
    {
      // mark and output mesh
      FacetFunction<std::size_t> boundaries(mesh);
      boundaries.set_all(0);
      SPS.mark(boundaries, 1);
      meshOut << boundaries;

      // Initialize variational forms
      printf("\tvariational forms...\n"); fflush(stdout);
      // PNP
      pnp_with_stokes::FunctionSpace V(mesh /*,periodic_boundary*/);
      pnp_with_stokes::BilinearForm a_pnp(V,V);
      pnp_with_stokes::LinearForm L_pnp(V);
      a_pnp.eps = eps; L_pnp.eps = eps;
      a_pnp.Dp = Dp; L_pnp.Dp = Dp;
      a_pnp.Dn = Dn; L_pnp.Dn = Dn;
      a_pnp.qp = qp; L_pnp.qp = qp;
      a_pnp.qn = qn; L_pnp.qn = qn;
      a_pnp.dt = C_dt; L_pnp.dt = C_dt;
      L_pnp.g = one;
      L_pnp.ds = boundaries;


      //Stokes
      stokes_with_pnp::FunctionSpace Vs(mesh /*,periodic_boundary*/);
      stokes_with_pnp::BilinearForm a_s(Vs,Vs);
      stokes_with_pnp::LinearForm L_s(Vs);
      L_s.qp = qp; L_s.qn = qn;
      a_s.mu = mu; L_s.mu = mu;
      a_s.alpha = penalty; L_s.alpha = penalty;


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


      // Interpolate previous solutions analytic expressions
      printf("\tinterpolate previous step solution onto new mesh...\n"); fflush(stdout);
      // PNP
      dolfin::Function prev_soln(V);
      dolfin::Function previous_cation(prev_soln[0]);
      previous_cation.interpolate(prev_cation_adapt);
      dolfin::Function previous_anion(prev_soln[1]);
      previous_anion.interpolate(prev_anion_adapt);
      dolfin::Function previous_potential(prev_soln[2]);
      previous_potential.interpolate(prev_potential_adapt);

      // Stokes
      dolfin::Function prev_stokes_soln(Vs);
      dolfin::Function previous_velocity(prev_stokes_soln[0]);
      previous_velocity.interpolate(prev_velocity_adapt);
      dolfin::Function previous_pressure(prev_stokes_soln[1]);
      previous_pressure.interpolate(prev_pressure_adapt);

      printf("\tinterpolate solution onto new mesh...\n"); fflush(stdout);
      //PNP
      dolfin::Function solutionFunction(V);
      dolfin::Function cationSolution(solutionFunction[0]);
      cationSolution.interpolate(cation_adapt);
      dolfin::Function anionSolution(solutionFunction[1]);
      anionSolution.interpolate(anion_adapt);
      dolfin::Function potentialSolution(solutionFunction[2]);
      potentialSolution.interpolate(potential_adapt);

      //Stokes
      dolfin::Function solutionStokesFunction(Vs);
      dolfin::Function VelocitySolution(solutionStokesFunction[0]);
      VelocitySolution.interpolate(velocity_adapt);
      dolfin::Function PressureSolution(solutionStokesFunction[1]);
      PressureSolution.interpolate(pressure_adapt);

      // Set Dirichlet boundaries
      printf("\tboundary conditions...\n"); fflush(stdout);
      SymmBoundaries boundary(dirichlet_coord, -Lx / 2.0, Lx / 2.0);
      dolfin::DirichletBC bc(V, zero_vec3, boundary);
      printf("\t\tdone\n"); fflush(stdout);
      // map dofs
      ivector cation_dofs;
      ivector anion_dofs;
      ivector potential_dofs;
      get_dofs(&solutionFunction, &cation_dofs, 0);
      get_dofs(&solutionFunction, &anion_dofs, 1);
      get_dofs(&solutionFunction, &potential_dofs, 2);

      //Stokes
      dolfin::SubSpace Vs1(Vs,0);
      dolfin::DirichletBC bc_stokes(Vs1, zero_vec3, boundary);
      ivector velocity_dofs;
      ivector pressure_dofs;
      get_dofs(&solutionStokesFunction, &velocity_dofs, 0);
      get_dofs(&solutionStokesFunction, &pressure_dofs, 1);

      //EAFE Formulation
      if (eafe_switch)
        printf("\tEAFE initialization...\n");
      EAFE::FunctionSpace V_cat(mesh /*,periodic_boundary*/);
      EAFE::BilinearForm a_cat(V_cat,V_cat);
      a_cat.alpha = an_alpha;
      a_cat.gamma = one;
      EAFE::FunctionSpace V_an(mesh /*,periodic_boundary*/);
      EAFE::BilinearForm a_an(V_an,V_an);
      a_an.alpha = cat_alpha;
      a_an.gamma = one;
      dolfin::Function CatCatFunction(V_cat);
      dolfin::Function CatBetaFunction(V_cat);
      dolfin::Function AnAnFunction(V_an);
      dolfin::Function AnBetaFunction(V_an);

      // initialize linear system
      printf("\tlinear algebraic objects...\n"); fflush(stdout);
      EigenMatrix A_pnp, A_cat, A_an;
      EigenVector b_pnp;
      EigenMatrix A_stokes;
      EigenVector b_stokes;

      //*************************************************************
      //  Initialize Newton solver
      //*************************************************************
      // Setup newton parameters and compute initial residual
      printf("\tNewton solver initialization...\n"); fflush(stdout);
      dolfin::Function solutionUpdate(V);
      dolfin::Function StokessolutionUpdate(Vs);
      unsigned int newton_iteration = 0;

      // set initial residual
      printf("\tupdate initial residual..."); fflush(stdout);
      initial_residual = get_initial_residual(
        &L_pnp,
        &L_s,
        &bc,
        &bc_stokes,
        &previous_cation,
        &previous_anion,
        &previous_potential,
        &previous_velocity,
        &previous_pressure
      );
      printf("done\n"); fflush(stdout);

      printf("\tcompute relative residual...\n"); fflush(stdout);
      L_pnp.CatCat = cationSolution;
      L_pnp.AnAn = anionSolution;
      L_pnp.EsEs = potentialSolution;
      L_pnp.CatCat_t0 = previous_cation;
      L_pnp.AnAn_t0 = previous_anion;
      L_pnp.uu = previous_velocity;
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
        a_pnp.uu = VelocitySolution;
        assemble(A_pnp, a_pnp);

        assemble(A_stokes, a_s);

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
        bc_stokes.apply(A_stokes);

        // Convert to fasp
        //PNP
        printf("\tconvert to FASP and solve...\n"); fflush(stdout);
        EigenVector_to_dvector(&b_pnp,&b_fasp);
        EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
        A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);

        copy_EigenMatrix_to_block_dCSRmat(&A_stokes, &A_stokes_fasp, &velocity_dofs, &pressure_dofs);

        /******************* TEST *******************/
        L_pnp.CatCat = cationSolution;
        L_pnp.AnAn = anionSolution;
        L_pnp.EsEs = potentialSolution;
        L_pnp.CatCat_t0 = previous_cation;
        L_pnp.AnAn_t0 = previous_anion;
        L_pnp.uu = previous_velocity;

        L_s.cation = cationSolution;
        L_s.anion = anionSolution;
        L_s.phi = potentialSolution;
        L_s.uu = VelocitySolution;
        L_s.pp = PressureSolution;
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

          &solutionUpdate,
          &StokessolutionUpdate,
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

        // if (status < 0)
        //   printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", status);
        // else
        //   printf("\tsolved linear system successfully...\n");

        // map solu_fasp into solutionUpdate
        printf("\tconvert FASP solution to function...\n"); fflush(stdout);
        // copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &cation_dofs, &cation_dofs);
        // copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &anion_dofs, &anion_dofs);
        // copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &potential_dofs, &potential_dofs);
        // copy_dvector_to_vector_function(&solu_stokes_fasp, &StokessolutionUpdate, &velocity_dofs, &velocity_dofs);
        // copy_dvector_to_vector_function(&solu_stokes_fasp,, &StokessolutionUpdate, &pressure_dofs, &pressure_dofs);

        // update solution and reset solutionUpdate
        printf("\tupdate solution...\n"); fflush(stdout);
        relative_residual = update_solution_pnp_stokes (
          &cationSolution,
          &anionSolution,
          &potentialSolution,
          &VelocitySolution,
          &PressureSolution,
          &(solutionUpdate[0]),
          &(solutionUpdate[1]),
          &(solutionUpdate[2]),
          &(StokessolutionUpdate[0]),
          &(StokessolutionUpdate[1]),
          relative_residual,
          initial_residual,
          &L_pnp,
          &L_s,
          &bc,
          &bc_stokes,
          &newtparam);
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
        L_pnp.AnAn_t0 = previous_anion;
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

        // output solution after solved for Newton update
        // cationFile << cationSolution;
        // anionFile << anionSolution;
        // potentialFile << potentialSolution;
        // velocityFile << VelocitySolution;
        // pressureFile << PresureSolution;

        fasp_dbsr_free(&A_fasp_bsr);
        fasp_bdcsr_free(&A_stokes_fasp);
        relative_residual = ( b_pnp.norm("l2") + b_stokes.norm("l2") ) / initial_residual;

      }

      if (relative_residual < nonlinear_tol)
        printf("\nSuccessfully solved the system below desired residual in %d steps!\n\n", newton_iteration);
      else {
        printf("\nDid not converge in %d Newton iterations at t=%e...\n", max_newton_iters,t);
        printf("\tcurrent relative residual is %e > %e\n\n", relative_residual, nonlinear_tol);
      }

      // compute local entropy and refine mesh
      printf("Computing electric field for refinement\n");
      unsigned int num_refines;
      num_refines = check_electric_field(
        &potentialSolution,
        &mesh_adapt,
        entropy_tol
      );

      // free fasp solution
      fasp_dvec_free(&solu_fasp);

      if ( (num_refines == 0) || ( ++num_adapts > max_adapts ) ){
        // successful solve
          if (num_refines == 0) printf("\tsuccessfully distributed electric field below desired electric field in %d adapts!\n\n", num_adapts);
          else printf("\nDid not adapt mesh to electric field in %d adapts...\n", max_adapts);
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

          ofs.open("./benchmarks/battery-stokes/data.txt", std::ofstream::out | std::ofstream::app);
          timeElaspsed = double(end - begin) / CLOCKS_PER_SEC;
          ofs << t << "\t" << newton_iteration << "\t" << relative_residual << "\t" << cationError << "\t" << anionError << "\t" << potentialError << "\t" << energy << "\t"<< timeElaspsed << "\t" << mesh.num_cells() << "\n";
          ofs.close();

          // store solution as solution from previous step
          std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(mesh_adapt) );
          initial_cation = adapt(cationSolution, mesh_ptr);
          initial_anion = adapt(anionSolution, mesh_ptr);
          initial_potential = adapt(potentialSolution, mesh_ptr);

          initial_velocity = adapt(VelocitySolution, mesh_ptr);
          initial_potential = adapt(PressureSolution, mesh_ptr);
          // sub_domains_adapt= adapt(sub_domains, mesh_ptr);

          // to ensure the building_box_tree is correctly indexed
          mesh = mesh_adapt;
          // sub_domains = sub_domains_adapt;
          mesh.bounding_box_tree()->build(mesh);
          mesh_adapt.bounding_box_tree()->build(mesh_adapt);

          // output solution after solved for timestep
          cationFile << initial_cation;
          anionFile << initial_anion;
          potentialFile << initial_potential;
          velocityFile << initial_velocity;
          pressureFile << initial_pressure;

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

      velocity_adapt = adapt(VelocitySolution, mesh_ptr);
      pressure_adapt = adapt(PressureSolution, mesh_ptr);

      prev_cation_adapt = adapt(previous_cation, mesh_ptr);
      prev_anion_adapt = adapt(previous_anion, mesh_ptr);
      prev_potential_adapt = adapt(previous_potential, mesh_ptr);
      prev_velocity_adapt = adapt(previous_velocity, mesh_ptr);
      prev_pressure_adapt = adapt(previous_pressure, mesh_ptr);
      mesh = mesh_adapt;
      mesh.bounding_box_tree()->build(mesh);  // to ensure the building_box_tree is correctly indexed


    }
  }
  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}

double update_solution_pnp_stokes (
  dolfin::Function* iterate0,
  dolfin::Function* iterate1,
  dolfin::Function* iterate2,
  dolfin::Function* iterate3,
  dolfin::Function* iterate4,
  dolfin::Function* update0,
  dolfin::Function* update1,
  dolfin::Function* update2,
  dolfin::Function* update3,
  dolfin::Function* update4,
  double relative_residual,
  double initial_residual,
  pnp_with_stokes::LinearForm* L,
  stokes_with_pnp::LinearForm* Ls,
  const dolfin::DirichletBC* bc,
  const dolfin::DirichletBC* bc_stokes,
  newton_param* params)
{
  // compute residual
  dolfin::Function _iterate0(*iterate0);
  dolfin::Function _iterate1(*iterate1);
  dolfin::Function _iterate2(*iterate2);
  dolfin::Function _iterate3(*iterate3);
  dolfin::Function _iterate4(*iterate4);
  dolfin::Function _update0(*update0);
  dolfin::Function _update1(*update1);
  dolfin::Function _update2(*update2);
  dolfin::Function _update3(*update3);
  dolfin::Function _update4(*update4);
  update_solution(&_iterate0, &_update0);
  update_solution(&_iterate1, &_update1);
  update_solution(&_iterate2, &_update2);
  update_solution(&_iterate3, &_update3);
  update_solution(&_iterate4, &_update4);
  dolfin::Constant C_dt(time_step_size);
  L->CatCat = _iterate0;
  L->AnAn = _iterate1;
  L->EsEs = _iterate2;
  L->uu = _iterate3;
  Ls->cation = _iterate0;
  Ls->anion = _iterate1;
  Ls->phi = _iterate2;
  Ls->uu = _iterate3;
  Ls->pp= _iterate4;
  EigenVector b, bs;
  assemble(b, *L);
  bc->apply(b);
  assemble(bs, *Ls);
  bc_stokes->apply(bs);
  double new_relative_residual = (b.norm("l2") + bs.norm("l2") )/ initial_residual;

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
    *(_iterate3.vector()) = *(iterate3->vector());
    *(_iterate4.vector()) = *(iterate4->vector());
    *(_update0.vector()) *= params->damp_factor;
    *(_update1.vector()) *= params->damp_factor;
    *(_update2.vector()) *= params->damp_factor;
    *(_update3.vector()) *= params->damp_factor;
    *(_update4.vector()) *= params->damp_factor;
    update_solution(&_iterate0, &_update0);
    update_solution(&_iterate1, &_update1);
    update_solution(&_iterate2, &_update2);
    update_solution(&_iterate3, &_update3);
    update_solution(&_iterate4, &_update4);
    L->CatCat = _iterate0;
    L->AnAn = _iterate1;
    L->EsEs = _iterate2;
    L->uu = _iterate3;
    Ls->cation = _iterate0;
    Ls->anion = _iterate1;
    Ls->phi = _iterate2;
    Ls->uu = _iterate3;
    Ls->pp= _iterate4;
    assemble(b, *L);
    bc->apply(b);
    assemble(bs, *Ls);
    bc_stokes->apply(bs);
    double new_relative_residual = (b.norm("l2") + bs.norm("l2") )/ initial_residual;
    printf("\t\trel_res after damping %d times: %e\n", damp_iters, new_relative_residual);
  }

  // check for decrease
  if ( new_relative_residual > relative_residual )
    return -new_relative_residual;

  // update iterates
  *(iterate0->vector()) = *(_iterate0.vector());
  *(iterate1->vector()) = *(_iterate1.vector());
  *(iterate2->vector()) = *(_iterate2.vector());
  *(iterate3->vector()) = *(_iterate3.vector());
  *(iterate4->vector()) = *(_iterate4.vector());
  return new_relative_residual;
}

double get_initial_residual (
  pnp_with_stokes::LinearForm* L,
  stokes_with_pnp::LinearForm* Ls,
  const dolfin::DirichletBC* bc,
  const dolfin::DirichletBC* bc_stokes,
  dolfin::Function* cation,
  dolfin::Function* anion,
  dolfin::Function* potential,
  dolfin::Function* velocity,
  dolfin::Function* pressure)
{
  pnp_with_stokes::FunctionSpace V( *(cation->function_space()->mesh()) );
  dolfin::Function adapt_func(V);
  dolfin::Function adapt_cation(adapt_func[0]);
  dolfin::Function adapt_anion(adapt_func[1]);
  dolfin::Function adapt_potential(adapt_func[2]);
  adapt_cation.interpolate(*cation);
  adapt_anion.interpolate(*anion);
  adapt_potential.interpolate(*potential);

  stokes_with_pnp::FunctionSpace Vs( *(velocity->function_space()->mesh()) );
  dolfin::Function adapt_func_stokes(Vs);
  dolfin::Function adapt_velocity(adapt_func_stokes[0]);
  dolfin::Function adapt_pressure(adapt_func_stokes[1]);
  adapt_velocity.interpolate(*velocity);
  adapt_pressure.interpolate(*pressure);

  L->CatCat = adapt_cation;
  L->AnAn = adapt_anion;
  L->EsEs = adapt_potential;
  L->CatCat_t0 = adapt_cation;
  L->AnAn_t0 = adapt_anion;
  L->uu = adapt_velocity;

  Ls->cation = adapt_cation;
  Ls->anion = adapt_anion;
  Ls->phi = adapt_potential;
  Ls->uu = adapt_velocity;
  Ls->pp = adapt_pressure;

  EigenVector b, bs;
  assemble(b, *L);
  bc->apply(b);
  assemble(bs, *Ls);
  bc_stokes->apply(bs);
  return b.norm("l2")+bs.norm("l2");
}
