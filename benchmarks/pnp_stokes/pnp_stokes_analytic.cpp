/*! \file pnp_adaptive.cpp
 *
 *  \brief Setup and solve the PNP equations using adaptive meshing and FASP
 *
 *  \note Currently initializes the problem based on specification
 */
#include <math.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "pnp_stokes_analytic.h"
#include "stokes_with_pnp.h"
#include "newton.h"
#include "newton_functs.h"
#include "L2Error.h"
#include "L2ErrorRT.h"
#include "L2ErrorDG.h"
#include "GS.h"
#include "umfpack.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
  #define FASP_BSR     ON  /** use BSR format in fasp */
  #define WITH_UMFPACK ON
}
using namespace dolfin;
// using namespace std;

double Pi = M_PI;

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

class anion_analytic : public dolfin::Expression
{
public:

  // anion_analytic() : Expression(1) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -std::log(0.1)*std::cos(Pi/20.0*(x[0]+5.0+20.0));
  }

};

class cation_analytic : public dolfin::Expression
{
public:

  // cation_analytic() : Expression(1) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -std::log(0.1)*std::sin(Pi/20.0*(x[0]+5.0+20.0));
  }

};

class potential_analytic : public dolfin::Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -std::cos(Pi/10.0*(x[0]+5.0+20.0));
  }

};

class velocity_analytic : public dolfin::Expression
{
public:

  velocity_analytic() : Expression(3) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = std::sin(Pi/10.0*(x[0]+5.0+20.0));
    values[1] = 0.0;
    values[2] = 0.0;
  }

};

class pressure_analytic : public dolfin::Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = (x[0]+5.0)*(x[0]-5.0);
  }

};


int main(int argc, char** argv)
{

  printf("\n-----------------------------------------------------------    ");
  printf("\n Solving the linearized Poisson-Nernst-Planck system           ");
  printf("\n of a single cation and anion ");
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
  auto mesh = std::make_shared<dolfin::BoxMesh>(p0, p1, domain_par.grid_x, domain_par.grid_y, domain_par.grid_z);
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

  File cationAnalyticFile("./benchmarks/pnp_stokes/output/cationAnalyti.pvd");
  File anionAnalyticFile("./benchmarks/pnp_stokes/output/anionAnalyti.pvd");
  File potentialAnalyticFile("./benchmarks/pnp_stokes/output/potentialAnalyti.pvd");
  File velocityAnalyticFile("./benchmarks/pnp_stokes/output/velocityAnalyti.pvd");
  File pressureAnalyticFile("./benchmarks/pnp_stokes/output/pressureAnalyti.pvd");

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


  // interpolate
  auto V= std::make_shared<pnp_stokes_analytic::FunctionSpace>(mesh);
  auto initialFunction = std::make_shared<Function>(V);
  auto initialCation1 = std::make_shared<Function>((*initialFunction)[0]);
  auto initialAnion1 = std::make_shared<Function>((*initialFunction)[1]);
  auto initialPotential1 = std::make_shared<Function>((*initialFunction)[2]);
  auto initialVelocity1 = std::make_shared<Function>((*initialFunction)[3]);
  auto initialPressure1 = std::make_shared<Function>((*initialFunction)[4]);
  auto one_vec3 = std::make_shared<Constant>(1.0,1.0,1.0);
  auto vel_vec = std::make_shared<Constant>(0.0,0.0,0.0);

  initialCation1->interpolate(Cation);
  initialAnion1->interpolate(Anion);
  initialPotential1->interpolate(Volt);
  initialVelocity1->interpolate(*vel_vec);
  initialPressure1->interpolate(*zero);


  //Analytic solutions
  auto CationAn = std::make_shared<cation_analytic>();
  auto AnionAn = std::make_shared<anion_analytic>();
  auto PotentialAn = std::make_shared<potential_analytic>();
  auto VelocityAn = std::make_shared<velocity_analytic>();
  auto PressureAn = std::make_shared<pressure_analytic>();
  initialPressure1->interpolate(*PressureAn);
  // cation_analytic CationAn();
  // anion_analytic AnionAn();
  // potential_analytic PotentialAn();
  // velocity_analytic VelocityAn();
  // pressure_analytic PressureAn();
  auto AnalyticFunction = std::make_shared<Function>(V);
  auto AnalyticCation = std::make_shared<Function>((*AnalyticFunction)[0]);
  auto AnalyticAnion = std::make_shared<Function>((*AnalyticFunction)[1]);
  auto AnalyticPotential = std::make_shared<Function>((*AnalyticFunction)[2]);
  auto AnalyticVelocity = std::make_shared<Function>((*AnalyticFunction)[3]);
  auto AnalyticPressure = std::make_shared<Function>((*AnalyticFunction)[4]);
  AnalyticCation->interpolate(*CationAn);
  AnalyticAnion->interpolate(*AnionAn);
  AnalyticPotential->interpolate(*PotentialAn);
  AnalyticVelocity->interpolate(*VelocityAn);
  AnalyticPressure->interpolate(*PressureAn);
  cationAnalyticFile << *AnalyticCation;
  anionAnalyticFile << *AnalyticAnion;
  potentialAnalyticFile << *AnalyticPotential;
  velocityAnalyticFile << *AnalyticVelocity;
  pressureAnalyticFile << *AnalyticPressure;

  auto boundary = std::make_shared<SymmBoundaries>(coeff_par.bc_coordinate, -domain_par.length_x/2.0, domain_par.length_x/2.0);
  auto bddd = std::make_shared<Bd_all>();


  //  auto initialFunction = std::make_shared<Function>(V);
  std::vector<std::shared_ptr<const Function>> vvv = {initialCation1,initialAnion1,initialPotential1,initialVelocity1,initialPressure1};
  assign(initialFunction , vvv);
  dolfin::DirichletBC bci1(V->sub(0), AnalyticCation, bddd);
  dolfin::DirichletBC bci2(V->sub(1), AnalyticAnion, bddd);
  dolfin::DirichletBC bci3(V->sub(2), AnalyticPotential, bddd);
  dolfin::DirichletBC bci4(V->sub(3), AnalyticVelocity, bddd);
  dolfin::DirichletBC bci5(V->sub(4), AnalyticPressure, bddd);
  bci1.apply(*(initialFunction->vector()));
  bci2.apply(*(initialFunction->vector()));
  bci3.apply(*(initialFunction->vector()));
  bci4.apply(*(initialFunction->vector()));
  bci5.apply(*(initialFunction->vector()));
  auto initialCation = std::make_shared<Function>((*initialFunction)[0]);
  auto initialAnion = std::make_shared<Function>((*initialFunction)[1]);
  auto initialPotential = std::make_shared<Function>((*initialFunction)[2]);
  auto initialVelocity = std::make_shared<Function>((*initialFunction)[3]);
  auto initialPressure = std::make_shared<Function>((*initialFunction)[4]);


  // set adaptivity parameters
  double entropy_tol = newtparam.adapt_tol;
  unsigned int num_adapts = 0, max_adapts = 5;
  bool adaptive_convergence = false;

  // output mesh
  meshOut << *mesh;

  // Initialize variational forms
  printf("\tvariational forms...\n"); fflush(stdout);
  pnp_stokes_analytic::BilinearForm a(V,V);
  pnp_stokes_analytic::LinearForm L(V);
  a.eps = eps; L.eps = eps;
  a.Dp = Dp; L.Dp = Dp;
  a.Dn = Dn; L.Dn = Dn;
  a.qp = qp; L.qp = qp;
  a.qn = qn; L.qn = qn;
  a.mu = mu; L.mu = mu;
  a.alpha1 = penalty1; L.alpha1 = penalty1;
  a.alpha2 = penalty2; L.alpha2 = penalty2;

  L.cation = AnalyticCation;
  L.anion = AnalyticAnion;
  L.potential = AnalyticPotential;
  L.velocity = AnalyticVelocity;
  L.pressure = AnalyticPressure;

  // Set Dirichlet boundaries
  printf("\tboundary conditions...\n"); fflush(stdout);
  dolfin::DirichletBC bc1(V->sub(0), zero, bddd);
  dolfin::DirichletBC bc2(V->sub(1), zero, bddd);
  dolfin::DirichletBC bc3(V->sub(2), zero, bddd);
  dolfin::DirichletBC bc_stokes(V->sub(3), zero_vec3, bddd);
  // dolfin::DirichletBC bc_stokes(Vs, zero_vec4, bddd);


  // write computed solution to file
  printf("\toutput projected solution to file\n"); fflush(stdout);
  cationFile << *initialCation;
  anionFile << *initialAnion;
  potentialFile << *initialPotential;
  velocityFile << *initialVelocity;
  pressureFile << *initialPressure;

  // map dofs
  ivector cation_dofs;
  ivector anion_dofs;
  ivector potential_dofs;
  get_dofs(initialFunction.get(), &cation_dofs, 0);
  get_dofs(initialFunction.get(), &anion_dofs, 1);
  get_dofs(initialFunction.get(), &potential_dofs, 2);
  ivector velocity_dofs;
  ivector pressure_dofs;
  get_dofs(initialFunction.get(), &velocity_dofs, 3);
  get_dofs(initialFunction.get(), &pressure_dofs, 4);
  int index_fix = pressure_dofs.val[0];

  // initialize linear system
  printf("\tlinear algebraic objects...\n"); fflush(stdout);
  EigenMatrix A;
  EigenVector b;
  // Fasp matrices and vectors
  dCSRmat A_fasp;
  dBSRmat A_fasp_bsr;
  dvector b_fasp, solu_fasp;


  //*************************************************************
  //  Initialize Newton solver
  //*************************************************************
  // Setup newton parameters and compute initial residual
  printf("\tNewton solver initialization...\n"); fflush(stdout);
  // auto solutionUpdate = std::make_shared<Function>(V);
  auto solutionUpdate = std::make_shared<dolfin::Function>(V);
  unsigned int newton_iteration = 0;

  printf("\tInitial residual...\n"); fflush(stdout);
  L.CatCat = initialCation;
  L.AnAn = initialAnion;
  L.EsEs = initialPotential;
  L.uu = initialVelocity;
  L.pp = initialPressure;
  assemble(b, L);
  bc1.apply(b);
  bc2.apply(b);
  bc3.apply(b);
  bc_stokes.apply(b);
  b[index_fix]=0.0;

  initial_residual = b.norm("l2");
  relative_residual =1.0;
  printf("Initial rez = %e\n",initial_residual);
  fasp_dvec_alloc(b.size(), &solu_fasp);

  //*************************************************************
  //  Newton solver
  //*************************************************************
  printf("Solve the nonlinear system\n"); fflush(stdout);

  double nonlinear_tol = newtparam.tol;
  unsigned int max_newton_iters = newtparam.max_it;
  while (relative_residual > nonlinear_tol && newton_iteration < max_newton_iters )
  {
    printf("\nNewton iteration: %d\n", ++newton_iteration); fflush(stdout);

    // Construct stiffness matrix
    printf("\tconstruct stiffness matrix...\n"); fflush(stdout);
    a.CatCat = initialCation;
    a.AnAn = initialAnion;
    a.EsEs = initialPotential;
    a.uu = initialVelocity;
    assemble(A, a);

    bc1.apply(A);
    bc2.apply(A);
    bc3.apply(A);
    bc_stokes.apply(A);
    replace_row(index_fix, &A);
    EigenMatrix A_res(A);

    dCSRmat A_res_fasp;
    EigenMatrix_to_dCSRmat(&A_res, &A_res_fasp);

    // Convert to fasp
    printf("\tconvert to FASP...\n"); fflush(stdout);
    EigenVector_to_dvector(&b,&b_fasp);
    EigenMatrix_to_dCSRmat(&A,&A_fasp);
    fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
    // A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
    // copy_EigenMatrix_to_block_dCSRmat(&A_stokes, &A_stokes_fasp, &velocity_dofs, &pressure_dofs);
    // status = fasp_solver_dcsr_krylov_amg(&A_fasp, &b_fasp, &solu_fasp, &itpar, &amgpar);

    // list_lu_solver_methods();
    // solve(A, *solutionUpdate->vector(), b, "umfpack");

    // UMFPACK LU SOLVER
    dCSRmat A_tran;
    fasp_dcsr_trans(&A_fasp, &A_tran);
    fasp_dcsr_sort(&A_tran);
    fasp_dcsr_cp(&A_tran, &A_fasp);
    fasp_dcsr_free(&A_tran);
    void *Numeric;
    Numeric = fasp_umfpack_factorize(&A_fasp, 3);
    status = fasp_umfpack_solve(&A_fasp, &b_fasp, &solu_fasp, Numeric, 3);
    fasp_umfpack_free_numeric(Numeric);

    printf("\t status = %d\n",status);



    // fasp_dvec_print(b_fasp.row,&b_fasp);
    char filemat[20] = "matrix.txt";
    char filevec[20] = "rhs.txt";
    fasp_dcoo_write(filemat,&A_fasp);
    fasp_dvec_write (filevec,&b_fasp);


    fasp_blas_dcsr_aAxpy(-1.0,
                           &A_res_fasp,
                           solu_fasp.val,
                           b_fasp.val);
    double rez = fasp_blas_dvec_norm2 (&b_fasp);
    printf("\t rez = %e , rez_rel = %e\n",rez,rez/initial_residual);


    printf("\tconvert FASP solution to function...\n"); fflush(stdout);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdate.get(), &cation_dofs, &cation_dofs);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdate.get(), &anion_dofs, &anion_dofs);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdate.get(), &potential_dofs, &potential_dofs);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdate.get(), &velocity_dofs, &velocity_dofs);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdate.get(), &pressure_dofs, &pressure_dofs);


    printf("\tupdate solution...\n"); fflush(stdout);
    Function dAnion = (*solutionUpdate)[0];
    Function dCation = (*solutionUpdate)[1];
    Function dPotential = (*solutionUpdate)[2];
    Function dU = (*solutionUpdate)[3];
    Function dPressure = (*solutionUpdate)[4];
    *(initialCation->vector())+=*(dAnion.vector());
    *(initialAnion->vector())+=*(dCation.vector());
    *(initialPotential->vector())+=*(dPotential.vector());
    *(initialVelocity->vector())+=*(dU.vector());
    *(initialPressure->vector())+=*(dPressure.vector());


    cationFile << *initialCation;
    anionFile << *initialAnion;
    potentialFile << *initialPotential;
    velocityFile << *initialVelocity;
    pressureFile << *initialPressure;

    // update nonlinear residual
    L.CatCat = initialCation;
    L.AnAn = initialAnion;
    L.EsEs = initialPotential;
    L.uu = initialVelocity;
    L.pp = initialPressure;
    assemble(b, L);
    bc1.apply(b);
    bc2.apply(b);
    bc3.apply(b);
    bc_stokes.apply(b);
    b[index_fix]=0.0;


    relative_residual = b.norm("l2")/ initial_residual;
    printf("\t\trel_res after: %e\n", relative_residual);
    if (relative_residual < 0.0) {
      printf("Newton backtracking failed!\n");
      printf("\tresidual has not decreased after damping %d times\n", newtparam.damp_it);
      printf("\tthe relative residual is %e\n", relative_residual);
      relative_residual *= -1.0;

    }

    }

    *(AnalyticCation->vector()) -= *(initialCation->vector());
    *(AnalyticAnion->vector()) -= *(initialAnion->vector());
    *(AnalyticPotential->vector()) -= *(initialPotential->vector());
    *(AnalyticVelocity->vector()) -= *(initialVelocity->vector());
    *(AnalyticPressure->vector()) -= *(initialPressure->vector());

    L2Error::Form_M L2error1(mesh,AnalyticCation);
    double cationError = assemble(L2error1);
    L2Error::Form_M L2error2(mesh,AnalyticAnion);
    double anionError = assemble(L2error2);
    L2Error::Form_M L2error3(mesh,AnalyticPotential);
    double potentialError = assemble(L2error3);
    L2ErrorRT::Form_M L2error4(mesh,AnalyticVelocity);
    double velocityError = assemble(L2error4);
    L2ErrorDG::Form_M L2error5(mesh,AnalyticPressure);
    double pressurelError = assemble(L2error5);

    printf("***********************************************\n");
    printf("***********************************************\n");
    printf("\tcation l2 error is:     %e\n", cationError);
    printf("\tanion l2 error is:      %e\n", anionError);
    printf("\tpotential l2 error is:  %e\n", potentialError);
    printf("\tVelocity l2 error is:      %e\n", velocityError);
    printf("\tPressure l2 error is:  %e\n", pressurelError);
    printf("***********************************************\n");
    printf("***********************************************\n");

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
