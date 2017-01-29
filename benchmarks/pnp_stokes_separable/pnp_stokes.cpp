/*! \file pnp_adaptive.cpp
 *
 *  \brief Setup and solve the PNP equations using adaptive meshing and FASP
 *
 *  \note Currently initializes the problem based on specification
 */

 #include <cinttypes>
 #include <cmath>
 #include <cstdlib>
 #include <map>
 #include <utility>
 #include <ufc.h>

 #include <map>
 #include <set>
 #include <string>
 #include <vector>
 #include <boost/multi_array.hpp>
 #include <memory>
 #include <unordered_map>

#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "pnp_stokes.h"
#include "pnp_stokes12.h"
#include "pnp_stokes21.h"
#include "pnp_stokes22.h"
#include "newton.h"
#include "newton_functs.h"
#include "L2Error.h"
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
  char domain_param_filename[] = "./benchmarks/pnp_stokes_separable/domain_params.dat";
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
  File MeshFileXML("./benchmarks/pnp_stokes_separable/XML/mesh.xml");
  MeshFileXML << *mesh;

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par, non_dim_coeff_par;
  char coeff_param_filename[] = "./benchmarks/pnp_stokes_separable/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  // non_dimesionalize_coefficients(&domain_par, &coeff_par, &non_dim_coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/pnp_stokes_separable/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  // Setup FASP solver
  printf("FASP solver parameters..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/pnp_stokes_separable/bcsr.dat";
  fasp_param_input(fasp_params, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  printf("done\n"); fflush(stdout);

  // Setup FASP solver for pnp
  printf("FASP solver parameters for pnp..."); fflush(stdout);
  input_param pnp_inpar;
  itsolver_param pnp_itpar;
  AMG_param  pnp_amgpar;
  ILU_param pnp_ilupar;
  Schwarz_param pnp_schpar;
  char fasp_pnp_params[] = "./benchmarks/pnp_stokes_separable/bsr.dat";
  fasp_param_input(fasp_pnp_params, &pnp_inpar);
  fasp_param_init(&pnp_inpar, &pnp_itpar, &pnp_amgpar, &pnp_ilupar, &pnp_schpar);
  printf("done\n"); fflush(stdout);

  // Setup FASP solver for stokes
  printf("FASP solver parameters for stokes..."); fflush(stdout);
  input_ns_param stokes_inpar;
  itsolver_ns_param stokes_itpar;
  AMG_ns_param  stokes_amgpar;
  ILU_param stokes_ilupar;
  Schwarz_param stokes_schpar;
  char fasp_ns_params[] = "./benchmarks/pnp_stokes_separable/ns.dat";
  fasp_ns_param_input(fasp_ns_params, &stokes_inpar);
  fasp_ns_param_init(&stokes_inpar, &stokes_itpar, &stokes_amgpar, &stokes_ilupar, &stokes_schpar);
  printf("done\n"); fflush(stdout);

  // open files for outputting solutions
  File cationFile("./benchmarks/pnp_stokes_separable/output/cation.pvd");
  File anionFile("./benchmarks/pnp_stokes_separable/output/anion.pvd");
  File potentialFile("./benchmarks/pnp_stokes_separable/output/potential.pvd");
  File velocityFile("./benchmarks/pnp_stokes_separable/output/velocity.pvd");
  File pressureFile("./benchmarks/pnp_stokes_separable/output/pressure.pvd");


  File cationFileXML("./benchmarks/pnp_stokes_separable/XML/cation.xml");
  File anionFileXML("./benchmarks/pnp_stokes_separable/XML/anion.xml");
  File potentialFileXML("./benchmarks/pnp_stokes_separable/XML/potential.xml");
  File velocityFileXML("./benchmarks/pnp_stokes_separable/XML/velocity.xml");
  File pressureFileXML("./benchmarks/pnp_stokes_separable/XML/pressure.xml");

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
  auto penalty1=std::make_shared<Constant>(1.0);
  auto penalty2=std::make_shared<Constant>(1.0);


  // interpolate
  auto V_PNP = std::make_shared<pnp_stokes::FunctionSpace>(mesh);
  auto initialFunctionPNP = std::make_shared<Function>(V_PNP);
  auto initialCation = std::make_shared<Function>((*initialFunctionPNP)[0]);
  auto initialAnion = std::make_shared<Function>((*initialFunctionPNP)[1]);
  auto initialPotential = std::make_shared<Function>((*initialFunctionPNP)[2]);
  auto V_S = std::make_shared<pnp_stokes22::FunctionSpace>(mesh);
  auto initialFunctionS = std::make_shared<Function>(V_S);
  auto initialVelocity = std::make_shared<Function>((*initialFunctionS)[0]);
  auto initialPressure = std::make_shared<Function>((*initialFunctionS)[1]);


  // std::string s1("./benchmarks/pnp_stokes/XML_2/cation.xml");
  // std::string s2("./benchmarks/pnp_stokes/XML_2/anion.xml");
  // std::string s3("./benchmarks/pnp_stokes/XML_2/potential.xml");
  // std::string s4("./benchmarks/pnp_stokes/XML_2/velocity.xml");
  // std::string s5("./benchmarks/pnp_stokes/XML_2/pressure.xml");
  //
  // std::shared_ptr<FunctionSpace> VC = V->sub(0);
  // std::shared_ptr<FunctionSpace> CG = VC->collapse();
  // VC = V->sub(3);
  // std::shared_ptr<FunctionSpace> RT = VC->collapse();
  // VC = V->sub(4);
  // std::shared_ptr<FunctionSpace> DG = VC->collapse();
  // // auto V1 = std::make_shared<pnp_stokes::Form_a_FunctionSpace_2>(mesh);
  //
  // auto initialCation = std::make_shared<Function>(CG,s1);
  // auto initialAnion = std::make_shared<Function>(CG,s2);
  // auto initialPotential = std::make_shared<Function>(CG,s3);
  // auto initialVelocity = std::make_shared<Function>(RT,s4);
  // auto initialPressure = std::make_shared<Function>(DG,s5);

  auto one_vec3 = std::make_shared<Constant>(1.0,1.0,1.0);
  auto vel_vec = std::make_shared<Constant>(0.0,0.0,0.0);

  initialCation->interpolate(Cation);
  // initialAnion->interpolate(Cation);
  initialAnion->interpolate(Anion);
  // initialPotential->interpolate(*zero);
  initialPotential->interpolate(Volt);
  initialVelocity->interpolate(*vel_vec);
  initialPressure->interpolate(*zero);


  // set adaptivity parameters
  double entropy_tol = newtparam.adapt_tol;
  unsigned int num_adapts = 0, max_adapts = 5;
  bool adaptive_convergence = false;

  // output mesh
  meshOut << *mesh;

  // Initialize variational forms
  printf("\tvariational forms...\n"); fflush(stdout);
  pnp_stokes::BilinearForm a(V_PNP,V_PNP);
  pnp_stokes::LinearForm L(V_PNP);
  a.eps = eps; L.eps = eps;
  a.Dp = Dp; L.Dp = Dp;
  a.Dn = Dn; L.Dn = Dn;
  a.qp = qp; L.qp = qp;
  a.qn = qn; L.qn = qn;

  pnp_stokes12::BilinearForm a12(V_S,V_PNP);

  pnp_stokes22::BilinearForm a22(V_S,V_S);
  pnp_stokes22::LinearForm L22(V_S);
  a22.mu = mu; L22.mu = mu;
  a22.alpha1 = penalty1; L22.alpha1 = penalty1;
  a22.alpha2 = penalty2; L22.alpha2 = penalty2;
  L22.eps = eps;

  pnp_stokes21::BilinearForm a21(V_PNP,V_S);
  a21.eps = eps;

  // Set Dirichlet boundaries
  printf("\tboundary conditions...\n"); fflush(stdout);
  auto boundary = std::make_shared<SymmBoundaries>(coeff_par.bc_coordinate, -domain_par.length_x/2.0, domain_par.length_x/2.0);
  auto bddd = std::make_shared<Bd_all>();
  dolfin::DirichletBC bc(V_PNP, zero_vec3, boundary);
  dolfin::DirichletBC bc2(V_PNP->sub(2), zero, boundary);
  dolfin::DirichletBC bc_stokes(V_S->sub(0), zero_vec3, boundary);
  // dolfin::DirichletBC bc_stokes(Vs, zero_vec4, bddd);

  std::unordered_map<std::size_t, double> boundary_values;
  bc2.get_boundary_values(boundary_values);
  const std::size_t sizePNP = boundary_values.size();
  std::vector<dolfin::la_index> dofsPNP(sizePNP);
  std::unordered_map<std::size_t, double>::const_iterator bv;
  std::size_t counter = 0;
  for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
  {
    dofsPNP[counter++]     = bv->first;
  }
  bc_stokes.get_boundary_values(boundary_values);
  const std::size_t sizeS= boundary_values.size();
  std::vector<dolfin::la_index> dofsS(sizeS);
  counter = 0;
  for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
  {
    dofsS[counter++]     = bv->first;
  }
 // A->zero_local(size, dofs.data());

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
  get_dofs(initialFunctionPNP.get(), &cation_dofs, 0);
  get_dofs(initialFunctionPNP.get(), &anion_dofs, 1);
  get_dofs(initialFunctionPNP.get(), &potential_dofs, 2);
  ivector velocity_dofs;
  ivector pressure_dofs;
  get_dofs(initialFunctionS.get(), &velocity_dofs, 0);
  get_dofs(initialFunctionS.get(), &pressure_dofs, 1);
  int index_fix = pressure_dofs.val[0];

  // output dofs
    /*
    fasp_ivec_write("cation_dofs", &cation_dofs);
    fasp_ivec_write("anion_dofs", &anion_dofs);
    fasp_ivec_write("potential_dofs", &potential_dofs);
    fasp_ivec_write("velocity_dofs", &velocity_dofs);
    fasp_ivec_write("pressure_dofs", &pressure_dofs);
     */


  // initialize linear system
  printf("\tlinear algebraic objects...\n"); fflush(stdout);
  EigenMatrix A, A12, A21,A22;
  EigenVector b,b2;
  // Fasp matrices and vectors
  dCSRmat A_fasp,A_fasp12,A_fasp21,A_fasp22;
  dvector b_fasp, b_fasp2, solu_fasp, solu_fasp2;

  block_dCSRmat A_fasp_bcsr;
  dvector b_fasp_bcsr, solu_fasp_bcsr;

    // allocate
    int i;
    A_fasp_bcsr.brow = 2;
    A_fasp_bcsr.bcol = 2;
    A_fasp_bcsr.blocks = (dCSRmat **)calloc(4, sizeof(dCSRmat *));
    fasp_mem_check((void *)A_fasp_bcsr.blocks, "block matrix:cannot allocate memory!\n", ERROR_ALLOC_MEM);
    for (i=0; i<4 ;i++) {
        A_fasp_bcsr.blocks[i] = (dCSRmat *)fasp_mem_calloc(1, sizeof(dCSRmat));
    }

    // form dof index for PNP and Stokes
    ivector pnp_dofs;
    ivector stokes_dofs;
    fasp_ivec_alloc(potential_dofs.row + cation_dofs.row + anion_dofs.row, &pnp_dofs);
    fasp_ivec_alloc(velocity_dofs.row + pressure_dofs.row, &stokes_dofs);

    // potential
    for (i=0; i<potential_dofs.row; i++)
        pnp_dofs.val[3*i] = potential_dofs.val[i];
    // cation
    for (i=0; i<cation_dofs.row; i++)
        pnp_dofs.val[3*i+1] = cation_dofs.val[i];
    // anion
    for (i=0; i<anion_dofs.row; i++)
        pnp_dofs.val[3*i+2] = anion_dofs.val[i];
    // velocity
    for (i=0; i<velocity_dofs.row; i++)
        stokes_dofs.val[i] = velocity_dofs.val[i];
    // pressure
    for (i=0; i<pressure_dofs.row; i++)
        stokes_dofs.val[velocity_dofs.row+i] = pressure_dofs.val[i];

  //*************************************************************
  //  Initialize Newton solver
  //*************************************************************
  // Setup newton parameters and compute initial residual
  printf("\tNewton solver initialization...\n"); fflush(stdout);
  // auto solutionUpdate = std::make_shared<Function>(V);
  auto solutionUpdatePNP = std::make_shared<dolfin::Function>(V_PNP);
  auto solutionUpdateS = std::make_shared<dolfin::Function>(V_S);
  unsigned int newton_iteration = 0;

  printf("\tInitial residual...\n"); fflush(stdout);
  L.CatCat = initialCation;
  L.AnAn = initialAnion;
  L.EsEs = initialPotential;
  L.uu = initialVelocity;
  assemble(b, L);
  bc.apply(b);

  L22.EsEs = initialPotential;
  L22.uu = initialVelocity;
  L22.pp = initialPressure;
  assemble(b2, L22);
  bc_stokes.apply(b2);
  b2[index_fix]=0.0;

  initial_residual = b.norm("l2")+b2.norm("l2");
  relative_residual =1.0;
  printf("Initial rez = %e\n",initial_residual);
  fasp_dvec_alloc(b.size(), &solu_fasp);
  fasp_dvec_alloc(b2.size(), &solu_fasp2);

  // return 0;


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

    assemble(A22, a22);

    a12.CatCat = initialCation;
    a12.AnAn = initialAnion;
    assemble(A12, a12);

    a21.EsEs = initialPotential;
    assemble(A21, a21);

    bc.apply(A);
    bc_stokes.apply(A22);
    // A12.zero_local(sizePNP, dofsPNP.data());
    A21.zero_local(sizeS, dofsS.data());
    A12.zero_local(sizePNP, dofsPNP.data());


    // bc_stokes.apply(A12);

    // bc_stokes.apply(A22);
    replace_row(index_fix, &A22);
    // EigenMatrix A_res(A);
    //
    // dCSRmat A_res_fasp;
    // EigenMatrix_to_dCSRmat(&A_res, &A_res_fasp);

    // Convert to fasp
    printf("\tconvert to FASP...\n"); fflush(stdout);
    EigenVector_to_dvector(&b,&b_fasp);
    EigenVector_to_dvector(&b2,&b_fasp2);
    EigenMatrix_to_dCSRmat(&A,&A_fasp);
    EigenMatrix_to_dCSRmat(&A12,&A_fasp12);
    EigenMatrix_to_dCSRmat(&A21,&A_fasp21);
    EigenMatrix_to_dCSRmat(&A22,&A_fasp22);
    fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
    fasp_dvec_set(b_fasp2.row, &solu_fasp2, 0.0);

    // A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
    // copy_EigenMatrix_to_block_dCSRmat(&A_stokes, &A_stokes_fasp, &velocity_dofs, &pressure_dofs);
    // status = fasp_solver_dcsr_krylov_amg(&A_fasp, &b_fasp, &solu_fasp, &itpar, &amgpar);

    // list_lu_solver_methods();
    // solve(A, *solutionUpdate->vector(), b, "umfpack");
    // 
    // std::cout << A.size(0) << "\t" <<A.size(1) << "\n";
    // std::cout << A12.size(0) << "\t" <<A12.size(1) << "\n";
    // std::cout << A21.size(0) << "\t" <<A21.size(1) << "\n";
    // std::cout << A22.size(0) << "\t" <<A22.size(1) << "\n";

      // --------------------------------------------------------------------------
      // 2 by 2 block solver
      // step 1: get blocks (order: PNP Stokes)
      fasp_dcsr_getblk(&A_fasp, pnp_dofs.val,    pnp_dofs.val,    pnp_dofs.row,    pnp_dofs.row,    A_fasp_bcsr.blocks[0]);
      fasp_dcsr_getblk(&A_fasp12, pnp_dofs.val,    stokes_dofs.val, pnp_dofs.row,    stokes_dofs.row, A_fasp_bcsr.blocks[1]);
      fasp_dcsr_getblk(&A_fasp21, stokes_dofs.val, pnp_dofs.val,    stokes_dofs.row, pnp_dofs.row,    A_fasp_bcsr.blocks[2]);
      fasp_dcsr_getblk(&A_fasp22, stokes_dofs.val, stokes_dofs.val, stokes_dofs.row, stokes_dofs.row, A_fasp_bcsr.blocks[3]);
      //  A_fasp_bcsr.blocks[0] = &A_fasp;
      //  A_fasp_bcsr.blocks[1] = &A_fasp12;
      //  A_fasp_bcsr.blocks[2] = &A_fasp21;
      //  A_fasp_bcsr.blocks[3] = &A_fasp22;

      // fasp_dcoo_write("11.txt",A_fasp_bcsr.blocks[0]);

      // step 2: get right hand side
      fasp_dvec_alloc(b_fasp.row+b_fasp2.row, &b_fasp_bcsr);

      for (i=0; i<pnp_dofs.row; i++)
          b_fasp_bcsr.val[i] = b_fasp.val[pnp_dofs.val[i]];
      for (i=0; i<stokes_dofs.row; i++)
          b_fasp_bcsr.val[pnp_dofs.row + i] = b_fasp2.val[stokes_dofs.val[i]];

      // step 3: solve
      fasp_dvec_alloc(b_fasp.row+b_fasp2.row, &solu_fasp_bcsr);
      fasp_dvec_set(solu_fasp_bcsr.row, &solu_fasp_bcsr, 0.0);
      fasp_solver_bdcsr_krylov_pnp_stokes(&A_fasp_bcsr, &b_fasp_bcsr, &solu_fasp_bcsr, &itpar, &pnp_itpar, &pnp_amgpar, &stokes_itpar, &stokes_amgpar, velocity_dofs.row, pressure_dofs.row);

      // step 4: put solution back
      for (i=0; i<pnp_dofs.row; i++)
          solu_fasp.val[pnp_dofs.val[i]] = solu_fasp_bcsr.val[i];
      for (i=0; i<stokes_dofs.row;    i++)
          solu_fasp2.val[stokes_dofs.val[i]] = solu_fasp_bcsr.val[pnp_dofs.row + i] ;
      // --------------------------------------------------------------------------

    /*
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
     */
    //
    // printf("\t status = %d\n",status);
    //
    //
    // // fasp_dvec_print(b_fasp.row,&b_fasp);
    //   /*
    // char filemat[20] = "matrix.txt";
    // char filevec[20] = "rhs.txt";
    // fasp_dcoo_write(filemat,&A_fasp);
    // fasp_dvec_write (filevec,&b_fasp);
    //   getchar();
    //    */
    //
    //
    // fasp_blas_dcsr_aAxpy(-1.0,
    //                        &A_res_fasp,
    //                        solu_fasp.val,
    //                        b_fasp.val);
    // double rez = fasp_blas_dvec_norm2 (&b_fasp);
    // printf("\t rez = %e , rez_rel = %e\n",rez,rez/initial_residual);
    //
    //
    // printf("\tconvert FASP solution to function...\n"); fflush(stdout);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdatePNP.get(), &cation_dofs, &cation_dofs);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdatePNP.get(), &anion_dofs, &anion_dofs);
    copy_dvector_to_vector_function(&solu_fasp, solutionUpdatePNP.get(), &potential_dofs, &potential_dofs);
    copy_dvector_to_vector_function(&solu_fasp2, solutionUpdateS.get(), &velocity_dofs, &velocity_dofs);
    copy_dvector_to_vector_function(&solu_fasp2, solutionUpdateS.get(), &pressure_dofs, &pressure_dofs);


    printf("\tupdate solution...\n"); fflush(stdout);
    Function dAnion = (*solutionUpdatePNP)[0];
    Function dCation = (*solutionUpdatePNP)[1];
    Function dPotential = (*solutionUpdatePNP)[2];
    Function dU = (*solutionUpdateS)[0];
    Function dPressure = (*solutionUpdateS)[1];
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

    cationFileXML << *initialCation;
    anionFileXML << *initialAnion;
    potentialFileXML << *initialPotential;
    velocityFileXML << *initialVelocity;
    pressureFileXML << *initialPressure;

    // update nonlinear residual
    L.CatCat = initialCation;
    L.AnAn = initialAnion;
    L.EsEs = initialPotential;
    L.uu = initialVelocity;
    assemble(b, L);


    L22.EsEs = initialPotential;
    L22.uu = initialVelocity;
    L22.pp = initialPressure;
    assemble(b2, L22);


    bc.apply(b);
    bc_stokes.apply(b2);
    b2[index_fix]=0.0;


    relative_residual = (b.norm("l2")+b2.norm("l2"))/ initial_residual;
    printf("\t\trel_res after: %e\n", relative_residual);
    if (relative_residual < 0.0) {
      printf("Newton backtracking failed!\n");
      printf("\tresidual has not decreased after damping %d times\n", newtparam.damp_it);
      printf("\tthe relative residual is %e\n", relative_residual);
      relative_residual *= -1.0;

    }
      // cationFile << *initialCation;
      // anionFile << *initialAnion;
      // potentialFile << *initialPotential;
      // velocityFile << *initialVelocity;
      // pressureFile << *initialPressure;

      // cationFile << dCation;
      // anionFile << dAnion;
      // potentialFile << dPotential;
      // velocityFile << dU;
      // pressureFile << dPressure;
    }



  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
