/*! \file test_eafe.cpp
 *
 *  \brief Main to test EAFE functionality on linear convection reaction problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "linear_pnp.h"
#include "newton.h"
#include "newton_functs.h"
#include "funcspace_to_vecspace.h"
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


int main()
{

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n Solving the linearized Poisson-Nernst-Planck system           "); fflush(stdout);
  printf("\n of a single cation and anion                                  "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  // read domain parameters
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/PNP/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);

  // read coefficients and boundary values
  coeff_param coeff_par, non_dim_coeff_par;
  char coeff_param_filename[] = "./benchmarks/PNP/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  print_coeff_param(&coeff_par);
  non_dimesionalize_coefficients(&domain_par, &coeff_par, &non_dim_coeff_par);


  //*************************************************************
  //  Analytic Expressions
  //*************************************************************
  printf("\n Initializing analytic expressions\n"); fflush(stdout);

  // Reference values
  double k_B    = 1.38064880e-23;     // Boltzmann Constant (m^2 kg / s^2 K)
  double e_chrg = 1.60217657e-19 ;    // Elementary Positive Charge (A s = C)
  double p_ref  = 1.0e+0;             // ionic reference density (mol / m^3)

  double temperature=coeff_par.temperature;
  double int_voltage = 0.0 / (k_B*temperature/e_chrg);      // dim'less internal contact voltage
  double ext_voltage = 0.0 / (k_B*temperature/e_chrg);      // dim'less external contact voltage
  double int_cat_bulk = 6.6e+9 / p_ref;                             // dim'less internal contact cation
  double ext_cat_bulk = 6.6e+9  / p_ref;                             // dim'less external contact cation
  double int_an_bulk  = 1.5e+22 / p_ref;                             // dim'less internal contact anion
  double ext_an_bulk  = 1.5e+22  / p_ref;                             // dim'less external contact anion

  // Log-ion boundary interpolant
  printf("   Interpolating contact values for charge carriers \n"); fflush(stdout);
  LogCharge Cation(ext_cat_bulk, int_cat_bulk,-domain_par.length_x/2.0,domain_par.length_x/2.0, 0);
  LogCharge Anion(ext_an_bulk, int_an_bulk,-domain_par.length_x/2.0, domain_par.length_x/2.0, 0);

  // Electric potential boundary interpolant
  printf("   Interpolating voltage drop\n"); fflush(stdout);
  Voltage volt(ext_voltage, ext_voltage,-domain_par.length_x/2.0, domain_par.length_x/2.0, 0);

  //*************************************************************
  //*************************************************************

  // Function space for PNP (Cation=log(Concentration of positive charges),Anion=log(Concentration of negative charges),Phi=Voltage)
  linear_pnp::FunctionSpace V(mesh);

  // Test on init function
  Function initFunc(V);
  Constant C1(1.0);
  Function initCat(initFunc[0]); initCat.interpolate(Cation);
  Function initAn(initFunc[1]);  initAn.interpolate(Anion);
  Function initPHI(initFunc[2]); initPHI.interpolate(volt);

  Constant Eps(non_dim_coeff_par.relative_permittivity);
  Constant Dp(non_dim_coeff_par.cation_diffusivity);
  Constant Dn(non_dim_coeff_par.anion_diffusivity);
  Constant qn(non_dim_coeff_par.cation_mobility);
  Constant qp(non_dim_coeff_par.anion_mobility);
  Constant fix(1.5E22);

  // PNP Formulation
  printf("Linearized PNP formluation...");
  linear_pnp::BilinearForm a_pnp(V,V);
  linear_pnp::LinearForm L_pnp(V);
  a_pnp.CatCat  = initCat; L_pnp.CatCat  = initCat;
  a_pnp.AnAn    = initAn;  L_pnp.AnAn    = initAn;
  a_pnp.EsEs    = initPHI; L_pnp.EsEs    = initPHI;
  a_pnp.CatDiff = initCat;
  a_pnp.AnDiff  = initAn;
  a_pnp.eps = Eps; L_pnp.eps = Eps;
  a_pnp.Dp  = Dp ; L_pnp.Dp  = Dp;
  a_pnp.qp  = qp ; L_pnp.qp  = qp;
  a_pnp.Dn  =Dn; L_pnp.Dn  = Dn;
  a_pnp.qn  = qn; L_pnp.qn  = qn;
  L_pnp.fix = C1;
  EigenMatrix A_pnp;
  EigenVector b_pnp;
  assemble(A_pnp,a_pnp); assemble(b_pnp,L_pnp);
  printf("done\n");

  // EAFE Formulation for Anion and Cation
  printf("EAFE formluation...");
  EAFE::FunctionSpace V_cat(mesh);
  EAFE::FunctionSpace V_an(mesh);
  EAFE::BilinearForm a_cat(V_cat,V_cat);
  EAFE::LinearForm L_cat(V_cat);
  EAFE::BilinearForm a_an(V_an,V_an);
  EAFE::LinearForm L_an(V_an);
  Function initCat_cat(V_cat); Function Phi_cat(V_cat);
  initCat_cat.interpolate(Cation); Phi_cat.interpolate(volt);
  Function initAn_an(V_an); Function Phi_an(V_an);
  initAn_an.interpolate(Anion); Phi_an.interpolate(volt);
  a_cat.eta  = initCat_cat;
  a_cat.beta = Phi_cat;
  a_cat.alpha = C1;
  a_cat.gamma = C1;
  L_cat.f= C1;
  EigenMatrix A_cat; EigenVector b_cat;
  assemble(b_cat,L_cat);
  assemble(A_cat,a_cat);
  a_an.eta  = initAn_an;
  a_an.beta = Phi_an;
  a_an.alpha = C1;
  a_an.gamma = C1;
  L_an.f= C1;
  EigenMatrix A_an; EigenVector b_an;
  assemble(A_an,a_an);
  assemble(b_an,L_an);
  printf("done\n");

  // Dimensions of the problems
  printf("Print of sizes...\n");
  int n = V.dim();
  int d = mesh.geometry().dim();
  int n_cat = V_cat.dim();
  int n_an = V_an.dim();
  printf("\tGeometric dimension = %d\n",d);
  printf("\tV number of DOF = %d\n",n);
  printf("\tV_cat number of DOF = %d\n",n_cat);
  printf("\tV_an number of DOF = %d\n",n_an);
  printf("\tA_pnp size = %ld x %ld\n",A_pnp.size(0),A_pnp.size(1));
  printf("\tA_cat size = %ld x %ld\n",A_cat.size(0),A_cat.size(1));
  printf("\tA_an size = %ld x %ld\n",A_an.size(0),A_an.size(1));
  printf("\tb_pnp size = %ld\n",b_pnp.size());
  printf("\tb_an size = %ld\n",b_cat.size());
  printf("\tb_an size = %ld\n",b_an.size());
  fflush(stdout);

  add_matrix(0, &V, &V_cat, &A_pnp, &A_cat);
  add_matrix(1, &V, &V_an, &A_pnp, &A_an);

  // Convert to fasp
  dCSRmat A_fasp;
  dvector b_fasp;
  dvector Solu_fasp;
  EigenVector_to_dvector(&b_pnp,&b_fasp);
  EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
  fasp_dvec_alloc(b_fasp.row, &Solu_fasp);
  fasp_dvec_set(b_fasp.row, &Solu_fasp, 0.0);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./benchmarks/PNP/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  status = fasp_solver_dcsr_krylov(&A_fasp, &b_fasp, &Solu_fasp, &itpar);


  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
