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


class CationExp : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 1.0;
  }
};
class AnionExp : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 2.0;
  }
};
class PhiExp : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 3.0;
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


  // Function space for PNP (Cation,Anion,Phi)
  linear_pnp::FunctionSpace V(mesh);

  // Test on init function
  Function initFunc(V);
  Constant C1(1.0);
  CationExp Cation;
  AnionExp Anion;
  PhiExp Phi;
  Function initCat(initFunc[0]); initCat.interpolate(Cation);
  Function initAn(initFunc[1]);  initAn.interpolate(Anion);
  Function initPHI(initFunc[2]); initPHI.interpolate(Phi);

  // PNP Formulation
  printf("Linearized PNP formluation...");
  linear_pnp::BilinearForm a_pnp(V,V);
  linear_pnp::LinearForm L_pnp(V);
  a_pnp.CatCat  = initCat; L_pnp.CatCat  = initCat;
  a_pnp.AnAn    = initAn;  L_pnp.AnAn    = initAn;
  a_pnp.EsEs    = initPHI; L_pnp.EsEs    = initPHI;
  a_pnp.CatDiff = initCat;
  a_pnp.AnDiff  = initAn;
  a_pnp.eps = C1; L_pnp.eps = C1;
  a_pnp.Dp  = C1; L_pnp.Dp  = C1;
  a_pnp.qp  = C1; L_pnp.qp  = C1;
  a_pnp.Dn  = C1; L_pnp.Dn  = C1;
  a_pnp.qn  = C1; L_pnp.qn  = C1;
  L_pnp.fix = C1;
  EigenMatrix A_pnp;
  EigenVector b_pnp;
  assemble(A_pnp,a_pnp); assemble(b_pnp,L_pnp);
  printf("done\n");

  // EAFE Formulation for Anion and Cation
  printf("EAFE formluation...");
  FunctionSpace V_cat = SubSpace(V,0);
  FunctionSpace V_an = SubSpace(V,1);
  EAFE::BilinearForm a_cat(V_cat,V_cat);
  EAFE::LinearForm L_cat(V_cat);
  EAFE::BilinearForm a_an(V_an,V_an);
  EAFE::LinearForm L_an(V_an);
  a_cat.eta  = initCat;
  a_cat.beta = initPHI;
  a_cat.alpha = C1;
  a_cat.gamma = C1;
  L_cat.f= C1;
  EigenVector b_cat;
  assemble(b_cat,L_cat);

  printf("done\n");

  // Mapping of the dof
  printf("Mapping of the dof...\n");
  int n = V.dim();
  int d = mesh.geometry().dim();
  int n_cat = V_cat.dim();
  int n_an = V_an.dim();
  printf("\tGeometric dimension = %d\n",d);
  printf("\tV number of DOF = %d\n",n);
  printf("\tV_cat number of DOF = %d\n",n_cat);
  printf("\tV_an number of DOF = %d\n",n_an);
  printf("\tb_pnp size = %d\n",b_pnp.size());
  printf("\tb_an size = %d\n",b_cat.size());
  printf("\tdone\n"); fflush(stdout);






  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
