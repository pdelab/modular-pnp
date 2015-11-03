/*! \file test_eafe.cpp
 *
 *  \brief Main to test EAFE functionality on linear convection reaction problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
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

int main()
{

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n Solving the linearized Poisson-Nernst-Planck system           "); fflush(stdout);
  printf("\n of a single cation and anion                                  "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

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



  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
