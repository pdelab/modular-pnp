/*! \file test_reading.h
 *
 *  \brief Main to test the reading functions
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>
#include "newton.h"
#include "newton_functs.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
    INT fasp_solver_bdcsr_krylov_block_3(block_dCSRmat *A,
                                         dvector *b,
                                         dvector *x,
                                         itsolver_param *itparam,
                                         AMG_param *amgparam,
                                         dCSRmat *A_diag);
#define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace std;
using namespace dolfin;

int main()
{
	// output problem specification
  printf("\n---------------------------------------------    "); fflush(stdout);
  printf("\n This code simulates an electrostatic system     "); fflush(stdout);
  printf("\n By solving the Poisson-Nernst-Planck system     "); fflush(stdout);
  printf("\n of a single cation and anion                    "); fflush(stdout);
  printf("\n---------------------------------------------\n\n"); fflush(stdout);

	/**
	 * Setup domain and subregions
	 */

	// read domain parameters
	domain_param domain_par;
	char domain_param_filename[] = "./problems/test_reading/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

	// build mesh
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);

	// adaptivity?


  /**
   * Initialize differential equation
   */

  // read coefficients and boundary values
  coeff_param coeff_par;
  char coeff_param_filename[] = "./problems/test_reading/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  print_coeff_param(&coeff_par);

	// dimensional analysis

  // initialize boundary conditions

	// initialize finite elements

	// initialize analytic expressions


  /**
   * Initialize variational problem
   */

	// initialize bilinear, linear, and scalar forms

	// interpolate analytic expressions

	// assign coefficients to forms


  /**
   * Newton solver
   */

	// read Newton solver parameters
	newton_param newton_par;
	char newton_param_filename[] = "./problems/test_reading/newton_params.dat";
  newton_param_input(newton_param_filename, &newton_par);
  print_newton_param(&newton_par);

	// compute initial nonlinear residual

	// build linearized system

	// solve linear system

	// update nonlinear solution

	// check for convergence


  printf("The code has run successfully... exiting.\n");
	return 0;
}
