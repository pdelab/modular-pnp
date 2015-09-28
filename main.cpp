/*! \file pnp.h
 *
 *  \brief Main file for PNP solver using Newton-Raphson and AMG methods.
 *		   Based on FASP and FEniCS 1.5 packages
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>
#include "./include/newton.h"
#include "./include/newton_functs.h"
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

	/*-------------------------------------*/
	/*--   Setup domain and subregions   --*/
	/*-------------------------------------*/

	// read domain parameters
	domain_param domain_par;
	char domain_param_filename[] = "./problems/voltage_benchmark/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  printf("Successfully read-in domain parameters\n");
  if ( strcmp(domain_par.mesh_file,"none")==0 ) {
    printf("\tDomain: %f x %f x %f\n",  domain_par.length_x,domain_par.length_y,domain_par.length_z);
    printf("\tGrid:   %d x %d x %d\n\n",domain_par.grid_x,domain_par.grid_y,domain_par.grid_z);
    fflush(stdout);
  } else {
    printf("\tMesh file:      %s\n",  domain_par.mesh_file);
    printf("\tSubdomain file: %s\n",  domain_par.subdomain_file);
    printf("\tSurface file:   %s\n\n",domain_par.surface_file);
    fflush(stdout);
  }

	// build mesh
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);

  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);
  // meshOut << mesh; meshOut << subdomains; meshOut << surfaces;

	// adaptivity?

	// initialize boundary conditions


	/*-------------------------------*/
	/*--   Initialize expressions	 --*/
	/*-------------------------------*/

	// dimensional analysis

	// initialize finite elements

	// initialize analytic expressions


	/*---------------------------------------*/
	/*--   Initialize variational problem  --*/
	/*---------------------------------------*/

	// initialize bilinear, linear, and scalar forms

	// interpolate analytic expressions

	// assign coefficients to forms


	/*-----------------------*/
	/*--   Newton solver   --*/
	/*-----------------------*/

	// read Newton solver parameters
	newton_param newton_par;
	char newton_param_filename[] = "./problems/voltage_benchmark/newton_params.dat";
  newton_param_input(newton_param_filename, &newton_par);
  printf("Successfully read-in Newton solver parameters\n");
  printf("\tNewton Maximum iterations: %d\n",newton_par.max_it);
  printf("\tNewton tolerance: %e\n",newton_par.tol);
  printf("\tNewton damping factor: %f\n",newton_par.damp_factor);
  printf("\n");

	// compute initial nonlinear residual

	// build linearized system

	// solve linear system

	// update nonlinear solution

	// check for convergence


  printf("The code has run successfully... exiting.\n");
	return 0;
}