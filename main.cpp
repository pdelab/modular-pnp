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
	char domain_param_filename[] = "./params/voltage_benchmark/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  printf("Successfully read-in domain parameters\n");
  if ( strcmp(domain_par.mesh_file,"none")==0 ) {
    printf("\tDomain: %f x %f x %f\n",  domain_par.length_x,domain_par.length_y,domain_par.length_z);
    printf("\tGrid:   %d x %d x %d\n\n",domain_par.grid_x,domain_par.grid_y,domain_par.grid_z);
  } else {
    printf("\tMesh file:      %s\n",  domain_par.mesh_file);
    printf("\tSubdomain file: %s\n",  domain_par.subdomain_file);
    printf("\tSurface file:   %s\n\n",domain_par.surface_file);
  }

	// build mesh
  Mesh mesh;
  MeshFunction<std::size_t> *subdomains;
  MeshFunction<std::size_t> *surfaces;

  // domain_build(&domain_par, &mesh, subdomains, surfaces);

/*
	// no mesh provided: use length and grid parameters
	domain_param *dom_par; dom_par = &domain_par;
	Mesh *meshh; meshh = &mesh;
	MeshFunction<std::size_t> *subdoms; subdoms = subdomains;
  MeshFunction<std::size_t> *surfs; surfs = surfaces;

	if ( strcmp(dom_par->mesh_file,"none")==0 ) {
      printf("\tDomain: %f x %f x %f\n",dom_par->length_x,dom_par->length_y,dom_par->length_z);
      printf("\tGrid: %d x %d x %d\n",dom_par->grid_x,dom_par->grid_y,dom_par->grid_z);

      BoxMesh box_mesh(-dom_par->length_x/2,-dom_par->length_y/2,-dom_par->length_z/2,dom_par->length_x/2,dom_par->length_y/2,dom_par->length_z/2, dom_par->grid_x, dom_par->grid_y, dom_par->grid_z);
      meshh = &box_mesh;

      MeshFunction<std::size_t> subdomains_object(box_mesh);
      subdomains_object.set_all(1);
      subdoms = &subdomains_object;

      MeshFunction<std::size_t>  surfaces_object(box_mesh);
      surfaces_object.set_all(1);
      surfs = &surfaces_object;
    } 
    else { // read in mesh from specified files
      printf(" Reading in the mesh from %s \n", dom_par->mesh_file);
      Mesh read_mesh(dom_par->mesh_file);
      meshh = &read_mesh;

      printf(" Reading in the mesh subdomains from %s \n", dom_par->subdomain_file);
      MeshFunction<std::size_t> subdomains_object(read_mesh, dom_par->subdomain_file);
      subdoms = &subdomains_object;
      
      printf(" Reading in the mesh surfaces from %s \n", dom_par->surface_file);
      MeshFunction<std::size_t>  surfaces_object(read_mesh, dom_par->surface_file);
      surfs = &surfaces_object;
    }

    // mesh = meshh;
    subdomains = subdoms;
    surfaces = surfs;
*/



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
	char newton_param_filename[] = "./params/voltage_benchmark/newton_params.dat";
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