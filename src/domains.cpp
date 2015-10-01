/*! \file domains.cpp
 *
 * \brief Mesh related functions
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include "newton.h"
extern "C"
{
#include "fasp.h"
#include "fasp_const.h"
}

using namespace std;
using namespace dolfin;

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn REAL newton_compute_residual (dvector *residual)
 *
 * \brief Compute the L2 norm of the residual vector
 *
 * \param domain_par 	parameters describing mesh
 * \param mesh 		 	mesh to be constructed
 * \param subdomains 	mesh subdomains to be constructed
 * \param surfaces	 	mesh surfaces to be constructed
 */
void domain_build (domain_param *domain_par,
				           dolfin::Mesh *mesh,
				           dolfin::MeshFunction<size_t> *subdomains,
				           dolfin::MeshFunction<size_t> *surfaces,
                   dolfin::File *mesh_output)
{

  printf("Constructing the mesh and subregions\n"); fflush(stdout);

	// no mesh provided: use length and grid parameters
	if ( strcmp(domain_par->mesh_file,"none")==0 ) {
      printf("\tDomain: %f x %f x %f\n",domain_par->length_x,domain_par->length_y,domain_par->length_z);
      printf("\tGrid: %d x %d x %d\n",domain_par->grid_x,domain_par->grid_y,domain_par->grid_z);
      fflush(stdout);

      // mesh
      dolfin::BoxMesh box_mesh(
        -domain_par->length_x/2,
        -domain_par->length_y/2,
        -domain_par->length_z/2,
        domain_par->length_x/2,
        domain_par->length_y/2,
        domain_par->length_z/2,
        domain_par->grid_x,
        domain_par->grid_y,
        domain_par->grid_z
      );
      *mesh = box_mesh;

      // subdomains
      dolfin::CellFunction<std::size_t> subdomains_object(*mesh);
      subdomains_object.set_all(1);
      *subdomains = subdomains_object;

      // surfaces
      dolfin::FacetFunction<std::size_t> surfaces_object(*mesh);
      surfaces_object.set_all(1);
      *surfaces = surfaces_object;

      printf("\tConstructed the mesh\n"); fflush(stdout);
    } 
    else { // read in mesh from specified files
      printf("### ERROR: Reading in meshes is currently unsupported: %s...\n\n", domain_par->mesh_file);
    }
   /* else { // read in mesh from specified files
      printf(" Reading in the mesh from %s \n", domain_par->mesh_file);
      dolfin::Mesh read_mesh(domain_par->mesh_file);
      *mesh = read_mesh;

      printf(" Reading in the mesh subdomains from %s \n", domain_par->subdomain_file);
      dolfin::MeshFunction<std::size_t> subdomains_object(*mesh, domain_par->subdomain_file);
      *subdomains = subdomains_object;
      
      printf(" Reading in the mesh surfaces from %s \n", domain_par->surface_file);
      dolfin::MeshFunction<std::size_t>  surfaces_object(*mesh, domain_par->surface_file);
      *surfaces = surfaces_object;
    }*/

    *mesh_output << *mesh;
    *mesh_output << *subdomains;
    *mesh_output << *surfaces;

}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
