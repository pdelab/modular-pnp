/*! \file domains.cpp
 *
 * \brief Mesh related functions
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include "newton.h"
#include "gradient_recovery.h"
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
 * \param domain_par  parameters describing mesh
 * \param mesh      mesh to be constructed
 * \param subdomains  mesh subdomains to be constructed
 * \param surfaces    mesh surfaces to be constructed
 */
void domain_build (domain_param *domain_par,
                   dolfin::Mesh *mesh,
                   dolfin::MeshFunction<size_t> *subdomains,
                   dolfin::MeshFunction<size_t> *surfaces)
{
  // no mesh provided: use length and grid parameters
  if ( strcmp(domain_par->mesh_file,"none")==0 ) {
      fflush(stdout);

      // mesh
      dolfin::Point p0( -domain_par->length_x/2, -domain_par->length_y/2, -domain_par->length_z/2);
      dolfin::Point p1(  domain_par->length_x/2,  domain_par->length_y/2,  domain_par->length_z/2);
      dolfin::BoxMesh box_mesh(p0, p1, domain_par->grid_x, domain_par->grid_y, domain_par->grid_z);
      *mesh = box_mesh;

      // subdomains
      dolfin::CellFunction<std::size_t> subdomains_object(*mesh);
      subdomains_object.set_all(1);
      *subdomains = subdomains_object;

      // surfaces
      dolfin::FacetFunction<std::size_t> surfaces_object(*mesh);
      surfaces_object.set_all(1);
      *surfaces = surfaces_object;
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
}

/**
 * \fn bool check_local_entropy (dolfin::Function *cation,
 *                               dolfin::Function *anion,
 *                               dolfin::Function *voltage,
 *                               dolfin::Mesh *target_mesh,
 *                               double entropy_tol)
 *
 * \brief Check if local entropy is below tolerance and refine
 *    mesh
 *
 * \param cation      cation function
 * \param anion       anion function
 * \param voltage     voltage function
 * \param mesh        ptr to refined mesh
 * \param entropy_tol mesh surfaces to be constructed
 */
bool check_local_entropy (dolfin::Function *cation,
                          dolfin::Function *anion,
                          dolfin::Function *voltage,
                          dolfin::Mesh *target_mesh,
                          double entropy_tol)
{
  // compute mesh from input voltage function and transfer
  printf("\tinitialize mesh, spaces, and forms...\n"); fflush(stdout);
  dolfin::Mesh mesh( *(voltage->function_space()->mesh()) );
  gradient_recovery::FunctionSpace gradient_space(mesh);
  gradient_recovery::BilinearForm a(gradient_space,gradient_space);
  gradient_recovery::LinearForm L(gradient_space);
  dolfin::Function cation_entropy(gradient_space);
  dolfin::Function anion_entropy(gradient_space);

  // compute entropic potentials
  printf("\tcompute entropy potentials...\n"); fflush(stdout);
  dolfin::Function cation_potential( *(voltage->function_space()) );
  cation_potential.interpolate(*voltage);
  *(cation_potential.vector()) *= +1.0;
  *(cation_potential.vector()) += *(cation->vector());
  dolfin::Function anion_potential( *(voltage->function_space()) );
  anion_potential.interpolate(*voltage);
  *(anion_potential.vector()) *= -1.0;
  *(anion_potential.vector()) += *(anion->vector());

  // compute entropy
  printf("\tset form...\n"); fflush(stdout);
  L.potential = cation_potential;
  printf("\tsolve for cation entropy...\n"); fflush(stdout);
  solve(a==L, cation_entropy);

  printf("\tset form...\n"); fflush(stdout);
  L.potential = anion_potential;
  printf("\tsolve for anion entropy...\n"); fflush(stdout);
  solve(a==L, anion_entropy);

  File entropyFile("./benchmarks/PNP/output/entropy.pvd");
  entropyFile << cation_entropy;
  entropyFile << anion_entropy;

  *target_mesh = refine(mesh);
  return true;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
