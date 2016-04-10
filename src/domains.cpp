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
#include "poisson_cell_marker.h"
#include "electric_cell_marker.h"
#include "fasp_to_fenics.h"
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
void domain_build (
  domain_param *domain_par,
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


INT mass_lumping_solver (
  dolfin::EigenMatrix* A,
  dolfin::EigenVector* b,
  dolfin::Function* solution
) {
  uint i, j;
  uint row_entries;
  uint col_index;
  double row_lump;

  // point to Eigenmatrix
  int *IA;
  uint rows = A->size(0);
  double* A_vals, b_vals, x_vals;
  IA = (int*) std::get<0>(A->data());
  A_vals = (double*) std::get<2>(A->data());

  double* rhs_vals;
  rhs_vals = b->data();

  std::vector<double> soln_vals;
  soln_vals.reserve(rows);

  for (i = 0; i < rows; i++) {

    row_lump = 0.0;
    for(j = IA[i]; j < IA[i+1]; j++) {
      row_lump += A_vals[j];
    }
    if (row_lump == 0.0) { return -48; }

    soln_vals[i] = rhs_vals[i] / row_lump;
  }

  solution->vector()->set_local(soln_vals);

  return FASP_SUCCESS;
}

/**
 * \fn bool check_local_entropy (dolfin::Function *cation,
 *                               dolfin::Function *anion,
 *                               dolfin::Function *voltage,
 *                               dolfin::Mesh *target_mesh,
 *                               double entropy_tol)
 *
 * \brief Check if local entropy is below tolerance and refine mesh
 *
 * \param cation          cation function
 * \param cation_valency  valency of cation
 * \param anion           anion function
 * \param anion_valency   valency of anion
 * \param voltage         voltage function
 * \param mesh            pionter to refined mesh
 * \param entropy_tol     tolerance for local entropy
 * \param max_elements    maximum size for mesh
 *
 * \return                levels of refinement
 */
unsigned int check_local_entropy (
  dolfin::Function *cation,
  double cation_valency,
  dolfin::Function *anion,
  double anion_valency,
  dolfin::Function *voltage,
  dolfin::Mesh *target_mesh,
  double entropy_tol,
  uint max_elements,
  uint max_depth
) {
  // compute mesh from input voltage function and transfer
  dolfin::Mesh mesh( *(voltage->function_space()->mesh()) );

  // refinement is too deep
  if (max_depth == 0) {
    printf("\tmesh refinement is attempting to over-refine...\n");
    printf("\t\t terminating refinement\n");
    *target_mesh = mesh;
    return -1;
  }

  gradient_recovery::FunctionSpace gradient_space(mesh);
  gradient_recovery::BilinearForm a(gradient_space,gradient_space);
  gradient_recovery::LinearForm L(gradient_space);
  dolfin::Function cation_entropy(gradient_space);
  dolfin::Function anion_entropy(gradient_space);

  // compute entropic potentials
  dolfin::Function cation_potential( *(voltage->function_space()) );
  cation_potential.interpolate(*voltage);
  *(cation_potential.vector()) *= cation_valency;
  *(cation_potential.vector()) += *(cation->vector());
  dolfin::Function anion_potential( *(voltage->function_space()) );
  anion_potential.interpolate(*voltage);
  *(anion_potential.vector()) *= anion_valency;
  *(anion_potential.vector()) += *(anion->vector());

  // setup matrix and rhs
  EigenMatrix A;
  assemble(A,a);
  EigenVector b(A.size(0));
  INT status = FASP_SUCCESS;

  // set form for cation
  L.eta = *cation;
  L.potential = cation_potential;
  assemble(b,L);
  status = mass_lumping_solver(&A, &b, &cation_entropy);

  // set form for anion
  L.eta = *anion;
  L.potential = anion_potential;
  assemble(b,L);
  status = mass_lumping_solver(&A, &b, &anion_entropy);

  // compute entropic error
  poisson_cell_marker::FunctionSpace DG(mesh);
  poisson_cell_marker::LinearForm error_form(DG);
  error_form.cat_entr = cation_entropy;
  error_form.cat_pot  = cation_potential;
  error_form.an_entr = anion_entropy;
  error_form.an_pot = anion_potential;
  dolfin::EigenVector error_vector;
  assemble(error_vector, error_form);

  // mark for refinement
  MeshFunction<bool> cell_marker(mesh, mesh.topology().dim(), false);
  double new_entropy_tol = entropy_tol;
  unsigned int marked_elem_count = 0;
  for (int errVecInd = 0; errVecInd < error_vector.size(); errVecInd++) {
    if (error_vector[errVecInd] > new_entropy_tol) {
      marked_elem_count++;
      cell_marker.values()[errVecInd] = true;
      if (marked_elem_count > max_elements) {
        cell_marker.values()[errVecInd] = false;
        new_entropy_tol *= 5.0;
        cell_marker.set_all(false);
        marked_elem_count = 0;
        errVecInd = -1;
        printf("\tOver-refined! Adjusting tolerance to %e\n", new_entropy_tol);
      }
    }
    else {
      cell_marker.values()[errVecInd] = false;
    }

  }

  // check for necessary refiments
  if ( marked_elem_count == 0 ) {
    *target_mesh = mesh;
    return 0;
  }
  else {
    std::shared_ptr<const Mesh> mesh_ptr(new const Mesh(refine(mesh, cell_marker)));
    dolfin::FunctionSpace adapt_function_space( adapt(*(voltage->function_space()), mesh_ptr) );

    // adapt functions
    dolfin::Function adapt_cation(adapt_function_space);
    dolfin::Function adapt_anion(adapt_function_space);
    dolfin::Function adapt_voltage(adapt_function_space);
    adapt_cation.interpolate(*cation);
    adapt_anion.interpolate(*anion);
    adapt_voltage.interpolate(*voltage);

    int num_refines = 0;
    num_refines = check_local_entropy(
      &adapt_cation,
      cation_valency,
      &adapt_anion,
      anion_valency,
      &adapt_voltage,
      target_mesh,
      new_entropy_tol,
      max_elements,
      max_depth - 1
    );
    return 1 + num_refines;
  }
}

/**
 * \fn bool check_electric_field (dolfin::Function *voltage,
 *                               dolfin::Mesh *target_mesh,
 *                               double entropy_tol)
 *
 * \brief Check if the electric field gradient(potential) is below tolerance and refine
 *    mesh
 *
 * \param voltage         voltage function
 * \param mesh            ptr to refined mesh
 * \param entropy_tol     tolerance for local entropy
 * \param max_elements    maximum size for mesh
 *
 * \return                levels of refinement
 */
unsigned int check_electric_field (
  dolfin::Function *voltage,
  dolfin::Mesh *target_mesh,
  double entropy_tol,
  uint max_elements,
  uint max_depth
) {
  // compute mesh from input voltage function and transfer
  dolfin::Mesh mesh( *(voltage->function_space()->mesh()) );

  // refinement is too deep
  if (max_depth == 0) {
    printf("\tmesh refinement is attempting to over-refine...\n");
    printf("\t\t terminating refinement\n");
    *target_mesh = mesh;
    return -1;
  }

  gradient_recovery::FunctionSpace gradient_space(mesh);
  gradient_recovery::BilinearForm a(gradient_space,gradient_space);
  gradient_recovery::LinearForm L(gradient_space);
  dolfin::Function ElecField(gradient_space);

  // compute entropic potentials
  dolfin::Function potential( *(voltage->function_space()) );
  potential.interpolate(*voltage);

  // setup matrix and rhs
  EigenMatrix A;
  assemble(A,a);
  EigenVector b(A.size(0));
  INT status = FASP_SUCCESS;

  // set form for electric field
  Constant one(1.0);
  L.eta = one;
  L.potential = potential;
  assemble(b,L);
  status = mass_lumping_solver(&A, &b, &ElecField);

  // compute entropic error
  electric_cell_marker::FunctionSpace DG(mesh);
  electric_cell_marker::LinearForm error_form(DG);
  error_form.pot = potential;
  error_form.gradpot  = ElecField;
  dolfin::EigenVector error_vector;
  assemble(error_vector, error_form);

  // mark for refinement
  MeshFunction<bool> cell_marker(mesh, mesh.topology().dim(), false);
  double new_entropy_tol = entropy_tol;
  unsigned int marked_elem_count = 0;
  for ( uint errVecInd = 0; errVecInd < error_vector.size(); errVecInd++) {
    if (error_vector[errVecInd] > new_entropy_tol) {
      marked_elem_count++;
      cell_marker.values()[errVecInd] = true;

      if (marked_elem_count > max_elements) {
        cell_marker.values()[errVecInd] = false;
        new_entropy_tol *= 5.0;
        cell_marker.set_all(false);
        marked_elem_count = 0;
        errVecInd = -1;
        printf("\tOver-refined! Adjusting tolerance to %e\n", new_entropy_tol);
      }
    }
    else {
      cell_marker.values()[errVecInd] = false;
    }
  }

  // check for necessary refiments
  if ( marked_elem_count == 0 ) {
    *target_mesh = mesh;
    return 0;
  }
  else {
    std::shared_ptr<const Mesh> mesh_ptr( new const Mesh(refine(mesh, cell_marker)) );
    dolfin::FunctionSpace adapt_function_space( adapt(*(voltage->function_space()), mesh_ptr) );

    // adapt functions
    dolfin::Function adapt_voltage( adapt_function_space );
    adapt_voltage.interpolate(*voltage);

    int num_refines = 0;
    num_refines =  check_electric_field(
      &adapt_voltage,
      target_mesh,
      entropy_tol,
      max_elements,
      max_depth - 1
    );
    return 1 + num_refines;
  }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
