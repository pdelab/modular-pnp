#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "domain.h"
#include "dirichlet.h"
#include "mesh_refiner.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "L2Error.h"
#include "SemiH1error.h"
#include "gradient_recovery.h"
#include "poisson_cell_marker.h"

//--------------------------------
Mesh_Refiner::Mesh_Refiner (
  const std::shared_ptr<const dolfin::Mesh> initial_mesh,
  const std::size_t max_elements_in,
  const std::size_t max_refine_depth_in,
  const double entropy_per_cell
) {
  _mesh.reset(new const dolfin::Mesh(*initial_mesh));

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  iteration = 0;
  Mesh_Refiner::needs_to_solve = true;
  Mesh_Refiner::needs_refinement = false;

  Mesh_Refiner::max_elements = max_elements_in;
  Mesh_Refiner::max_refine_depth = max_refine_depth_in;
  Mesh_Refiner::entropy_tolerance_per_cell = entropy_per_cell;
};
//--------------------------------
Mesh_Refiner::~Mesh_Refiner () {};
//--------------------------------

//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::get_mesh () {
  return _mesh;
};
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::multilevel_refinement (
  std::shared_ptr<const dolfin::Function> entropy_potential
) {
  return Mesh_Refiner::recursive_refinement(
    entropy_potential,
    Mesh_Refiner::entropy_tolerance_per_cell,
    0
  );
}
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::recursive_refinement (
  std::shared_ptr<const dolfin::Function> entropy_potential,
  double entropy_tolerance,
  std::size_t depth
) {

  if (depth > Mesh_Refiner::max_refine_depth || _mesh->num_cells() > Mesh_Refiner::max_elements) {
    printf("Mesh refinement is attempting to over-refine...\n");
    Mesh_Refiner::needs_to_solve = depth > 0;
    Mesh_Refiner::needs_refinement = false;
    return _mesh;
  }

  // mark cells and see if cells were marked
  Mesh_Refiner::mark_for_refinement(entropy_potential, entropy_tolerance);
  if (!Mesh_Refiner::needs_refinement) {
    Mesh_Refiner::needs_to_solve = depth > 0;
    return _mesh;
  }

  // count cells in resulting mesh
  Mesh_Refiner::needs_to_solve = true;
  auto temp_mesh = std::make_shared<dolfin::Mesh>(*_mesh);
  auto adapted_mesh = dolfin::adapt(*temp_mesh, *_cell_marker);
  std::size_t adapted_mesh_size = adapted_mesh->num_cells();
  bool too_many_resulting_cells = adapted_mesh_size > Mesh_Refiner::max_elements;

  if (too_many_resulting_cells) {
    return Mesh_Refiner::recursive_refinement(entropy_potential, 2.5 * entropy_tolerance, depth);
  }

  _mesh = adapted_mesh;
  return Mesh_Refiner::recursive_refinement(entropy_potential, entropy_tolerance, depth + 1);
}
//--------------------------------
std::size_t Mesh_Refiner::mark_for_refinement (
  std::shared_ptr<const dolfin::Function> entropy_potential,
  double entropy_tolerance
) {
  // setup forms for gradient recovery
  auto gradient_space = std::make_shared<gradient_recovery::FunctionSpace>(_mesh);
  gradient_recovery::BilinearForm bilinear_lumping(gradient_space, gradient_space);
  gradient_recovery::LinearForm gradient_form(gradient_space);

  auto potential_interpolant = std::make_shared<dolfin::Function>(
    dolfin::adapt(*(entropy_potential->function_space()), _mesh)
  );
  potential_interpolant->interpolate(*entropy_potential);
  gradient_form.potential = potential_interpolant;

  gradient_form.weight = std::make_shared<dolfin::Constant>(1.0);

  auto recovery_matrix = std::make_shared<dolfin::EigenMatrix>();
  dolfin::assemble(*recovery_matrix, bilinear_lumping);

  auto recovery_vector = std::make_shared<dolfin::EigenVector>(
    potential_interpolant->vector()->mpi_comm(),
    recovery_matrix->size(0)
  );
  dolfin::assemble(*recovery_vector, gradient_form);
  auto entropy = std::make_shared<dolfin::Function>(gradient_space);
  Mesh_Refiner::mass_lumping_solver(recovery_matrix, recovery_vector, entropy);

  // compute entropic error
  auto DG = std::make_shared<poisson_cell_marker::FunctionSpace>(_mesh);
  poisson_cell_marker::LinearForm error_form(DG);
  error_form.entropy_potential = potential_interpolant;
  error_form.entropy = entropy;
  dolfin::EigenVector error_vector;
  dolfin::assemble(error_vector, error_form);

  // mark cells according to entropic error
  std::size_t marked_count = 0;
  _cell_marker.reset(
    new dolfin::MeshFunction<bool>(_mesh, _mesh->topology().dim(), false)
  );
  for (std::size_t index = 0; index < error_vector.size(); index++) {
    if (error_vector[index] > entropy_tolerance) {
      _cell_marker->set_value(index, true);
      marked_count++;
    }
  }

  Mesh_Refiner::needs_refinement = marked_count > 0 ? true : false;
  return marked_count;
};
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::refine_mesh () {
  auto refined_mesh = dolfin::adapt(*_mesh, *_cell_marker);
  _mesh = refined_mesh;

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  Mesh_Refiner::needs_refinement = false;
  Mesh_Refiner::needs_to_solve = true;
  return _mesh;
};
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::refine_uniformly () {
  auto refined_mesh = dolfin::adapt(*_mesh);
  _mesh = refined_mesh;

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  Mesh_Refiner::needs_refinement = false;
  Mesh_Refiner::needs_to_solve = true;
  return _mesh;
};
//--------------------------------
void Mesh_Refiner::mass_lumping_solver (
  std::shared_ptr<dolfin::EigenMatrix> A,
  std::shared_ptr<dolfin::EigenVector> b,
  std::shared_ptr<dolfin::Function> solution
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
    for(j = IA[i]; j < IA[i+1]; j++)
      row_lump += A_vals[j];

    if (row_lump == 0.0) {
      printf("Mass lumping detected a zero row-sum\n"); fflush(stdout);
      return;
    }

    soln_vals[i] = rhs_vals[i] / row_lump;
  }

  solution->vector()->set_local(soln_vals);
}
//--------------------------------
