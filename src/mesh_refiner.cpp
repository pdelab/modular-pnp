#include <iostream>
#include <fstream>
#include <algorithm>
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
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector
) {
  std::size_t num_cells = _mesh->num_cells();

  printf("Entering mesh adaptation routine\n");
  auto refined_mesh = Mesh_Refiner::recursive_refinement(
    diffusivity_vector,
    entropy_potential_vector,
    entropy_log_weight_vector,
    Mesh_Refiner::entropy_tolerance_per_cell,
    0
  );

  if (num_cells == refined_mesh->num_cells()) {
    printf("Refinement algorithm propsed no refinement\n");
    Mesh_Refiner::needs_refinement = false;
    Mesh_Refiner::needs_to_solve = false;
  }

  return refined_mesh;
}
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::recursive_refinement (
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector,
  double entropy_tolerance,
  std::size_t depth
) {
  std::size_t grown_mesh_size = 5 * _mesh->num_cells();
  std::size_t max_element_iterate = std::min(grown_mesh_size, Mesh_Refiner::max_elements);

  if (depth > Mesh_Refiner::max_refine_depth || _mesh->num_cells() > max_element_iterate) {
    printf("\nMesh refinement is attempting to over-refine...\n");
    Mesh_Refiner::needs_to_solve = depth > 0;
    Mesh_Refiner::needs_refinement = false;
    return _mesh;
  }

  // mark cells and see if cells were marked
  Mesh_Refiner::mark_for_refinement(
    diffusivity_vector,
    entropy_potential_vector,
    entropy_log_weight_vector,
    entropy_tolerance
  );
  if (!Mesh_Refiner::needs_refinement) {
    Mesh_Refiner::needs_to_solve = depth > 0;
    return _mesh;
  }

  // count cells in resulting mesh
  Mesh_Refiner::needs_to_solve = true;
  auto temp_mesh = std::make_shared<dolfin::Mesh>(*_mesh);
  auto adapted_mesh = dolfin::adapt(*temp_mesh, *_cell_marker);
  std::size_t adapted_mesh_size = adapted_mesh->num_cells();
  bool accept_refinement = adapted_mesh_size < (max_element_iterate + 1);

  if (accept_refinement) {
    _mesh = adapted_mesh;
    return Mesh_Refiner::recursive_refinement(
      diffusivity_vector,
      entropy_potential_vector,
      entropy_log_weight_vector,
      entropy_tolerance,
      depth + 1
    );
  }

  // aim for a twenty percent update in mesh size
  printf("\tmesh refinement is too aggressive... ");
  printf("mark elements to have proportional refinement\n");
  auto conservative_mesh = std::make_shared<dolfin::Mesh>(*_mesh);
  std::size_t target_size = max_element_iterate;

  while (!accept_refinement) {
    // decrement target size of mesh
    target_size = (std::size_t) std::round(0.95 * ((double) target_size));
    if (target_size < _mesh->num_cells() + 1) { break; }

    Mesh_Refiner::mark_for_refinement_with_target_size(
      diffusivity_vector,
      entropy_potential_vector,
      entropy_log_weight_vector,
      target_size
    );

    dolfin::Mesh conservative_temp_mesh(*_mesh);
    conservative_mesh = dolfin::adapt(conservative_temp_mesh, *_cell_marker);
    accept_refinement = conservative_mesh->num_cells() < (max_element_iterate + 1);
  }

  _mesh.reset(new dolfin::Mesh(*conservative_mesh));
  return _mesh;

}
//--------------------------------
std::size_t Mesh_Refiner::mark_for_refinement_with_target_size (
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector,
  std::size_t target_size
) {
  // set target cell count
  std::size_t permissible_cells_to_add = target_size - _mesh->num_cells();
  std::size_t permissible_cells = std::round(((double) permissible_cells_to_add) / 6.0);

  // compute error vector of interpolant
  dolfin::EigenVector error_eigenvector = Mesh_Refiner::compute_entropy_error_vector(
    diffusivity_vector,
    entropy_potential_vector,
    entropy_log_weight_vector
  );

  // sort errors and estimate corresponding entropy_tolerance
  std::vector<double> error_vector;
  for (std::size_t i = 0; i < error_eigenvector.size(); i++) {
    error_vector.push_back(error_eigenvector[i]);
  }
  std::sort(error_vector.begin(), error_vector.end());
  int toleranceIndex = error_vector.size() - permissible_cells;
  toleranceIndex = toleranceIndex < 0 ? 0 : toleranceIndex;
  const double entropy_tolerance = error_vector[toleranceIndex];

  // mark cells according to entropic error
  std::size_t marked_count = 0;
  _cell_marker.reset( new dolfin::MeshFunction<bool>(_mesh, _mesh->topology().dim(), false) );

  for (std::size_t index = 0; index < error_eigenvector.size(); index++) {
    if (error_eigenvector[index] > entropy_tolerance) {
      _cell_marker->set_value(index, true);
      marked_count++;
    }
  }

  Mesh_Refiner::needs_refinement = marked_count > 0 ? true : false;
  return marked_count;
};
//--------------------------------
std::size_t Mesh_Refiner::mark_for_refinement (
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector,
  double entropy_tolerance
) {
  // compute error vector of interpolant
  dolfin::EigenVector error_vector = Mesh_Refiner::compute_entropy_error_vector(
    diffusivity_vector,
    entropy_potential_vector,
    entropy_log_weight_vector
  );

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
dolfin::EigenVector Mesh_Refiner::compute_entropy_error_vector (
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector
) {
  // setup forms for gradient recovery
  auto gradient_space = std::make_shared<gradient_recovery::FunctionSpace>(_mesh);
  gradient_recovery::BilinearForm bilinear_lumping(gradient_space, gradient_space);
  gradient_recovery::LinearForm gradient_form(gradient_space);

  // loop over subfunctions of entropy potential
  dolfin::EigenVector error_vector;
  std::size_t component_count = entropy_potential_vector.size();

  dolfin::File entropy_error_file("./entropy_error.pvd");

  for (std::size_t comp = 0; comp < component_count; comp++) {
    auto potential_interpolant = std::make_shared<dolfin::Function>(
      dolfin::adapt(*(entropy_potential_vector[comp]->function_space()), _mesh)
    );
    potential_interpolant->interpolate( *(entropy_potential_vector[comp]) );
    gradient_form.potential = potential_interpolant;

    auto log_weight_interpolant = std::make_shared<dolfin::Function>(
      dolfin::adapt(*(entropy_log_weight_vector[comp]->function_space()), _mesh)
    );
    log_weight_interpolant->interpolate( *(entropy_log_weight_vector[comp]) );;
    // gradient_form.log_weight = log_weight_interpolant;

    auto diffusivity_interpolant = std::make_shared<dolfin::Function>(
      dolfin::adapt(*(diffusivity_vector[comp]->function_space()), _mesh)
    );
    diffusivity_interpolant->interpolate( *(diffusivity_vector[comp]) );
    // gradient_form.diffusivity = diffusivity_interpolant;

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
    poisson_cell_marker::BilinearForm error_bilinear_form(DG, DG);
    poisson_cell_marker::LinearForm error_linear_form(DG);
    error_linear_form.entropy_potential = potential_interpolant;

    // <<< test out using only the entropy and not the entropy error
    dolfin::Constant zeros(0.0, 0.0, 0.0);
    auto zeros_ptr = std::make_shared<dolfin::Constant>(zeros);
    error_linear_form.entropy = zeros_ptr;
    // error_linear_form.entropy = entropy;
    error_linear_form.diffusivity = diffusivity_interpolant;
    error_linear_form.log_weight = log_weight_interpolant;

    if (error_vector.size() < 1) {
      dolfin::assemble(error_vector, error_linear_form);
      entropy_error_file << Mesh_Refiner::as_function(DG, error_vector);
    } else {
      dolfin::EigenVector local_error_vector;
      dolfin::assemble(local_error_vector, error_linear_form);
      error_vector += local_error_vector;
      entropy_error_file << Mesh_Refiner::as_function(DG, local_error_vector);
    }
  }

  return error_vector;
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
dolfin::Function Mesh_Refiner::as_function(
  std::shared_ptr<dolfin::FunctionSpace> function_space,
  dolfin::EigenVector vec
) {
  double* values;
  values = vec.data();
  std::vector<double> std_vec;
  std_vec.reserve(vec.size());
  for (uint i = 0; i < vec.size(); i++)
    std_vec[i] = values[i];

  dolfin::Function function(function_space);
  function.vector()->set_local(std_vec);

  return function;
}
