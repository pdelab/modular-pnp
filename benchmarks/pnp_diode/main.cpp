/// Main file for solving the linearized PNP problem
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <dolfin.h>
#include "mesh_refiner.h"
#include "domain.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "pnp_newton_solver.h"

using namespace std;

// helper functions for marking cells for refinement
std::vector<std::shared_ptr<const dolfin::Function>> extract_log_densities (
  std::shared_ptr<dolfin::Function> solution
);
std::vector<std::shared_ptr<const dolfin::Function>> compute_entropy_potential(
  std::shared_ptr<dolfin::Function> solution
);

// the main body of the script
int main (int argc, char** argv) {
  printf("\n");
  printf("----------------------------------------------------\n");
  printf(" Setting up the linearized PNP problem\n");
  printf("----------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  dolfin::parameters["linear_algebra_backend"] = "Eigen";
  dolfin::parameters["allow_extrapolation"] = true;

  // Deleting the folders:
  boost::filesystem::remove_all("./benchmarks/pnp_diode/output");

  // read in parameters
  printf("Reading parameters from files...\n");
  char domain_param_filename[] = "./benchmarks/pnp_diode/domain.dat";
  printf("\tdomain... %s\n", domain_param_filename);
  domain_param domain;
  domain_param_input(domain_param_filename, &domain);
  std::shared_ptr<dolfin::Mesh> initial_mesh;
  initial_mesh.reset(new dolfin::Mesh(domain_build(domain)));
  // print_domain_param(&domain);

  // set parameters for FASP solver
  char fasp_params[] = "./benchmarks/pnp_diode/bsr.dat";
  printf("\tFASP parameters... %s\n", fasp_params);
  itsolver_param itsolver;
  input_param input;
  AMG_param amg;
  ILU_param ilu;
  fasp_param_input(fasp_params, &input);
  fasp_param_init(&input, &itsolver, &amg, &ilu, NULL);

  //-------------------------
  // Mesh Adaptivity Loop
  //-------------------------

  // parameters for mesh adaptivity
  double growth_factor = 1.2;
  double entropy_per_cell = 1.0e-6;
  std::size_t max_refine_depth = 3;
  std::size_t max_elements = 750000;
  Mesh_Refiner mesh_adapt(
    initial_mesh,
    max_elements,
    max_refine_depth,
    entropy_per_cell
  );

  // parameters for PNP Newton solver
  const std::size_t max_newton = 100;
  const double max_residual_tol = 1.0e-10;
  const double relative_residual_tol = 1.0e-8;
  const bool use_eafe_approximation = true;
  std::shared_ptr<double> initial_residual_ptr = std::make_shared<double>(-1.0);

  // construct initial guess
  Initial_Guess initial_guess_expression;
  auto adaptive_solution = std::make_shared<dolfin::Function>(
    std::make_shared<vector_linear_pnp_forms::FunctionSpace>(mesh_adapt.get_mesh())
  );
  adaptive_solution->interpolate(initial_guess_expression);

  dolfin::File initial_guess_file("./benchmarks/pnp_diode/output/initial_guess.pvd");
  initial_guess_file << *adaptive_solution;

  while (mesh_adapt.needs_to_solve) {
    auto mesh = mesh_adapt.get_mesh();
    auto computed_solution = solve_pnp(
      mesh_adapt.iteration++,
      mesh,
      adaptive_solution,
      max_newton,
      max_residual_tol,
      relative_residual_tol,
      initial_residual_ptr,
      use_eafe_approximation,
      itsolver,
      amg,
      "./benchmarks/pnp_diode/output/"
    );

    // compute entropy terms to mark cells for refinement
    auto entropy_potential = compute_entropy_potential(computed_solution);
    auto log_densities = extract_log_densities(computed_solution);

    mesh_adapt.max_elements = (std::size_t) std::floor( growth_factor * mesh->num_cells() );
    mesh_adapt.multilevel_refinement(entropy_potential, log_densities);

    // update solution
    adaptive_solution.reset( new dolfin::Function(computed_solution->function_space()) );
    adaptive_solution->interpolate(*computed_solution);
    initial_guess_file << *adaptive_solution;
  }

  printf("\nCompleted adaptivity loop\n\n");
  return 0;
}

/**
 * Helper functions for marking elements in need of refinement
 */
std::vector<std::shared_ptr<const dolfin::Function>> extract_log_densities (
  std::shared_ptr<dolfin::Function> solution
) {
  std::size_t component_count = solution->function_space()->element()->num_sub_elements();

  std::vector<std::shared_ptr<const dolfin::Function>> function_vec;
  for (std::size_t comp = 1; comp < component_count; comp++) {
    auto subfunction_space = (*solution)[comp].function_space()->collapse();
    dolfin::Function log_density(subfunction_space);
    log_density.interpolate((*solution)[comp]);

    auto const_log_density = std::make_shared<const dolfin::Function>(log_density);
    function_vec.push_back(const_log_density);
  }
  return function_vec;
}

std::vector<std::shared_ptr<const dolfin::Function>> compute_entropy_potential (
  std::shared_ptr<dolfin::Function> solution
) {
  std::vector<std::shared_ptr<const dolfin::Function>> function_vec;
  std::size_t component_count = solution->function_space()->element()->num_sub_elements();

  for (std::size_t comp = 1; comp < component_count; comp++) {
    auto subfunction_space = (*solution)[comp].function_space()->collapse();
    dolfin::Function potential(subfunction_space);
    dolfin::Function entropy_potential(subfunction_space);

    potential.interpolate((*solution)[0]);
    entropy_potential.interpolate((*solution)[comp]);
    *(potential.vector()) *= valencies[comp];
    entropy_potential = entropy_potential + potential;

    auto const_entropy_potential = std::make_shared<const dolfin::Function>(entropy_potential);
    function_vec.push_back(const_entropy_potential);
  }

  return function_vec;
}
