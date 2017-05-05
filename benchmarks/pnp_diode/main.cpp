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

#include "diode.h"
#include "vector_linear_pnp_forms.h"
#include "pnp_newton_solver.h"

#include "cross_section_surface_area_forms.h"
#include "cross_section_surface_current_forms.h"

using namespace std;

// helper functions for marking cells for refinement
std::vector<std::shared_ptr<const dolfin::Function>> get_diode_diffusivity(
  std::shared_ptr<const dolfin::FunctionSpace> function_space
);

std::vector<std::shared_ptr<const dolfin::Function>> extract_log_densities (
  std::shared_ptr<dolfin::Function> solution
);
std::vector<std::shared_ptr<const dolfin::Function>> compute_entropy_potential(
  std::shared_ptr<dolfin::Function> solution
);

// cross-section for estimating the current
double computeCurrentFlux(
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity,
  std::vector<std::shared_ptr<const dolfin::Function>> log_density,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential
);

class CrossSection : public dolfin::SubDomain {
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
    return (fabs(x[0] - 0.5) < 1.0e-8) or (fabs(x[0] + 0.5) < 1.0e-8);
  }
};

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

  double max_volts = 0.1;
  double delta_volts = 0.1;
  dolfin::File accepted_solution_file("./benchmarks/pnp_diode/output/accepted_solution.pvd");

  for (double voltage_drop = -max_volts; voltage_drop < max_volts + 1.e-5; voltage_drop += delta_volts) {
    printf("Solving for voltage drop : %5.2e\n\n", voltage_drop);

    // parameters for mesh adaptivity
    double growth_factor = 1.2;
    double entropy_per_cell = 5.0e-3;
    std::size_t max_refine_depth = 4;
    std::size_t max_elements = 250000;
    Mesh_Refiner mesh_adapt(
      initial_mesh,
      max_elements,
      max_refine_depth,
      entropy_per_cell
    );

    // parameters for PNP Newton solver
    const std::size_t max_newton = 50;
    const double max_residual_tol = 1.0e-10;
    const double relative_residual_tol = 1.0e-4;
    const bool use_eafe_approximation = true;
    std::shared_ptr<double> initial_residual_ptr = std::make_shared<double>(-1.0);

    // construct initial guess
    double induced_current;
    Initial_Guess initial_guess_expression(voltage_drop);
    auto adaptive_solution = std::make_shared<dolfin::Function>(
      std::make_shared<vector_linear_pnp_forms::FunctionSpace>(mesh_adapt.get_mesh())
    );
    adaptive_solution->interpolate(initial_guess_expression);

    dolfin::File initial_guess_file("./benchmarks/pnp_diode/output/initial_guess.pvd");
    while (mesh_adapt.needs_to_solve) {
      auto mesh = mesh_adapt.get_mesh();

      initial_guess_file << *adaptive_solution;

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

      // compute current / entropy terms
      auto diffusivity = get_diode_diffusivity(computed_solution->function_space());
      auto entropy_potential = compute_entropy_potential(computed_solution);
      auto log_densities = extract_log_densities(computed_solution);

      // Compute current flux through cross section
      induced_current = computeCurrentFlux(diffusivity, log_densities, entropy_potential);

      // adapt computed solutions
      mesh_adapt.max_elements = (std::size_t) std::floor(growth_factor * mesh->num_cells());
      mesh_adapt.multilevel_refinement(diffusivity, entropy_potential, log_densities);
      adaptive_solution = adapt( *computed_solution, mesh_adapt.get_mesh() );
    }


    printf("\nCompleted adaptivity loop for %eV with induced current %emA\n\n", voltage_drop, induced_current);
    accepted_solution_file << *adaptive_solution;
  }

  return 0;
}



/**
 * Helper functions for marking elements in need of refinement
 */
std::vector<std::shared_ptr<const dolfin::Function>> get_diode_diffusivity(
  std::shared_ptr<const dolfin::FunctionSpace> function_space
) {
  // get analytic diffusivity
  dolfin::Function diffusivity(function_space);
  Diffusivity_Expression diff_expr;
  diffusivity.interpolate(diff_expr);

  // transfer to vector of functions
  std::size_t component_count = function_space->element()->num_sub_elements();
  std::vector<std::shared_ptr<const dolfin::Function>> function_vec;
  for (std::size_t comp = 1; comp < component_count; comp++) {
    auto subfunction_space = diffusivity[comp].function_space()->collapse();
    dolfin::Function diffusivity_comp(subfunction_space);
    diffusivity_comp.interpolate(diffusivity[comp]);

    auto const_diffusivity_ptr = std::make_shared<const dolfin::Function>(diffusivity_comp);
    function_vec.push_back(const_diffusivity_ptr);
  }

  return function_vec;
}


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


/**
 * Compute the current determined by the finite element solution
 */
double computeCurrentFlux(
  std::vector<std::shared_ptr<const dolfin::Function>> diffusivity,
  std::vector<std::shared_ptr<const dolfin::Function>> log_density,
  std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential
) {
  auto mesh_ptr = log_density[0]->function_space()->mesh();
  dolfin::FacetFunction<std::size_t> cross_section_facets(mesh_ptr);

  // define cross section
  CrossSection cross_section;
  cross_section_facets.set_all(0);
  cross_section.mark(cross_section_facets, 1);

  dolfin::File cross_section_file("./benchmarks/pnp_diode/output/cross_section.pvd");
  cross_section_file << cross_section_facets;

  // compute average current flux through the surface
  cross_section_surface_area_forms::Functional surface_area_form(mesh_ptr);
  const dolfin::Constant area_scale(1.0);
  surface_area_form.scale = std::make_shared<dolfin::Constant>(area_scale);
  surface_area_form.dS = std::make_shared<dolfin::FacetFunction<std::size_t>>(cross_section_facets);
  const double surface_area = assemble(surface_area_form);

  // compute relevant functions
  cross_section_surface_current_forms::Functional current_form(mesh_ptr);
  const dolfin::Constant n_vector(1.0, 0.0, 0.0);
  current_form.normal_vector = std::make_shared<dolfin::Constant>(n_vector);
  current_form.dS = std::make_shared<dolfin::FacetFunction<std::size_t>>(cross_section_facets);
  current_form.cation_diff = diffusivity[0];
  current_form.anion_diff = diffusivity[1];
  current_form.log_cation = log_density[0];
  current_form.log_anion = log_density[1];
  current_form.cation_flux = entropy_potential[0];
  current_form.anion_flux = entropy_potential[1];
  const double current = assemble(current_form);

  // scaling
  const double elementary_charge = 1.60217662e-19; // C
  const double reference_length = 1e-6; // m
  const double reference_diffusivity = 2.87e-3; // m^2 / s
  const double reference_density = 1.5e+22; // mM = 1 / m^3
  const double microamp_scale_factor = 1.0e+6 * elementary_charge * reference_diffusivity * reference_density * reference_length;

  printf("current flux: %5.3e mA\n", microamp_scale_factor * current / surface_area);
  return microamp_scale_factor * current / surface_area;
}
