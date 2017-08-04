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
std::vector<std::shared_ptr<const dolfin::Function>> get_diode_diffusivity(
  std::shared_ptr<const dolfin::FunctionSpace> function_space
);

std::vector<std::shared_ptr<const dolfin::Function>> extract_log_densities (
  std::shared_ptr<dolfin::Function> solution
);
std::vector<std::shared_ptr<const dolfin::Function>> compute_entropy_potential(
  std::shared_ptr<dolfin::Function> solution, std::vector<double> valencies
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
  boost::filesystem::remove_all("./benchmarks/physic_bench/output");

  // read in parameters
  printf("Reading parameters from files...\n");
  std::shared_ptr<dolfin::Mesh> initial_mesh;
  initial_mesh.reset(new dolfin::Mesh);
  *initial_mesh = dolfin::Mesh("./benchmarks/physic_bench/mesh1.xml.gz");
  double Lx=20.0,Ly=2.0,Lz=2.0;

  // set parameters for FASP solver
  char fasp_params[] = "./benchmarks/physic_bench/bsr.dat";
  printf("\tFASP parameters... %s\n", fasp_params);
  itsolver_param itsolver;
  input_param input;
  AMG_param amg;
  ILU_param ilu;
  fasp_param_input(fasp_params, &input);
  fasp_param_init(&input, &itsolver, &amg, &ilu, NULL);

  //-------------------------``
  // Mesh Adaptivity Loop
  //-------------------------
  dolfin::File accepted_solution_file("./benchmarks/physic_bench/output/accepted_solution.pvd");
  dolfin::File accepted_solution_xml("./benchmarks/physic_bench/output/accepted_solution.xml");
  dolfin::File mesh_file("./benchmarks/physic_bench/output/accepted_mesh.pvd");
  dolfin::File mesh_xml("./benchmarks/physic_bench/output/accepted_mesh.xml.gz");
  std::string output_path("./benchmarks/physic_bench/output/");

  dolfin::File computed_solution_file("./benchmarks/physic_bench/output/computed_solution.pvd");
  dolfin::File adapted_solution_file("./benchmarks/physic_bench/output/adapted_solution.pvd");


  // mesh adaptivity
  const double growth_factor = 3.0;
  const double entropy_error_per_cell = 1.0e-3;
  const std::size_t max_refine_depth = 3;
  const std::size_t max_elements = 200000;

  // parameters for PNP Newton solver
  const std::size_t max_newton = 200;
  const double max_residual_tol = 1.0e-10;
  const double relative_residual_tol = 1.0e-10;
  const bool use_eafe_approximation = false;

    Mesh_Refiner mesh_adapt(
      initial_mesh,
      max_elements,
      max_refine_depth,
      entropy_error_per_cell
    );

    std::shared_ptr<double> initial_residual_ptr = std::make_shared<double>(-1.0);

    // construct initial guess
    Linear_Function PNP(0,-Lx/2.0,Lx/2.0,{-0.01,0.0,0.0},{0.01,0.0,0.0});
    // Linear_Function PNP(0,-Lx/2.0,Lx/2.0,{-1.0,0.0,-2.30258509299},{1.0,-2.30258509299,0.0});
    auto adaptive_solution = std::make_shared<dolfin::Function>(
      std::make_shared<vector_linear_pnp_forms::FunctionSpace>(mesh_adapt.get_mesh())
    );
    adaptive_solution->interpolate(PNP);

    dolfin::File initial_guess_file("./benchmarks/physic_bench/output/initial_guess.pvd");
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
        ilu,
        output_path
      );

     //auto computed_solution = std::make_shared<dolfin::Function>(computed_solution2);

     if (mesh->num_cells()<max_elements){
      // compute current / entropy terms
      printf("Computing current\n");
      auto diffusivity = get_diode_diffusivity(computed_solution->function_space());
      auto entropy_potential = compute_entropy_potential(computed_solution,{0.0,-1.0,1.0});
      auto log_densities = extract_log_densities(computed_solution);


      // adapt computed solutions
      mesh_adapt.max_elements = (std::size_t) std::floor(growth_factor * mesh->num_cells());
      mesh_adapt.multilevel_refinement(diffusivity, entropy_potential, log_densities);
      adaptive_solution = adapt( *computed_solution, mesh_adapt.get_mesh() );
      adaptive_solution->interpolate(PNP);

     adapted_solution_file << *adaptive_solution;
     computed_solution_file << *computed_solution;
     mesh_file << *mesh;
   }
   else{
     adaptive_solution = computed_solution;
     mesh_adapt.needs_to_solve = false;
   }


    }


    accepted_solution_file << *adaptive_solution;
    accepted_solution_xml << *adaptive_solution;
    mesh_file << *mesh_adapt.get_mesh();
    mesh_xml << *mesh_adapt.get_mesh();

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
  dolfin::Constant ones(0.0, 1.0, 1.0);
  diffusivity.interpolate(ones);

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
  std::shared_ptr<dolfin::Function> solution, std::vector<double> valencies
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
    *(potential.vector()) += *potential.vector();

    auto const_entropy_potential = std::make_shared<const dolfin::Function>(entropy_potential);
    function_vec.push_back(const_entropy_potential);
  }

  return function_vec;
}
