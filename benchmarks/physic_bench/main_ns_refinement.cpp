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
  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
}

#include "vector_linear_pnp_ns_forms.h"
#include "pnp_ns_newton_solver.h"
#include "mesh_refiner.h"

using namespace std;

// helper functions for marking cells for refinement
std::vector<std::shared_ptr<const dolfin::Function>> get_diffusivity(
  std::shared_ptr<const dolfin::FunctionSpace> function_space
);
std::vector<std::shared_ptr<const dolfin::Function>> extract_log_densities (
  std::shared_ptr<dolfin::Function> solution
);
std::vector<std::shared_ptr<const dolfin::Function>> compute_entropy_potential(
  std::shared_ptr<dolfin::Function> solution, std::vector<double> valencies
);


int main (int argc, char** argv) {
  printf("\n");
  printf("----------------------------------------------------\n");
  printf(" Setting up the linearized PNP+Stokes problem\n");
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
  *initial_mesh = dolfin::Mesh("./benchmarks/physic_bench/mesh2.xml.gz");
  double Lx=2.0,Ly=2.0,Lz=2.0;


  // Setup FASP solver
  printf("FASP solver parameters..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/physic_bench/bcsr.dat";
  fasp_param_input(fasp_params, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  printf("done\n"); fflush(stdout);

  // Setup FASP solver for pnp
  printf("FASP solver parameters for pnp..."); fflush(stdout);
  input_param pnp_inpar;
  itsolver_param pnp_itpar;
  AMG_param  pnp_amgpar;
  ILU_param pnp_ilupar;
  Schwarz_param pnp_schpar;
  char fasp_pnp_params[] = "./benchmarks/physic_bench/bsr.dat";
  fasp_param_input(fasp_pnp_params, &pnp_inpar);
  fasp_param_init(&pnp_inpar, &pnp_itpar, &pnp_amgpar, &pnp_ilupar, &pnp_schpar);
  printf("done\n"); fflush(stdout);

  // Setup FASP solver for stokes
  printf("FASP solver parameters for stokes..."); fflush(stdout);
  input_ns_param ns_inpar;
  itsolver_ns_param ns_itpar;
  AMG_ns_param  ns_amgpar;
  ILU_param ns_ilupar;
  Schwarz_param ns_schpar;
  char fasp_ns_params[] = "./benchmarks/physic_bench/ns.dat";
  fasp_ns_param_input(fasp_ns_params, &ns_inpar);
  fasp_ns_param_init(&ns_inpar, &ns_itpar, &ns_amgpar, &ns_ilupar, &ns_schpar);
  printf("done\n"); fflush(stdout);



  //-------------------------
  // Construct Problem
  //-------------------------

  // std::vector<Linear_Function> InitialGuess;
  // Linear_Function PNP(0,-5.0,5.0,{0.0,-2.30258509299,1.0},{-2.30258509299,0.0,-1.0});
  // Linear_Function Vel(0,-5.0,5.0,{1.0,0.0,0.0},{1.0,0.0,0.0});
  // Linear_Function Pres(0,-5.0,5.0,0.0,0.0);
  // InitialGuess.push_back(PNP);
  // InitialGuess.push_back(Vel);
  // InitialGuess.push_back(Pres);

  auto mesh_PNP = std::make_shared<dolfin::Mesh>("./benchmarks/physic_bench/output_PNP/accepted_mesh.xml.gz");
  auto CGpnp = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_cc>(mesh_PNP);
  // auto RT = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_uu>(mesh_PNP);
  // auto DG = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_pp>(mesh_PNP);

  auto CG = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_cc>(initial_mesh);
  auto RT = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_uu>(initial_mesh);
  auto DG = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_pp>(initial_mesh);

  dolfin::Function pnp_solution(CGpnp,"./benchmarks/physic_bench/output_PNP/accepted_solution.xml");
  dolfin::Function pnp_init(CG);
  dolfin::Function u_init(RT);
  dolfin::Function p_init(DG);
  auto initvel = std::make_shared<dolfin::Constant>(0.0,0.0,0.0);
  auto initp = std::make_shared<dolfin::Constant>(0.0);
  pnp_init.interpolate(pnp_solution);
  u_init.interpolate(*initvel);
  p_init.interpolate(*initp);

  std::vector<dolfin::Function> InitialGuess;
  InitialGuess.push_back(pnp_init);
  InitialGuess.push_back(u_init);
  InitialGuess.push_back(p_init);

  printf("\n");


  //-------------------------
  // Mesh Adaptivity Loop
  //-------------------------

  // parameters for mesh adaptivity
  double growth_factor = 3.0;
  double entropy_per_cell = 1.0e-4;
  std::size_t max_refine_depth = 3;
  std::size_t max_elements = 25000;
  Mesh_Refiner mesh_adapt(
    initial_mesh,
    max_elements,
    max_refine_depth,
    entropy_per_cell
  );

  // parameters for PNP Newton solver
  const std::size_t max_newton = 25;
  const double max_residual_tol = 1.0e-7;
  const double relative_residual_tol = 1.0e-7;
  const bool use_eafe_approximation = false;
  std::shared_ptr<double> initial_residual_ptr = std::make_shared<double>(-1.0);

  // construct initial guess
  std::vector<std::shared_ptr<dolfin::Function>> adaptive_solution;
  adaptive_solution.reserve(3);
  adaptive_solution.resize(3);
  adaptive_solution[0].reset(new dolfin::Function(std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_cc>(mesh_adapt.get_mesh())));
  adaptive_solution[0]->interpolate(InitialGuess[0]);
  adaptive_solution[1].reset(new dolfin::Function(std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_uu>(mesh_adapt.get_mesh())));
  adaptive_solution[1]->interpolate(InitialGuess[1]);
  adaptive_solution[2].reset(new dolfin::Function(std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_pp>(mesh_adapt.get_mesh())));
  adaptive_solution[2]->interpolate(InitialGuess[2]);

  dolfin::File initial_guess_file0("./benchmarks/physic_bench/output/initial_cation.pvd");
  dolfin::File initial_guess_file1("./benchmarks/physic_bench/output/initial_anion.pvd");
  dolfin::File initial_guess_file2("./benchmarks/physic_bench/output/initial_potential.pvd");
  dolfin::File initial_guess_file3("./benchmarks/physic_bench/output/initial_velocity.pvd");
  dolfin::File initial_guess_file4("./benchmarks/physic_bench/output/initial_pressure.pvd");

  dolfin::File solution_file0("./benchmarks/physic_bench/output/cation_solution.pvd");
  dolfin::File solution_file1("./benchmarks/physic_bench/output/anion_solution.pvd");
  dolfin::File solution_file2("./benchmarks/physic_bench/output/potential_solution.pvd");
  dolfin::File solution_file3("./benchmarks/physic_bench/output/velocity_solution.pvd");
  dolfin::File solution_file4("./benchmarks/physic_bench/output/pressure_solution.pvd");

  solution_file0 << (*adaptive_solution[0])[0];
  solution_file1 << (*adaptive_solution[0])[1];
  solution_file2 << (*adaptive_solution[0])[2];
  solution_file3 << (*adaptive_solution[1]);
  solution_file4 << (*adaptive_solution[2]);

  while (mesh_adapt.needs_to_solve) {
    auto mesh = mesh_adapt.get_mesh();

    initial_guess_file0 << (*adaptive_solution[0])[0];
    initial_guess_file1 << (*adaptive_solution[0])[1];
    initial_guess_file2 << (*adaptive_solution[0])[2];
    initial_guess_file3 << (*adaptive_solution[1]);
    initial_guess_file4 << (*adaptive_solution[2]);

    auto computed_solution = solve_pnp_stokes (
      mesh_adapt.iteration++,
      mesh,
      Lx,Ly,Lz,
      adaptive_solution,
      max_newton,
      max_residual_tol,
      relative_residual_tol,
      initial_residual_ptr,
      itpar,
      pnp_itpar,
      pnp_amgpar,
      ns_itpar,
      ns_amgpar,
      "./benchmarks/physic_bench/output/"
    );

    adaptive_solution[0]->interpolate(computed_solution[0]);
    adaptive_solution[1]->interpolate(computed_solution[1]);
    adaptive_solution[2]->interpolate(computed_solution[2]);

    if (mesh->num_cells()<max_elements){
      // compute entropy terms to mark cells for refinement
      auto diffusivity = get_diffusivity(adaptive_solution[0]->function_space());
      auto entropy_potential = compute_entropy_potential(adaptive_solution[0],{-1.0,1.0});
      auto log_densities = extract_log_densities(adaptive_solution[0]);

      mesh_adapt.max_elements = (std::size_t) std::floor( growth_factor * mesh->num_cells() );
      mesh_adapt.multilevel_refinement(diffusivity, entropy_potential, log_densities);

      // update solution
      adaptive_solution[0].reset( new dolfin::Function(computed_solution[0].function_space()) );
      adaptive_solution[0]->interpolate(computed_solution[0]);
      adaptive_solution[1].reset( new dolfin::Function(computed_solution[1].function_space()) );
      adaptive_solution[1]->interpolate(computed_solution[1]);
      adaptive_solution[2].reset( new dolfin::Function(computed_solution[2].function_space()) );
      adaptive_solution[2]->interpolate(computed_solution[2]);

      // adaptive_solution[0]->interpolate(InitialGuess[0]);
      // adaptive_solution[1]->interpolate(InitialGuess[1]);
      // adaptive_solution[2]->interpolate(InitialGuess[2]);
    }
    else{
      mesh_adapt.needs_to_solve = false;
    }
    solution_file0 << (*adaptive_solution[0])[0];
    solution_file1 << (*adaptive_solution[0])[1];
    solution_file2 << (*adaptive_solution[0])[2];
    solution_file3 << (*adaptive_solution[1]);
    solution_file4 << (*adaptive_solution[2]);

  }
  printf("Solver exiting\n"); fflush(stdout);

  // dolfin::File xml_mesh("./benchmarks/physic_bench/output/mesh.xml");
  dolfin::File pvd_file0("./benchmarks/physic_bench/output/pnp_solution.pvd");
  dolfin::File pvd_file1("./benchmarks/physic_bench/output/velocity_solution.pvd");
  // xml_mesh << *mesh;
  pvd_file0 << *adaptive_solution[0];
  pvd_file1 << *adaptive_solution[1];

  return 0;
}


/**
 * Helper functions for marking elements in need of refinement
 */
 std::vector<std::shared_ptr<const dolfin::Function>> get_diffusivity(
   std::shared_ptr<const dolfin::FunctionSpace> function_space
 ) {
   dolfin::Constant ones(1.0, 1.334/2.032, 0.0);
   dolfin::Function diffusivity(function_space);
   diffusivity.interpolate(ones);

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
  for (std::size_t comp = 0; comp < component_count-1; comp++) {
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

  for (std::size_t comp = 0; comp < component_count-1; comp++) {
    auto subfunction_space = (*solution)[comp].function_space()->collapse();
    dolfin::Function potential(subfunction_space);
    dolfin::Function entropy_potential(subfunction_space);

    potential.interpolate((*solution)[2]);
    entropy_potential.interpolate((*solution)[comp]);
    *(potential.vector()) *= valencies[comp];
     *entropy_potential.vector() += *potential.vector();

    auto const_entropy_potential = std::make_shared<const dolfin::Function>(entropy_potential);
    function_vec.push_back(const_entropy_potential);
  }

  return function_vec;
}
