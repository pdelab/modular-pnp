/// Main file for solving the linearized PNP problem
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <dolfin.h>
#include "pde.h"
#include "newton_status.h"
#include "mesh_refiner.h"
#include "domain.h"
#include "error.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "linear_pnp.h"

using namespace std;

///
/// dimensional analysis
///
const double elementary_charge = 1.60217662e-19; // C
const double boltzmann = 1.38064852e-23; // J / K
const double temperature = 3e+2; // K
const double thermodynamic_beta = elementary_charge / (temperature * boltzmann); // V
const double vacuum_permittivity = 8.854187817e-12; // C / V

const double reference_length = 1e-6; // m
const double voltage_ground = 0.0;
const double reference_density = 1.5e+22; // mM = 1 / m^3
const double reference_diffusivity = 2.87e+1; // cm^2 / s
const double reference_relative_permittivity = 1.17e+1; // dimensionless

const double permittivity_factor = reference_relative_permittivity * vacuum_permittivity /
  (elementary_charge * thermodynamic_beta * reference_density * reference_length * reference_length);

double scale_density (double density) { return density / reference_density; };
double scale_potential (double phi) { return (phi - voltage_ground) * thermodynamic_beta; };
double scale_diffusivity (double diff) { return diff / reference_diffusivity; };
double scale_rel_permittivity (double rel_perm) { return rel_perm / reference_relative_permittivity; };


///
/// define coefficients
///
const double majority_carrier = 1.5e+22; // mM = 1 / m^3
const double minority_carrier = 6.6e+9; // mM = 1 / m^3
std::vector<double> valencies = { 0.0, 1.0, -1.0 }; // potential "valency" is at valencies[0] and should be zero
std::vector<double> reactions (double x) { return { 0.0, 0.0, 0.0 }; };
std::vector<double> diffusivities (double x) {
  return {
    0.0,
    x < 0.0 ? scale_diffusivity(1.07e+1) : scale_diffusivity(1.09e+1), // cm^2 / s
    x < 0.0 ? scale_diffusivity(2.69e+1) : scale_diffusivity(2.87e+1) // cm^2 / s
  };
};
double relative_permittivity (double x) { return scale_rel_permittivity(1.17e+1); }; // dimensionless
double fixed (double x) {
  return x < 0.0 ? scale_density(majority_carrier) : -scale_density(majority_carrier); // mM = 1 / m^3
};

// boundary conditions
std::vector<double> left_contact (double x) {
  return {
    scale_potential(+1.0), // V
    std::log(scale_density(minority_carrier)), // log(mM)
    std::log(scale_density(majority_carrier)) // log(mM)
  };
};
std::vector<double> right_contact (double x) {
  return {
    scale_potential(-1.0), // V
    std::log(scale_density(majority_carrier)), // log(mM)
    std::log(scale_density(minority_carrier)) // log(mM)
  };
};


///
/// define expressions from coefficients
///
class Initial_Guess : public dolfin::Expression {
  public:
    Initial_Guess() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      std::vector<double> left(left_contact(-1.0));
      std::vector<double> right(right_contact(+1.0));
      values[0] = 0.5 * (left[0] * (1.0 - x[0]) + right[0] * (x[0] + 1.0));
      values[1] = 0.5 * (left[1] * (1.0 - x[0]) + right[1] * (x[0] + 1.0));
      values[2] = 0.5 * (left[2] * (1.0 - x[0]) + right[2] * (x[0] + 1.0));
    }
};

class Permittivity_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = relative_permittivity(x[0]);
    }
};

class Poisson_Scale_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = 1.0 / permittivity_factor;
    }
};

class Fixed_Charged_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = fixed(x[0]);
    }
};

class Diffusivity_Expression : public dolfin::Expression {
  public:
    Diffusivity_Expression() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      std::vector<double> diff(diffusivities(x[0]));
      values[0] = diff[0];
      values[1] = diff[1];
      values[2] = diff[2];
    }
};

class Reaction_Expression : public dolfin::Expression {
  public:
    Reaction_Expression() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      std::vector<double> reac(reactions(x[0]));
      values[0] = reac[0];
      values[1] = reac[1];
      values[2] = reac[2];
    }
};

class Valency_Expression : public dolfin::Expression {
  public:
    Valency_Expression() : dolfin::Expression(3) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = valencies[0]; // potential valency is not used and should be zero
      values[1] = valencies[1];
      values[2] = valencies[2];
    }
};


// auxiliary functions for main
std::shared_ptr<dolfin::Function> solve_pnp (
  std::size_t iteration,
  std::shared_ptr<const dolfin::Mesh> mesh,
  std::shared_ptr<dolfin::Function> initial_guess,
  bool use_eafe_approximation,
  itsolver_param itsolver,
  AMG_param amg
);

std::vector<std::shared_ptr<const dolfin::Function>> extract_log_densities (
  std::shared_ptr<dolfin::Function> solution
);

std::vector<std::shared_ptr<const dolfin::Function>> compute_entropy_potential(
  std::shared_ptr<dolfin::Function> solution
);

void print_error (
  dolfin::Function computed_solution
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
  bool use_eafe_approximation = true;

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

  // construct initial guess
  Initial_Guess initial_guess_expression;
  auto adaptive_solution = std::make_shared<dolfin::Function>(
    std::make_shared<vector_linear_pnp_forms::FunctionSpace>(mesh_adapt.get_mesh())
  );
  adaptive_solution->interpolate(initial_guess_expression);

  dolfin::File initial_guess_file("./benchmarks/pnp_diode/output/initial_guess.pvd");
  initial_guess_file << *adaptive_solution;

  while (mesh_adapt.needs_to_solve) {
    // compute solution on current mesh
    auto mesh = mesh_adapt.get_mesh();
    auto computed_solution = solve_pnp(
      mesh_adapt.iteration++,
      mesh,
      adaptive_solution,
      use_eafe_approximation,
      itsolver,
      amg
    );

    // print error of computed solution
    // print_error(*computed_solution);

    // compute entropy terms
    auto entropy_potential = compute_entropy_potential(computed_solution);
    auto log_densities = extract_log_densities(computed_solution);

    // adapt mesh
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

/// Extract log-densities from computed solution
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

/// Compute potential functions for components of the entropy
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

/// compute solution to PNP equation
/// using a Newton solver on the given mesh
std::shared_ptr<dolfin::Function> solve_pnp (
  std::size_t adaptivity_iteration,
  std::shared_ptr<const dolfin::Mesh> mesh,
  std::shared_ptr<dolfin::Function> initial_guess,
  bool use_eafe_approximation,
  itsolver_param itsolver,
  AMG_param amg
) {
  // setup function spaces and forms
  printf("\nConstruct vector PNP problem\n");
  std::shared_ptr<dolfin::FunctionSpace> function_space;
  std::shared_ptr<dolfin::Form> bilinear_form;
  std::shared_ptr<dolfin::Form> linear_form;
  function_space.reset(
    new vector_linear_pnp_forms::FunctionSpace(mesh)
  );
  bilinear_form.reset(
    new vector_linear_pnp_forms::Form_a(function_space, function_space)
  );
  linear_form.reset(
    new vector_linear_pnp_forms::Form_L(function_space)
  );

  // set initializer for PDE coefficients
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> pnp_coefficients = {
    {"permittivity", {1.0}},
    {"poisson_scale", {permittivity_factor}},
    {"diffusivity", {0.0, 2.0, 2.0}},
    {"valency", {0.0, 1.0, -1.0}}
  };
  std::map<std::string, std::vector<double>> pnp_sources = {
    {"fixed_charge", {1.0}},
    {"poisson_scale", {permittivity_factor}},
    {"reaction", {0.0, 0.0, 0.0}}
  };

  // build problem
  Linear_PNP pnp_problem(
    mesh,
    function_space,
    bilinear_form,
    linear_form,
    pnp_coefficients,
    pnp_sources,
    itsolver,
    amg
  );

  // set eafe flag
  if (use_eafe_approximation) {
  printf("Setting solver to use EAFE approximation\n");
    pnp_problem.use_eafe();
  }

  printf("Define PNP coefficients from expressions\n");
  dolfin::Function permittivity(pnp_problem.permittivity_space);
  Permittivity_Expression permittivity_expr;
  permittivity.interpolate(permittivity_expr);

  dolfin::Function poisson_scale(pnp_problem.permittivity_space);
  Poisson_Scale_Expression poisson_scale_expr;
  poisson_scale.interpolate(poisson_scale_expr);

  dolfin::Function charges(pnp_problem.fixed_charge_space);
  Fixed_Charged_Expression fc_expr;
  charges.interpolate(fc_expr);

  dolfin::Function diffusivity(pnp_problem.diffusivity_space);
  Diffusivity_Expression diff_expr;
  diffusivity.interpolate(diff_expr);

  dolfin::Function reaction(pnp_problem.reaction_space);
  Reaction_Expression reac_expr;
  reaction.interpolate(reac_expr);

  dolfin::Function valency(pnp_problem.valency_space);
  Valency_Expression valency_expr;
  valency.interpolate(valency_expr);

  std::map<std::string, dolfin::Function> pnp_coefficient_fns = {
    {"permittivity", permittivity},
    {"poisson_scale", poisson_scale},
    {"diffusivity", diffusivity},
    {"valency", valency}
  };
  std::map<std::string, dolfin::Function> pnp_source_fns = {
    {"fixed_charge", charges},
    {"reaction", reaction}
  };

  pnp_problem.set_coefficients(
    pnp_coefficient_fns,
    pnp_source_fns
  );


  //-------------------------
  // Print various solutions
  //-------------------------
  std::string path("./benchmarks/pnp_diode/output/adapt_");
  path += std::to_string(adaptivity_iteration);
  dolfin::File solution_file0(path + "_1solution.pvd");
  dolfin::File solution_file1(path + "_2solution.pvd");
  dolfin::File solution_file2(path + "_3solution.pvd");
  dolfin::File total_charge_file(path + "_total_charge.pvd");

  // initial guess for prescibed Dirichlet
  printf("Record interpolant for given Dirichlet BCs (initial guess for solution)\n");
  std::vector<std::size_t> components = {0, 0, 0};
  std::vector<std::vector<double>> bcs;

  std::vector<double> left(left_contact(-1.0));
  std::vector<double> right(right_contact(+1.0));
  bcs.push_back({left[0], right[0]});
  bcs.push_back({left[1], right[1]});
  bcs.push_back({left[2], right[2]});

  pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function initial_guess_function = pnp_problem.get_solution();
  initial_guess_function.interpolate(*initial_guess);
  pnp_problem.set_solution(initial_guess_function);

  // output to file
  initial_guess_function = pnp_problem.get_solution();
  solution_file0 << initial_guess_function[0];
  solution_file1 << initial_guess_function[1];
  solution_file2 << initial_guess_function[2];
  total_charge_file << pnp_problem.get_total_charge();
  printf("\n");



  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  const std::size_t max_newton = 50;
  const double max_residual_tol = 1.0e-10;
  const double relative_residual_tol = 1.0e-8;
  const double initial_residual = pnp_problem.compute_residual("l2");
  const double initial_max_residual = pnp_problem.compute_residual("max");
  Newton_Status newton(
    max_newton,
    initial_residual,
    relative_residual_tol,
    max_residual_tol
  );

  printf("\tinitial residual :     %10.5e\n", newton.initial_residual);
  printf("\tinitial max residual : %10.5e\n", initial_max_residual);
  printf("\n");

  newton.update_max_residual(initial_max_residual);
  while (newton.needs_to_iterate()) {
    // solve
    printf("Solving for Newton iterate %lu \n", newton.iteration);
    const double previous_residual = pnp_problem.compute_residual("l2");
    const dolfin::Function previous_solution = pnp_problem.get_solution();
    dolfin::Function computed_solution(pnp_problem.fasp_solve());

    // update newton measurements with backtracking
    printf("Newton measurements for iteration :\n");
    double residual_check = pnp_problem.compute_residual("l2");
    std::size_t backtrack_count = 0;

    // ensure L_infinity norm has bounded growth at each iteration
    double prev_max_dof = std::abs(previous_solution.vector()->max());
    double prev_min_dof = std::abs(previous_solution.vector()->min());
    double prev_L_infty = prev_max_dof > prev_min_dof ? prev_max_dof : prev_min_dof;

    double max_dof = std::abs(computed_solution.vector()->max());
    double min_dof = std::abs(computed_solution.vector()->min());
    double L_infty = max_dof > min_dof ? max_dof : min_dof;

    if (L_infty > 1.1 * prev_L_infty) {
      printf("\tupdate causes too much growth in solution : %e\n", L_infty / prev_L_infty);
      dolfin::Function newton_update(computed_solution.function_space());
      newton_update = computed_solution - previous_solution;

      double update_max_dof = std::abs(newton_update.vector()->max());
      double update_min_dof = std::abs(newton_update.vector()->min());
      double update_L_infty = update_max_dof > update_min_dof ? update_max_dof : update_min_dof;

      double factor = prev_L_infty / (update_L_infty + 1E-12);
      newton_update = newton_update * factor;
      computed_solution = previous_solution + newton_update;
    }

    // dolfin::File backFile("./benchmarks/pnp_diode/output/backtrack.pvd");
    while (residual_check > previous_residual || isnan(residual_check)) {
      printf("\trelative residual increased : %e < %e\n", previous_residual, residual_check);
      dolfin::Function backtrack(computed_solution.function_space());
      backtrack = previous_solution - computed_solution;
      *(backtrack.vector()) *= 1.0 - std::pow(0.5, ++backtrack_count);

      dolfin::Function backtrack_solution(computed_solution.function_space());
      backtrack_solution = computed_solution + backtrack;
      // backFile << backtrack_solution;
      pnp_problem.set_solution(backtrack_solution);
      residual_check = pnp_problem.compute_residual("l2");
    }
    printf("\trelative residual decreased : %e > %e\n", previous_residual, residual_check);
    double residual = pnp_problem.compute_residual("l2");
    double max_residual = pnp_problem.compute_residual("max");
    newton.update_residuals(residual, max_residual);
    newton.update_iteration();

    // output
    printf("\tmaximum residual :  %10.5e\n", newton.max_residual);
    printf("\trelative residual : %10.5e\n", newton.relative_residual);
    printf("\toutput solution to file...\n");
    solution_file0 << computed_solution[0];
    solution_file1 << computed_solution[1];
    solution_file2 << computed_solution[2];
    total_charge_file << pnp_problem.get_total_charge();
    printf("\n");
  }


  // check status of nonlinear solve
  if (newton.converged()) {
    printf("Solver succeeded!\n");
  } else {
    newton.print_status();
  }
  printf("\nSolver exiting\n"); fflush(stdout);

  // plot coefficients if requested
  bool plot_coefficients = true;
  if (plot_coefficients) {
    printf("\toutput coefficients to file\n");
    dolfin::File permittivity_file("./benchmarks/pnp_diode/output/permittivity.pvd");
    dolfin::File charges_file("./benchmarks/pnp_diode/output/charges.pvd");
    dolfin::File diffusivity_file("./benchmarks/pnp_diode/output/diffusivity.pvd");
    dolfin::File reaction_file("./benchmarks/pnp_diode/output/reaction.pvd");
    dolfin::File valency_file("./benchmarks/pnp_diode/output/valency.pvd");
    permittivity_file << permittivity;
    charges_file << charges;
    diffusivity_file << diffusivity[1];
    diffusivity_file << diffusivity[2];
    reaction_file << reaction[1];
    reaction_file << reaction[2];
    valency_file << valency[1];
    valency_file << valency[2];
  }

  return std::make_shared<dolfin::Function>(pnp_problem.get_solution());
}
