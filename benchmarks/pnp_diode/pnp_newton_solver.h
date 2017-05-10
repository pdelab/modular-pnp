#ifndef __PNP_NEWTON_SOLVER_H
#define __PNP_NEWTON_SOLVER_H

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
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "linear_pnp.h"
#include "diode.h"
#include "pnp_newton_solver.h"

/// compute solution to PNP equation
/// using a Newton solver on the given mesh
std::shared_ptr<dolfin::Function> solve_pnp (
  std::size_t adaptivity_iteration,
  std::shared_ptr<const dolfin::Mesh> mesh,
  std::shared_ptr<dolfin::Function> initial_guess,
  const std::size_t max_newton,
  const double max_residual_tol,
  const double relative_residual_tol,
  std::shared_ptr<double> initial_residual_ptr,
  bool use_eafe_approximation,
  itsolver_param itsolver,
  AMG_param amg,
  std::string output_dir
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
    amg,
    "uu"
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
  std::string path(output_dir + "adapt_");
  path += std::to_string(adaptivity_iteration);
  dolfin::File total_charge_file(path + "_total_charge.pvd");
  dolfin::File total_solution_file(path + "_total_solution.pvd");

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
  total_solution_file << initial_guess_function;
  total_charge_file << pnp_problem.get_total_charge();
  printf("\n");



  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  if (*initial_residual_ptr < 0.0) {
    *initial_residual_ptr = pnp_problem.compute_residual("l2");
  }
  const double initial_max_residual = pnp_problem.compute_residual("max");
  Newton_Status newton(
    max_newton,
    *initial_residual_ptr,
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
    double prev_max_dof = previous_solution.vector()->max();
    double prev_min_dof = previous_solution.vector()->min();
    double prev_range = prev_max_dof - prev_min_dof + 1E-12;

    double max_dof = computed_solution.vector()->max();
    double min_dof = computed_solution.vector()->min();
    double range = max_dof - min_dof;

    // avoid updates that cause more than than 1% growth in the solution
    double increase_tolerance = 1E-1;
    if (range > (1.0 + increase_tolerance) * prev_range) {
      printf("\tupdate causes too much growth in solution : %e / %e = %e\n", range, prev_range, range / prev_range);
      dolfin::Function newton_update(computed_solution.function_space());
      newton_update = computed_solution - previous_solution;

      double max_update = newton_update.vector()->max();
      double min_update = newton_update.vector()->min();
      double growth_factor = increase_tolerance * prev_range / (max_update - min_update + 1E-12);
      newton_update = newton_update * growth_factor;

      computed_solution = previous_solution + newton_update;
    }

    // dolfin::File backFile(output_dir + "backtrack.pvd");
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
    total_solution_file << computed_solution;
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
  bool plot_coefficients = false;
  if (plot_coefficients) {
    printf("\toutput coefficients to file\n");
    dolfin::File permittivity_file(output_dir + "permittivity.pvd");
    dolfin::File charges_file(output_dir + "charges.pvd");
    dolfin::File diffusivity_file(output_dir + "diffusivity.pvd");
    dolfin::File reaction_file(output_dir + "reaction.pvd");
    dolfin::File valency_file(output_dir + "valency.pvd");
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

#endif
