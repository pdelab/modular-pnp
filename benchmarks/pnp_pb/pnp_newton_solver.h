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
#include "error.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "linear_pnp.h"
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
  ILU_param ilu,
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
  double Eps = 0.01;
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> pnp_coefficients = {
    {"permittivity", {Eps}},
    {"diffusivity", {0.0, 1.0, 1.0}},
    {"valency", {0.0, 1.0, -1.0}}
  };
  std::map<std::string, std::vector<double>> pnp_sources = {
    {"fixed_charge", {0.0}},
    {"phib", {100.0,100.0,100.0}}
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
    ilu,
    "uu"
  );

  dolfin::Function phib(pnp_problem.phib_space);
  PhibExpression bdexpr(Eps);
  phib.interpolate(bdexpr);

  std::map<std::string, dolfin::Function> pnp_source_fns  = { {"phib", phib}  };
  std::map<std::string, dolfin::Function> emptymap = {};
  pnp_problem.set_coefficients(    emptymap,    pnp_source_fns  );

  //-------------------------
  // Print various solutions
  //-------------------------
  std::string path(output_dir + "adapt_");
  path += std::to_string(adaptivity_iteration);
  path += "_";

  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0(path +"sphi.pvd");
  dolfin::File solution_file1(path  +"eta1.pvd");
  dolfin::File solution_file2(path  +"eta2.pvd");

  dolfin::File xml_filePhipb(path  +"phipb.xml");
  dolfin::File xml_file0(path  +"phi.xml");
  dolfin::File xml_file1(path  +"eta1.xml");
  dolfin::File xml_file2(path  +"eta2.xml");


  pnp_problem.init_BC();
  pnp_problem.set_solution(*initial_guess);

  dolfin::Function solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  printf("\n");


  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  double mesh_initial_residual = pnp_problem.compute_residual("l2");
  const double dof_size = pnp_problem._eigen_vector->size();
  mesh_initial_residual /= dof_size;
  if (*initial_residual_ptr < 0.0) {
    *initial_residual_ptr = mesh_initial_residual;
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

  // avoid updates that cause more than than 5% growth in the solution
  bool fasp_reset = false;
  uint fasp_fail_count = 0;
  double increase_tolerance = 1E-1;

  while (newton.needs_to_iterate()) {
      // solve
      printf("\t\tSolving for Newton iterate %lu \n", newton.iteration);
      solutionFn = pnp_problem.fasp_solve();

      // update newton measurements
      printf("\t\tNewton measurements for iteration :\n");
      double residual = pnp_problem.compute_residual("l2");
      double max_residual = pnp_problem.compute_residual("max");
      newton.update_residuals(residual, max_residual);
      newton.update_iteration();

      // output
      printf("\t\tmaximum residual :  %10.5e\n", newton.max_residual);
      printf("\t\trelative residual : %10.5e\n", newton.relative_residual);
      printf("\t\toutput solution to file...\n");
      solution_file0 << solutionFn[0];
      solution_file1 << solutionFn[1];
      solution_file2 << solutionFn[2];
      dolfin::Function phi(solutionFn[0]);
      xml_file0 << phi;
      printf("\n");
  }

  // check status of nonlinear solve
  if (newton.converged()) {
    printf("\tSolver succeeded!\n");
  } else {
    newton.print_status();
  }

    ExactExpression ExExp(Eps);
    auto ExSol = std::make_shared<dolfin::Function>(solutionFn);
    auto Sol = std::make_shared<dolfin::Function>(solutionFn);
    ExSol->interpolate(ExExp);
    Error Err(ExSol);
    double L2Err = Err.compute_h1_error(Sol);
    printf("The L2 Error is %f for mesh size %f\n",L2Err,mesh->hmax()); fflush(stdout);

    printf("\tSolver exiting\n"); fflush(stdout);
    // return pnp_problem.get_solutions();
    return std::make_shared<dolfin::Function>(pnp_problem.get_solution());
}

#endif
