#ifndef __PNP_NS_NEWTON_SOLVER_STOKES_H
#define __PNP_NS_NEWTON_SOLVER_STOKES_H

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
#include "domain.h"
#include "dirichlet.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
}

#include "vector_linear_pnp_ns_forms.h"
#include "linear_pnp_ns.h"

#include "spheres.h"


using namespace std;



std::vector<dolfin::Function> solve_pnp_stokes (
  std::size_t adaptivity_iteration,
  std::shared_ptr<const dolfin::Mesh> mesh,
  double Lx,
  double Ly,
  double Lz,
  std::vector<std::shared_ptr<dolfin::Function>> initial_guess,
  const std::size_t max_newton,
  const double max_residual_tol,
  const double relative_residual_tol,
  std::shared_ptr<double> initial_residual_ptr,
  const itsolver_param &itsolver,
  const itsolver_param &pnpitsolver,
  const AMG_param &pnpamg,
  const itsolver_ns_param &nsitsolver,
  const AMG_ns_param &nsamg,
  std::string output_dir
){
  //-------------------------
  // Construct Problem
  //-------------------------
  printf("\tConstruct vector PNP problem\n");

  // setup function spaces and forms
  std::shared_ptr<dolfin::FunctionSpace> function_space;
  std::shared_ptr<dolfin::Form> bilinear_form;
  std::shared_ptr<dolfin::Form> linear_form;
  function_space.reset(
    new vector_linear_pnp_ns_forms::FunctionSpace(mesh)
  );
  bilinear_form.reset(
    new vector_linear_pnp_ns_forms::Form_a(function_space, function_space)
  );
  linear_form.reset(
    new vector_linear_pnp_ns_forms::Form_L(function_space)
  );
  std::vector<std::shared_ptr<dolfin::FunctionSpace>> functions_space;
  std::shared_ptr<dolfin::FunctionSpace> function_space1;
  std::shared_ptr<dolfin::FunctionSpace> function_space2;
  std::shared_ptr<dolfin::FunctionSpace> function_space3;
  function_space1.reset(new vector_linear_pnp_ns_forms::CoefficientSpace_cc(mesh));
  function_space2.reset(new vector_linear_pnp_ns_forms::CoefficientSpace_uu(mesh));
  function_space3.reset(new vector_linear_pnp_ns_forms::CoefficientSpace_pp(mesh));
  functions_space.push_back(function_space1);
  functions_space.push_back(function_space2);
  functions_space.push_back(function_space3);


  // set PDE coefficients
  // L = 10 nm, T = 298K
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> coefficients = {
    {"permittivity", {0.019044}},
    {"diffusivity0", {1.0}},
    {"diffusivity1", {1.334/2.032}},
    {"valency0", {1.0}},
    {"valency1", {-1.0}},
    {"mu", {1.0}},
    {"penalty1", {1.0}},
    {"penalty2", {1.0}},
    {"Re", {0.01}},
  };

  std::map<std::string, std::vector<double>> sources = {};

  const std::vector<std::string> variables = {"cc","uu","pp"};

  // build problem
  Linear_PNP_NS pnp_ns_problem (
    mesh,
    function_space,
    functions_space,
    bilinear_form,
    linear_form,
    coefficients,
    sources,
    itsolver,
    pnpitsolver,
    pnpamg,
    nsitsolver,
    nsamg,
    variables
  );

  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0(output_dir+"cation_solution.pvd");
  dolfin::File solution_file1(output_dir+"anion_solution.pvd");
  dolfin::File solution_file2(output_dir+"potential_solution.pvd");
  dolfin::File solution_file3(output_dir+"velocity_solution.pvd");

  // initial guess for prescibed Dirichlet
  printf("\tInitialize Dirichlet BCs & Initial Guess\n");
  pnp_ns_problem.get_dofs();
  pnp_ns_problem.get_dofs_fasp({0,1,2},{3,4});

  pnp_ns_problem.init_BC(0.0,Lx);
  pnp_ns_problem.set_solutions(initial_guess);
  printf("\n");


  auto vec1=std::make_shared<dolfin::Constant>(-2.30258509299,0.0,1.0);
  auto vec2=std::make_shared<dolfin::Constant>(0.0,0.0,0.0);
  std::vector<dolfin::Function> solutionFn;
  solutionFn = pnp_ns_problem.get_solutions();
  auto sp_domain = std::make_shared<SpheresSubDomain>();
  dolfin::DirichletBC bc_sp0(pnp_ns_problem._functions_space[0],vec1,sp_domain);
  dolfin::DirichletBC bc_sp1(pnp_ns_problem._functions_space[1],vec2,sp_domain);
  bc_sp0.apply(*solutionFn[0].vector());
  bc_sp1.apply(*solutionFn[1].vector());
  pnp_ns_problem.set_solutions(solutionFn);

  solution_file0 << solutionFn[0][0];
  solution_file1 << solutionFn[0][1];
  solution_file2 << solutionFn[0][2];
  solution_file3 << solutionFn[1];


  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("\tInitializing nonlinear solver\n");

  // set nonlinear solver parameters
  const double initial_residual = pnp_ns_problem.compute_residual("l2");
  Newton_Status newton(
    max_newton,
    initial_residual,
    relative_residual_tol,
    max_residual_tol
  );

  printf("\tinitial residual : %10.5e\n", newton.initial_residual);
  printf("\n");

  while (newton.needs_to_iterate()) {
    // solve
    printf("\t\tSolving for Newton iterate %lu \n", newton.iteration);
    solutionFn = pnp_ns_problem.fasp_solve();

    // update newton measurements
    printf("\t\tNewton measurements for iteration :\n");
    double residual = pnp_ns_problem.compute_residual("l2");
    double max_residual = pnp_ns_problem.compute_residual("max");
    newton.update_residuals(residual, max_residual);
    newton.update_iteration();

    // output
    printf("\t\tmaximum residual :  %10.5e\n", newton.max_residual);
    printf("\t\trelative residual : %10.5e\n", newton.relative_residual);
    printf("\t\toutput solution to file...\n");
    solution_file0 << solutionFn[0][0];
    solution_file1 << solutionFn[0][1];
    solution_file2 << solutionFn[0][2];
    solution_file3 << solutionFn[1];
    printf("\n");
  }

  dolfin::File xml_mesh(output_dir+"mesh.xml");
  dolfin::File xml_file0(output_dir+"pnp_solution.xml");
  dolfin::File xml_file1(output_dir+"velocity_solution.xml");
  xml_mesh << *mesh;
  xml_file0 << solutionFn[0];
  xml_file1 << solutionFn[1];

  // check status of nonlinear solve
  if (newton.converged()) {
    printf("\tSolver succeeded!\n");
  } else {
    newton.print_status();
  }

  printf("\tSolver exiting\n"); fflush(stdout);
  return pnp_ns_problem.get_solutions();
}

#endif
