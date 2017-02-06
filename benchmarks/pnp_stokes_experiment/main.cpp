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

using namespace std;


class pnppart : public dolfin::Expression
{
public:

  pnppart() : Expression(3) {}

  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
  {
    values[0] = 0.0;
    values[1] = 0.0;
    values[2] = 0.0;
  }

};


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
  boost::filesystem::remove_all("./benchmarks/pnp_stokes_experiment/output");

  // read in parameters
  printf("Reading parameters from files...\n");
  char domain_param_filename[] = "./benchmarks/pnp_stokes_experiment/domain.dat";
  printf("\tdomain... %s\n", domain_param_filename);
  domain_param domain;
  domain_param_input(domain_param_filename, &domain);
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = domain_build(domain);
  // print_domain_param(&domain);


  // Setup FASP solver
  printf("FASP solver parameters..."); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char fasp_params[] = "./benchmarks/pnp_stokes_experiment/bcsr.dat";
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
  char fasp_pnp_params[] = "./benchmarks/pnp_stokes_experiment/bsr.dat";
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
  char fasp_ns_params[] = "./benchmarks/pnp_stokes_experiment/ns.dat";
  fasp_ns_param_input(fasp_ns_params, &ns_inpar);
  fasp_ns_param_init(&ns_inpar, &ns_itpar, &ns_amgpar, &ns_ilupar, &ns_schpar);
  printf("done\n"); fflush(stdout);



  //-------------------------
  // Construct Problem
  //-------------------------
  printf("\nConstruct vector PNP problem\n");

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
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> coefficients = {
    {"permittivity", {1.0}},
    {"diffusivity", {1.0, 1.0}},
    {"valency", {1.0, -1.0}},
    {"mu", {0.1}},
    {"penalty1", {1.0}},
    {"penalty2", {1.0}},
  };

  std::map<std::string, std::vector<double>> sources = {
    {"fixed_charge", {0.0}}
  };

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
    itpar,
    pnp_itpar,
    pnp_amgpar,
    ns_itpar,
    ns_amgpar,
    variables
  );

  // pnp_ns_problem.set_coefficients(
    // coefficients
    // pnp_source_fns
  // );

  // bool plot_coefficients = false;
  // if (plot_coefficients) {
  //   printf("\toutput coefficients to file\n");
  //   dolfin::File permittivity_file("./benchmarks/pnp_stokes_experiment/output/permittivity.pvd");
  //   dolfin::File charges_file("./benchmarks/pnp_stokes_experiment/output/charges.pvd");
  //   dolfin::File diffusivity_file("./benchmarks/pnp_stokes_experiment/output/diffusivity.pvd");
  //   dolfin::File valency_file("./benchmarks/pnp_stokes_experiment/output/valency.pvd");
  //   permittivity_file << permittivity;
  //   charges_file << charges;
  //   diffusivity_file << diffusivity[1];
  //   diffusivity_file << diffusivity[2];
  //   diffusivity_file << diffusivity[3];
  //   valency_file << valency[1];
  //   valency_file << valency[2];
  //   valency_file << valency[3];
  // }




  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0("./benchmarks/pnp_stokes_experiment/output/cation_solution.pvd");
  dolfin::File solution_file1("./benchmarks/pnp_stokes_experiment/output/anion_solution.pvd");
  dolfin::File solution_file2("./benchmarks/pnp_stokes_experiment/output/potential_solution.pvd");
  dolfin::File solution_file3("./benchmarks/pnp_stokes_experiment/output/velocity_solution.pvd");

  // initial guess for prescibed Dirichlet
  printf("Initialize Dirichlet BCs & Initial Guess\n");
  pnp_ns_problem.get_dofs();
  pnp_ns_problem.get_dofs_fasp({2,0,1},{3,4});

  pnp_ns_problem.init_BC(0.0);
  std::vector<Linear_Function> InitialGuess;
  Linear_Function PNP(0,-5.0,5.0,{0.0,-1.0,1.0},{-1.0,0.0,-1.0});
  Linear_Function Vel(0,-5.0,5.0,{0.0,0.0,0.0},{0.0,0.0,0.0});
  Linear_Function Pres(0,-5.0,5.0,0.0,0.0);
  InitialGuess.push_back(PNP);
  InitialGuess.push_back(Vel);
  InitialGuess.push_back(Pres);
  pnp_ns_problem.set_solutions(InitialGuess);
  printf("\n");

  std::vector<dolfin::Function> solutionFn;
  solutionFn = pnp_ns_problem.get_solutions();
  solution_file0 << solutionFn[0][0];
  solution_file1 << solutionFn[0][1];
  solution_file2 << solutionFn[0][2];
  solution_file3 << solutionFn[1];

  // return 0;

  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  const std::size_t max_newton = 5;
  const double max_residual_tol = 1.0e-10;
  const double relative_residual_tol = 1.0e-4;
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
    printf("Solving for Newton iterate %lu \n", newton.iteration);
    solutionFn = pnp_ns_problem.fasp_solve();

    // update newton measurements
    printf("Newton measurements for iteration :\n");
    double residual = pnp_ns_problem.compute_residual("l2");
    double max_residual = pnp_ns_problem.compute_residual("max");
    newton.update_residuals(residual, max_residual);
    newton.update_iteration();

    // output
    printf("\tmaximum residual :  %10.5e\n", newton.max_residual);
    printf("\trelative residual : %10.5e\n", newton.relative_residual);
    printf("\toutput solution to file...\n");
    solution_file0 << solutionFn[0][0];
    solution_file1 << solutionFn[0][1];
    solution_file2 << solutionFn[0][2];
    solution_file3 << solutionFn[1];
    printf("\n");
  }


  // check status of nonlinear solve
  if (newton.converged()) {
    printf("Solver succeeded!\n");
  } else {
    newton.print_status();
  }

  printf("Solver exiting\n"); fflush(stdout);
  return 0;
}
