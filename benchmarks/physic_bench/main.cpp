/// Main file for solving the linearized PNP problem
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <dolfin.h>
#include "pde.h"
#include "error.h"
#include "newton_status.h"
#include "domain.h"
#include "dirichlet.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "linear_pnp.h"

using namespace std;


int main (int argc, char** argv) {
  printf("\n");
  printf("----------------------------------------------------\n");
  printf(" Setting up the linearized PNP problem\n");
  printf("----------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  dolfin::parameters["linear_algebra_backend"] = "Eigen";
  // dolfin::parameters["allow_extrapolation"] = true;

  // Deleting the folders:
  boost::filesystem::remove_all("./benchmarks/physic_bench/output");

  // read in parameters
  printf("Reading parameters from files...\n");
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = dolfin::Mesh("./benchmarks/physic_bench/mesh1.xml.gz");
  double Lx=20.0,Ly=2.0,Lz=2.0;


  char fasp_params[] = "./benchmarks/physic_bench/bsr.dat";
  printf("\tFASP parameters... %s\n", fasp_params);
  input_param input;
  itsolver_param itsolver;
  AMG_param amg;
  ILU_param ilu;
  fasp_param_input(fasp_params, &input);
  fasp_param_init(
    &input,
    &itsolver,
    &amg,
    &ilu,
    NULL
  );
  bool use_eafe_approximation = true;


  //-------------------------
  // Construct Problem
  //-------------------------
  printf("\nConstruct vector PNP problem\n");

  // setup function spaces and forms
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


  // set PDE coefficients
  double Eps = 1E-3;
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> pnp_coefficients = {
    {"permittivity", {Eps}},
    {"diffusivity", {0.0, 1.0, 1.0}},
    {"valency", {0.0, 1.0, -1.0}}
  };
  std::map<std::string, std::vector<double>> pnp_sources = {
    {"fixed_charge", {0.0}},
    {"g", {100.0*Eps}}
  };

  // build problem
  Linear_PNP pnp_problem (
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

  if (use_eafe_approximation) {
    printf("Setting solver to use EAFE approximation\n");
    pnp_problem.use_eafe();
  }

  // dolfin::Function charges(pnp_problem.fixed_charge_space);
  // dolfin::Constant fc_expr (0.0);
  // charges.interpolate(fc_expr);



  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0("./benchmarks/physic_bench/output/solution_phi.pvd");
  dolfin::File solution_file1("./benchmarks/physic_bench/output/solution_eta1.pvd");
  dolfin::File solution_file2("./benchmarks/physic_bench/output/solution_eta2.pvd");

  dolfin::File xml_filePhipb("./benchmarks/physic_bench/output/solution_phipb.xml");
  dolfin::File xmlSolution("./benchmarks/physic_bench/output/solution.xml");
  dolfin::File xml_file0("./benchmarks/physic_bench/output/solution_phi.xml");
  dolfin::File xml_file1("./benchmarks/physic_bench/output/solution_eta1.xml");
  dolfin::File xml_file2("./benchmarks/physic_bench/output/solution_eta2.xml");

  // initial guess for prescibed Dirichlet
  printf("Record interpolant for given Dirichlet BCs (initial guess for solution)\n");
  pnp_problem.init_BC(Lx,Ly,Lz);
  pnp_problem.init_measure(mesh,Lx,Ly,Lz);
  // std::vector<double> initvector = {-1.0,1.0,-1.0};

  Linear_Function Phi(0,-Lx/2.0,Lx/2.0,-1.0,1.0);
  Linear_Function Eta1(0,-Lx/2.0,Lx/2.0,0.0,-2.30258509299);
  Linear_Function Eta2(0,-Lx/2.0,Lx/2.0,-2.30258509299,0.0);
  std::vector <Linear_Function> initial_guess;
  initial_guess.push_back(Phi);
  initial_guess.push_back(Eta1);
  initial_guess.push_back(Eta2);

  pnp_problem.set_solution(initial_guess);

  dolfin::Function solutionFn = pnp_problem.get_solution();


  // auto PreviousMesh = std::make_shared<dolfin::Mesh>("./benchmarks/physic_bench/previous_solution/mesh1.xml.gz");
  // auto PreviousV=std::make_shared<vector_linear_pnp_forms::FunctionSpace>(PreviousMesh);
  // dolfin::Function PreviousSolution(PreviousV,"./benchmarks/physic_bench/previous_solution/solution.xml");
  // solutionFn.interpolate(PreviousSolution);
  // pnp_problem.set_solution(solutionFn);

  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  printf("\n");


  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  const std::size_t max_newton = 20;
  const double max_residual_tol = 1.0e-10;
  const double relative_residual_tol = 1.0e-10;
  const double initial_residual = pnp_problem.compute_residual("l2");
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
    solutionFn = pnp_problem.fasp_solve();

    // update newton measurements
    printf("Newton measurements for iteration :\n");
    double residual = pnp_problem.compute_residual("l2");
    double max_residual = pnp_problem.compute_residual("max");
    newton.update_residuals(residual, max_residual);
    newton.update_iteration();

    // output
    printf("\tmaximum residual :  %10.5e\n", newton.max_residual);
    printf("\trelative residual : %10.5e\n", newton.relative_residual);
    printf("\toutput solution to file...\n");
    solution_file0 << solutionFn[0];
    solution_file1 << solutionFn[1];
    solution_file2 << solutionFn[2];
    xmlSolution << solutionFn;
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
