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
#include "error.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
}

#include "vector_linear_pnp_ns_forms.h"
#include "linear_pnp_ns.h"

using namespace std;

int main (int argc, char** argv) {
  printf("\n");
  printf("----------------------------------------------------\n");
  printf(" Setting up the PNP+Stokes problem\n");
  printf("----------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  dolfin::parameters["linear_algebra_backend"] = "Eigen";
  dolfin::parameters["allow_extrapolation"] = true;
  std::vector<double> RelErr;

  // Deleting the folders:
  boost::filesystem::remove_all("./benchmarks/physic_bench/output_NS");

  // read in parameters
  printf("Reading parameters from files...\n");
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = dolfin::Mesh("./benchmarks/physic_bench/mesh3.xml.gz");
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
  double Eps = 0.02099601;
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> coefficients = {
    {"permittivity", {Eps}},
    {"diffusivity0", {1.0}},
    {"diffusivity1", {1.0}},
    {"valency0", {1.0}},
    {"valency1", {-1.0}},
    {"mu", {1.03432589}},
    {"penalty1", {1.0}},
    {"penalty2", {1.0}},
    {"Re", {3.5959E-3}},
  };
  std::map<std::string, std::vector<double>> sources = {{"g",{0.1602176621}}};

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

  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0("./benchmarks/physic_bench/output_NS/potential_solution.pvd");
  dolfin::File solution_file1("./benchmarks/physic_bench/output_NS/cation_solution.pvd");
  dolfin::File solution_file2("./benchmarks/physic_bench/output_NS/anion_solution.pvd");
  dolfin::File solution_file3("./benchmarks/physic_bench/output_NS/velocity_solution.pvd");

  // initial guess for prescibed Dirichlet
  printf("Initialize Dirichlet BCs & Initial Guess\n");
  pnp_ns_problem.get_dofs();
  pnp_ns_problem.get_dofs_fasp({0,1,2},{3,4});
  pnp_ns_problem.init_BC (Lx,Ly,Lz);
  pnp_ns_problem.init_measure (mesh,Lx,Ly,Lz);

  // From PNP
  auto mesh_PNP = std::make_shared<dolfin::Mesh>("./benchmarks/physic_bench/output_PNP_6/accepted_mesh.xml.gz");
  auto CG = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_cc>(mesh_PNP);
  // auto RT = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_uu>(mesh_PNP);
  // auto DG = std::make_shared<vector_linear_pnp_ns_forms::CoefficientSpace_pp>(mesh_PNP);
  dolfin::Function pnp_solution(CG,"./benchmarks/physic_bench/output_PNP_6/accepted_solution.xml");
  //
  dolfin::Function pnp_init(pnp_ns_problem._functions_space[0]);
  dolfin::Function u_init(pnp_ns_problem._functions_space[1]);
  dolfin::Function p_init(pnp_ns_problem._functions_space[2]);
  // // VelExpression VelE;
  // // auto sp_domain = std::make_shared<SphereSubDomain>();
  // // dolfin::DirichletBC bc_sp1(pnp_ns_problem._functions_space[1],ub,sp_domain);
  // auto initpnp = std::make_shared<dolfin::Constant>(-1.0,0.0,-2.0);
  auto initvel = std::make_shared<dolfin::Constant>(0.0,0.0,0.0);
  auto initp = std::make_shared<dolfin::Constant>(0.0);
  pnp_init.interpolate(pnp_solution);
  u_init.interpolate(*initvel);
  p_init.interpolate(*initp);


  // Previous case
  // auto initpnp = std::make_shared<dolfin::Constant>(1.0,-1.0,-1.0);
  // auto initvel = std::make_shared<dolfin::Constant>(0.0,0.0,-0.0);
  // auto initp = std::make_shared<dolfin::Constant>(0.0);
  //
  // dolfin::Function pnp_init(pnp_ns_problem._functions_space[0]);
  // dolfin::Function u_init(pnp_ns_problem._functions_space[1]);
  // dolfin::Function p_init(pnp_ns_problem._functions_space[2]);
  //
  // pnp_init.interpolate(*initpnp);
  // u_init.interpolate(*initvel);
  // p_init.interpolate(*initp);
  // auto sp_domain = std::make_shared<SphereSubDomain>();
  // std::vector<std::size_t> v1 = {0,1,2};
  // std::vector<double> v2 = {-Lx/2.0,-Ly/2.0,-Lz/2.0};
  // std::vector<double> v3 = { Lx/2.0,Ly/2.0,Lz/2.0};
  // auto BCdomain_xyz = std::make_shared<Dirichlet_Subdomain>(v1,v2,v3,1E-5);
  // dolfin::DirichletBC bc_sp1(pnp_ns_problem._functions_space[1],ub,sp_domain);
  // dolfin::DirichletBC bc_sp2(pnp_ns_problem._functions_space[1],ub,BCdomain_xyz);
  // bc_sp1.apply(*u_init.vector());
  // bc_sp2.apply(*u_init.vector());

  std::vector<dolfin::Function> solutionFn;
  solutionFn.push_back(pnp_init);
  solutionFn.push_back(u_init);
  solutionFn.push_back(p_init);
  pnp_ns_problem.set_solutions(solutionFn);

  // CASE 2
  // std::vector<dolfin::Function> solutionFn2;
  // dolfin::Function pnp_init2(pnp_ns_problem._functions_space[0],"./benchmarks/physic_bench/DATA/pnp_solution.xml");
  // dolfin::Function u_init2(pnp_ns_problem._functions_space[1],"./benchmarks/physic_bench/DATA/velocity_solution.xml");
  // dolfin::Function p_init2(pnp_ns_problem._functions_space[2],"./benchmarks/physic_bench/DATA/pressure_solution.xml");
  // solutionFn2.push_back(pnp_init2);
  // solutionFn2.push_back(u_init2);
  // solutionFn2.push_back(p_init2);


  solution_file0 << solutionFn[0][0];
  solution_file1 << solutionFn[0][1];
  solution_file2 << solutionFn[0][2];
  solution_file3 << solutionFn[1];
  printf("\n");

  dolfin::File xml_pnp("./benchmarks/physic_bench/DATA/pnp_solution.xml");
  dolfin::File xml_vel("./benchmarks/physic_bench/DATA/velocity_solution.xml");
  dolfin::File xml_pressure("./benchmarks/physic_bench/DATA/pressure_solution.xml");


  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  const std::size_t max_newton = 5;
  const double max_residual_tol = 1.0e-8;
  const double relative_residual_tol = 1.0e-8;
  const double initial_residual = pnp_ns_problem.compute_residual("l2");
  Newton_Status newton(
    max_newton,
    initial_residual,
    relative_residual_tol,
    max_residual_tol
  );

  printf("\tinitial residual : %10.5e\n", newton.initial_residual);
  printf("\n");

  // pnp_ns_problem.set_solutions(solutionFn2);


  // while (newton.iteration<=max_newton) {
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

    xml_pnp << solutionFn[0];
    xml_vel << solutionFn[1];
    xml_pressure<< solutionFn[2];
    RelErr.push_back(newton.relative_residual);

  }


  // check status of nonlinear solve
  if (newton.converged()) {
    printf("Solver succeeded!\n");
  } else {
    newton.print_status();
  }

  dolfin::File xml_mesh("./benchmarks/physic_bench/output_NS/mesh.xml");
  dolfin::File xml_file0("./benchmarks/physic_bench/output_NS/pnp_solution.xml");
  dolfin::File xml_file1("./benchmarks/physic_bench/output_NS/velocity_solution.xml");
  xml_mesh << *mesh;
  xml_file0 << solutionFn[0];
  xml_file1 << solutionFn[1];

  std::cout << "[";
  for (std::vector<double>::const_iterator i = RelErr.begin(); i != RelErr.end(); ++i)
    std::cout << *i  << ",";
  std::cout << "]";
  std::cout << std::endl;


  printf("Solver exiting\n"); fflush(stdout);
  return 0;
}
