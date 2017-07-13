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
#include "norm_pnp.h"

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
  boost::filesystem::remove_all("./benchmarks/pnp_pb/output");

  // read in parameters
  printf("Reading parameters from files...\n");
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = dolfin::Mesh("./benchmarks/pnp_pb/mesh1.xml.gz");
  double L=0.2;
  double Lx=L,Ly=L,Lz=L;
  // print_domain_param(&domain);


  char fasp_params[] = "./benchmarks/pnp_pb/bsr.dat";
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
  bool use_eafe_approximation = false;


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
  double Eps = 1E-6;
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

  dolfin::Function phib(pnp_problem.phib_space);
  PhibExpression bdexpr(Eps);
  phib.interpolate(bdexpr);

  // dolfin::Function charges(pnp_problem.fixed_charge_space);
  // dolfin::Constant fc_expr (0.0);
  // charges.interpolate(fc_expr);

  std::map<std::string, dolfin::Function> pnp_source_fns  = { {"phib", phib}  };
  std::map<std::string, dolfin::Function> emptymap = {};
  pnp_problem.set_coefficients(    emptymap,    pnp_source_fns  );

  dolfin::File phib_file("./benchmarks/pnp_pb/output/phib.pvd");
  dolfin::File phib_file_xml("./benchmarks/pnp_pb/output/phib.xml");
  phib_file << phib;
  phib_file_xml << phib;
  // std::string ss = phib.str(true);
  std::cout << "Phib = " << phib.vector()->norm("l2") << phib.vector()->max() << phib.vector()->min() << std::endl;


  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0("./benchmarks/pnp_pb/output/solution_phi.pvd");
  dolfin::File solution_file1("./benchmarks/pnp_pb/output/solution_eta1.pvd");
  dolfin::File solution_file2("./benchmarks/pnp_pb/output/solution_eta2.pvd");

  dolfin::File xml_filePhipb("./benchmarks/pnp_pb/output/solution_phipb.xml");
  dolfin::File xmlSolution("./benchmarks/pnp_pb/output/solution.xml");
  dolfin::File xml_file0("./benchmarks/pnp_pb/output/solution_phi.xml");
  dolfin::File xml_file1("./benchmarks/pnp_pb/output/solution_eta1.xml");
  dolfin::File xml_file2("./benchmarks/pnp_pb/output/solution_eta2.xml");

  // initial guess for prescibed Dirichlet
  printf("Record interpolant for given Dirichlet BCs (initial guess for solution)\n");
  pnp_problem.init_BC();
  std::vector<double> initvector = {-1.0,1.0,-1.0};
  pnp_problem.set_solution(initvector);

  dolfin::Function solutionFn = pnp_problem.get_solution();


  // auto PreviousMesh = std::make_shared<dolfin::Mesh>("./benchmarks/pnp_pb/previous_solution/mesh1.xml.gz");
  // auto PreviousV=std::make_shared<vector_linear_pnp_forms::FunctionSpace>(PreviousMesh);
  // dolfin::Function PreviousSolution(PreviousV,"./benchmarks/pnp_pb/previous_solution/solution.xml");
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
  const std::size_t max_newton = 5;
  const double max_residual_tol = 1.0e-11;
  const double relative_residual_tol = 1.0e-8;
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
    dolfin::Function phi(solutionFn[0]);
    xml_file0 << phi;
    xml_filePhipb << phib;
    xmlSolution << solutionFn;
    // xml_file1 << solutionFn[1];
    // xml_file2 << solutionFn[2];
    printf("\n");
  }


  // check status of nonlinear solve
  if (newton.converged()) {
    printf("Solver succeeded!\n");
  } else {
    newton.print_status();
  }

  ExactExpression ExExp(Eps);
  auto ExSol = std::make_shared<dolfin::Function>(solutionFn);
  auto Sol = std::make_shared<dolfin::Function>(solutionFn);
  ExSol->interpolate(ExExp);
  Error Err(ExSol);
  double L2Err = Err.compute_l2_error(Sol);
  double H1Err = Err.compute_semi_h1_error(Sol);

  norm_pnp::Functional Fc(mesh);
  auto sphib = std::make_shared<dolfin::Function>(phib);
  auto perm = std::make_shared<dolfin::Constant>(Eps);
  auto diff = std::make_shared<dolfin::Constant>(1.0);
  auto val = std::make_shared<dolfin::Constant>(1.0);
  Fc.uu = Sol;
  Fc.permittivity = perm;
  Fc.diffusivity = diff;
  Fc.valency = val;
  Fc.phib = sphib;
  double ErrFlux = assemble(Fc);

  printf("L2 Error = %f , semi H1 Error = %f , Flux Error = %f, Mesh size  = %f\n",L2Err,H1Err,ErrFlux,mesh->hmax()); fflush(stdout);
  printf("%f &  %f & %f & %f \\\\ \n",L2Err,H1Err,ErrFlux,mesh->hmax()); fflush(stdout);

  printf("Solver exiting\n"); fflush(stdout);
  return 0;
}
