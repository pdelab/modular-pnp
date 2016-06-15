/// Main file for solving the linearized PNP problem
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "domain.h"
#include "dirichlet.h"
#include "vector_pnp.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

using namespace std;

int main (int argc, char** argv)
{
  printf("----------------------------------------------------\n");
  printf(" Setting up the linearized PNP problem\n");
  printf("----------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  dolfin::parameters["linear_algebra_backend"] = "Eigen";
  dolfin::parameters["allow_extrapolation"] = true;

  // read in parameters
  printf("Reading parameters from files...\n");

  char domain_param_filename[] = "./benchmarks/PNP/domain_params.dat";
  printf("\tdomain... %s\n", domain_param_filename);
  domain_param domain;
  domain_param_input(domain_param_filename, &domain);
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = domain_build(domain);
  // print_domain_param(&domain);

  char fasp_params[] = "./benchmarks/PNP/bsr.dat";
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
  printf("\n");


  // construct the PDE
  printf("Define coefficients\n");
  std::map<std::string, std::vector<double>> poisson_coefficients = {
    {"permittivity", {1.0}},
    {"fixed_charge", {1.0}},
    {"diffusivity", {0.0, 2.0, 2.0, 10.0}},
    {"valency", {0.0, 1.0, -1.0, -1.0}}
  };

  printf("\nConstructing the vector Poisson problem\n");
  Vector_PNP::Vector_PNP pnp_problem (
    mesh,
    domain,
    poisson_coefficients,
    itsolver,
    amg
  );
  printf("\tconstructed poisson problem\n\n");
  pnp_problem.print_coefficients();
  printf("\n");

  dolfin::File solution_file0("./benchmarks/linear_pnp_vector/output/1solution.pvd");
  dolfin::File solution_file1("./benchmarks/linear_pnp_vector/output/2solution.pvd");
  dolfin::File solution_file2("./benchmarks/linear_pnp_vector/output/3solution.pvd");
  dolfin::File solution_file3("./benchmarks/linear_pnp_vector/output/4solution.pvd");

  dolfin::Function solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  pnp_problem.set_solution({
    1.0,
    std::log(2.0),
    std::log(2.0),
    std::log(1.0)
  });
  solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  printf("Set Dirichlet BCs\n\n");
  std::vector<std::size_t> components = {0, 0, 0, 0};
  std::vector<std::vector<double>> bcs;
  bcs.push_back({0.0,  1.0});
  bcs.push_back({std::log(1.0), std::log(2.0)});
  bcs.push_back({std::log(1.5), std::log(1.0)});
  bcs.push_back({std::log(0.5), std::log(2.0)});

  pnp_problem.set_DirichletBC(components, bcs);
  solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function dolfin_solution(pnp_problem.dolfin_solve());
  solution_file0 << dolfin_solution[0];
  solution_file1 << dolfin_solution[1];
  solution_file2 << dolfin_solution[2];
  solution_file3 << dolfin_solution[3];

  // pnp_problem.set_DirichletBC(components, bcs);
  // dolfin::Function la_solution(pnp_problem.la_solve());
  // solution_file0 << la_solution[0];
  // solution_file1 << la_solution[1];
  // solution_file2 << la_solution[2];
  // solution_file3 << la_solution[3];

  pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function fasp_solution(pnp_problem.fasp_solve());
  solution_file0 << fasp_solution[0];
  solution_file1 << fasp_solution[1];
  solution_file2 << fasp_solution[2];
  solution_file3 << fasp_solution[3];

  printf("Done\n\n"); fflush(stdout);
  pnp_problem.free_fasp();

  return 0;
}
