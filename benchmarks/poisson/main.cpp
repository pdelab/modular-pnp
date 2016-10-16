/// Main file for solving the Poisson problem
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "domain.h"
#include "dirichlet.h"
#include "poisson.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

using namespace std;

int main (int argc, char** argv)
{
  printf("----------------------------------------------------\n");
  printf(" Setting up the Poisson problem\n");
  printf("----------------------------------------------------\n\n");
  fflush(stdout);

  // Need to use Eigen for linear algebra
  dolfin::parameters["linear_algebra_backend"] = "Eigen";
  dolfin::parameters["allow_extrapolation"] = true;

  // read in parameters
  printf("Reading parameters from files...\n");

  char domain_param_filename[] = "./benchmarks/poisson/domain.dat";
  printf("\tdomain... %s\n", domain_param_filename);
  domain_param domain;
  domain_param_input(domain_param_filename, &domain);
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = domain_build(domain);
  // print_domain_param(&domain);

  char fasp_params[] = "./benchmarks/poisson/bsr.dat";
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
  std::map<std::string, double> poisson_coefficients = {
    {"diffusivity", 1.0},
    {"reactivity", 0.0},
    {"source", 1.0e-2}
  };
  std::map<std::string, double>::iterator coeff;
  for (coeff = poisson_coefficients.begin(); coeff != poisson_coefficients.end(); ++coeff) {
    printf("\t%s will be set to %e\n", coeff->first.c_str(), coeff->second);
  }

  printf("\nConstructing the Poisson problem\n");
  Poisson::Poisson poisson_problem (
    mesh,
    domain,
    poisson_coefficients,
    itsolver,
    amg
  );
  printf("\tconstructed poisson problem\n\n");
  poisson_problem.print_coefficients();
  printf("\n");

  dolfin::File solution_file("./benchmarks/poisson/output/solution.pvd");
  solution_file << poisson_problem.get_solution();

  poisson_problem.set_solution(1.0);
  solution_file << poisson_problem.get_solution();

  poisson_problem.set_DirichletBC(0, 0.0, 1.0);
  solution_file << poisson_problem.get_solution();

  dolfin::Function solution(poisson_problem.dolfin_solve());
  solution_file << solution;

  return 0;
}
