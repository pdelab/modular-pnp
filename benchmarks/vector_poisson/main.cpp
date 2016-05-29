/// Main file for solving the Poisson problem
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "domain.h"
#include "dirichlet.h"
#include "vector_poisson.h"
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
    {"diffusivity", {1.0, 2.0, 2.0, 10.0}},
    {"reactivity", {1.0, 0.0, 0.0, 2.0}},
    {"source", {1.0e-2, 1.0, 10.0, 1.0e-2}}
  };
  std::map<std::string, std::vector<double>>::iterator coeff;
  for (coeff = poisson_coefficients.begin(); coeff != poisson_coefficients.end(); ++coeff) {
    printf("\t%s will be set to [ %e, %e, %e ]\n", coeff->first.c_str(), coeff->second[0], coeff->second[1], coeff->second[2]);
  }

  printf("\nConstructing the vector Poisson problem\n");
  Vector_Poisson::Vector_Poisson poisson_problem (
    mesh,
    domain,
    poisson_coefficients,
    itsolver,
    amg
  );
  printf("\tconstructed poisson problem\n\n");
  poisson_problem.print_coefficients();
  printf("\n");

  dolfin::File solution_file0("./benchmarks/vector_poisson/1solution.pvd");
  dolfin::File solution_file1("./benchmarks/vector_poisson/2solution.pvd");
  dolfin::File solution_file2("./benchmarks/vector_poisson/3solution.pvd");
  dolfin::File solution_file3("./benchmarks/vector_poisson/4solution.pvd");

  dolfin::Function solutionFn = poisson_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  poisson_problem.set_solution({1.0, 0.0, 2.0, -1.0});
  solutionFn = poisson_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  printf("Set Dirichlet BCs\n\n");
  std::vector<std::size_t> components = {0, 1, 0, 2};
  std::vector<std::vector<double>> bcs;
  bcs.push_back({ 0.0,  1.0});
  bcs.push_back({-1.0,  1.0});
  bcs.push_back({ 0.0, 10.0});
  bcs.push_back({ 5.0, -5.0});

  poisson_problem.set_DirichletBC(components, bcs);
  solutionFn = poisson_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  dolfin::Function solution(poisson_problem.dolfin_solve());
  solution_file0 << solution[0];
  solution_file1 << solution[1];
  solution_file2 << solution[2];
  solution_file3 << solution[3];

  return 0;
}
