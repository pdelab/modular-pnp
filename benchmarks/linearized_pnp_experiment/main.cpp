/// Main file for solving the linearized PNP problem
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <dolfin.h>
#include "pde.h"
#include "domain.h"
#include "dirichlet.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"
#include "linear_pnp.h"

using namespace std;

class Permittivity_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = 1.0;
    }
};

class Fixed_Charged_Expression : public dolfin::Expression {
  public:
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = 1.0;
    }
};

class Diffusivity_Expression : public dolfin::Expression {
  public:
    Diffusivity_Expression() : dolfin::Expression(4) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] = 1.0; // potential diffusivity is not used
      values[1] = 1.0;
      values[2] = 1.0;
      values[3] = 1.0;
    }
};

class Valency_Expression : public dolfin::Expression {
  public:
    Valency_Expression() : dolfin::Expression(4) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] =  0.0; // potential valency is not used
      values[1] =  1.0;
      values[2] = -1.0;
      values[3] = -1.0;
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

  // read in parameters
  printf("Reading parameters from files...\n");
  char domain_param_filename[] = "./benchmarks/linearized_pnp_experiment/domain.dat";
  printf("\tdomain... %s\n", domain_param_filename);
  domain_param domain;
  domain_param_input(domain_param_filename, &domain);
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh);
  *mesh = domain_build(domain);
  // print_domain_param(&domain);


  char fasp_params[] = "./benchmarks/linearized_pnp_experiment/bsr.dat";
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
  printf("Initialize coefficients\n");
  std::map<std::string, std::vector<double>> pnp_coefficients = {
    {"permittivity", {1.0}},
    {"diffusivity", {0.0, 2.0, 2.0, 10.0}},
    {"valency", {0.0, 1.0, -1.0, -1.0}}
  };
  std::map<std::string, std::vector<double>> pnp_sources = {
    {"fixed_charge", {1.0}}
  };

  // build problem
  Linear_PNP::Linear_PNP pnp_problem (
    mesh,
    function_space,
    bilinear_form,
    linear_form,
    pnp_coefficients,
    pnp_sources,
    itsolver,
    amg
  );

  printf("Define PNP coefficients from expressions\n");
  dolfin::Function permittivity(pnp_problem.fixed_charge_space);
  Permittivity_Expression permittivity_expr;
  permittivity.interpolate(permittivity_expr);

  dolfin::Function charges(pnp_problem.fixed_charge_space);
  Fixed_Charged_Expression fc_expr;
  charges.interpolate(fc_expr);
  dolfin::Function diffusivity(pnp_problem.diffusivity_space);
  Diffusivity_Expression diff_expr;
  diffusivity.interpolate(diff_expr);

  dolfin::Function valency(pnp_problem.valency_space);
  Valency_Expression valency_expr;
  valency.interpolate(valency_expr);

  std::map<std::string, dolfin::Function> pnp_coefficient_fns = {
    {"permittivity", permittivity},
    {"diffusivity", diffusivity},
    {"valency", valency}
  };
  std::map<std::string, dolfin::Function> pnp_source_fns = {
    {"fixed_charge", charges}
  };

  pnp_problem.set_coefficients(
    pnp_coefficient_fns,
    pnp_source_fns
  );

  bool plot_coefficients = false;
  if (plot_coefficients) {
    printf("\toutput coefficients to file\n");
    dolfin::File permittivity_file("./benchmarks/linearized_pnp_experiment/output/permittivity.pvd");
    dolfin::File charges_file("./benchmarks/linearized_pnp_experiment/output/charges.pvd");
    dolfin::File diffusivity_file("./benchmarks/linearized_pnp_experiment/output/diffusivity.pvd");
    dolfin::File valency_file("./benchmarks/linearized_pnp_experiment/output/valency.pvd");
    permittivity_file << permittivity;
    charges_file << charges;
    diffusivity_file << diffusivity[1];
    diffusivity_file << diffusivity[2];
    diffusivity_file << diffusivity[3];
    valency_file << valency[1];
    valency_file << valency[2];
    valency_file << valency[3];
  }




  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0("./benchmarks/linearized_pnp_experiment/output/1solution.pvd");
  dolfin::File solution_file1("./benchmarks/linearized_pnp_experiment/output/2solution.pvd");
  dolfin::File solution_file2("./benchmarks/linearized_pnp_experiment/output/3solution.pvd");
  dolfin::File solution_file3("./benchmarks/linearized_pnp_experiment/output/4solution.pvd");

  // default solution
  printf("Record default initial guess for solution\n");
  dolfin::Function solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  // prescribed constant solution
  printf("Record user-set initial guess for solution\n");
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

  // initial guess for prescibed Dirichlet
  printf("Record interpolant for given Dirichlet BCs (initial guess for solution)\n");
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


  // computed solution for prescribed Dirichlet
  // printf("\tcomputed linearized solution using dolfin solver\n\t");
  // const clock_t begin_time = clock();
  // pnp_problem.set_DirichletBC(components, bcs);
  // dolfin::Function dolfin_solution(pnp_problem.dolfin_solve());
  // solution_file0 << dolfin_solution[0];
  // solution_file1 << dolfin_solution[1];
  // solution_file2 << dolfin_solution[2];
  // solution_file3 << dolfin_solution[3];
  // const float dolfin_runtime = float(clock () - begin_time) / CLOCKS_PER_SEC;
  // printf("\truntime : %e seconds\n", dolfin_runtime);
  // printf("\n");


  // FASP computed solution using basic Newton iteration
  // printf("\tcomputed linearized solution using FASP solver\n");
  // pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function fasp_solution(pnp_problem.get_solution());
  // fasp_solution = pnp_problem.fasp_solve();
  // solution_file0 << fasp_solution[0];
  // solution_file1 << fasp_solution[1];
  // solution_file2 << fasp_solution[2];
  // solution_file3 << fasp_solution[3];
  // printf("\n");



  // Measure error of FASP computed solution for random RHS
  printf("Measure error of FASP computed solution\n");
  pnp_problem.set_DirichletBC(components, bcs);
  std::size_t problem_size = fasp_solution.vector()->size();

  srand(time(NULL));
  std::vector<double> random_values;
  random_values.reserve(problem_size);
  for (uint i = 0; i < problem_size; i++) {
    random_values[i] = (double) rand() / RAND_MAX;
  }

  dolfin::EigenVector target_vector(problem_size);
  target_vector.set_local(random_values);

  dolfin::EigenVector fasp_vector(problem_size);
  fasp_vector = pnp_problem.fasp_test_solver(target_vector);

  dolfin::EigenVector error_vector(problem_size);
  error_vector = target_vector;
  error_vector -= fasp_vector;

  const double linear_error = error_vector.norm("l2");
  printf("\tFASP error in the l2-sense: %e\n", linear_error);
  printf("\taverage FASP error entrywise: %e\n",
    linear_error / ((double) problem_size)
  );

  printf("Done\n\n"); fflush(stdout);
  return 0;
}
