/// Main file for solving the linearized PNP problem
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
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
      values[0] = x[0] < 0 ? 0.1 : 1.0;
      values[1] = x[0] < 0 ? 0.1 : 1.0;
      values[2] = x[0] < 0 ? 0.1 : 1.0;
      values[3] = x[0] < 0 ? 0.1 : 1.0;
    }
};

class Valency_Expression : public dolfin::Expression {
  public:
    Valency_Expression() : dolfin::Expression(4) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      values[0] =  0.0;
      values[1] =  1.0;
      values[2] = -1.0;
      values[3] = -1.0;
    }
};

int main (int argc, char** argv) {
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



  //-------------------------
  // Construct Problem
  //-------------------------

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
  printf("Define coefficients\n");
  std::map<std::string, std::vector<double>> pnp_coefficients = {
    {"permittivity", {1.0}},
    {"diffusivity", {0.0, 2.0, 2.0, 10.0}},
    {"valency", {0.0, 1.0, -1.0, -1.0}}
  };
  std::map<std::string, std::vector<double>> pnp_sources = {
    {"fixed_charge", {1.0}}
  };

  // build problem
  printf("\nConstructing the vector PNP problem\n");
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
  printf("\tconstructed PNP problem\n\n");
  pnp_problem.print_coefficients();
  printf("\n");


  printf("\tdefining PNP coefficients from expressions\n\n");
  dolfin::Function permittivity(pnp_problem.fixed_charge_space);
  dolfin::File permittivity_file("./benchmarks/problem/output/permittivity.pvd");
  Permittivity_Expression permittivity_expr;
  permittivity.interpolate(permittivity_expr);
  permittivity_file << permittivity;

  dolfin::Function charges(pnp_problem.fixed_charge_space);
  dolfin::File charges_file("./benchmarks/problem/output/charges.pvd");
  Fixed_Charged_Expression fc_expr;
  charges.interpolate(fc_expr);
  charges_file << charges;

  dolfin::Function diffusivity(pnp_problem.diffusivity_space);
  dolfin::File diffusivity_file("./benchmarks/problem/output/diffusivity.pvd");
  Diffusivity_Expression diff_expr;
  diffusivity.interpolate(diff_expr);
  // diffusivity_file << diffusivity[0];
  diffusivity_file << diffusivity[1];
  diffusivity_file << diffusivity[2];
  diffusivity_file << diffusivity[3];

  dolfin::Function valency(pnp_problem.valency_space);
  dolfin::File valency_file("./benchmarks/problem/output/valency.pvd");
  Valency_Expression valency_expr;
  valency.interpolate(valency_expr);
  // valency_file << valency[0];
  valency_file << valency[1];
  valency_file << valency[2];
  valency_file << valency[3];

  std::map<std::string, dolfin::Function> pnp_coefficient_fns = {
    {"permittivity", permittivity},
    {"diffusivity", diffusivity},
    {"valency", valency}
  };
  std::map<std::string, dolfin::Function> pnp_source_fns = {
    {"fixed_charge", charges}
  };

  printf("\tresetting PNP coefficients to expressions\n\n");
  pnp_problem.set_coefficients(
    pnp_coefficient_fns,
    pnp_source_fns
  );
  printf("\tdone\n\n");




  //-------------------------
  // Print various solutions
  //-------------------------
  dolfin::File solution_file0("./benchmarks/problem/output/1solution.pvd");
  dolfin::File solution_file1("./benchmarks/problem/output/2solution.pvd");
  dolfin::File solution_file2("./benchmarks/problem/output/3solution.pvd");
  dolfin::File solution_file3("./benchmarks/problem/output/4solution.pvd");

  // default solution
  dolfin::Function solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  // prescribed constant solution
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


  // computed solution for prescribed Dirichlet
  pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function dolfin_solution(pnp_problem.dolfin_solve());
  solution_file0 << dolfin_solution[0];
  solution_file1 << dolfin_solution[1];
  solution_file2 << dolfin_solution[2];
  solution_file3 << dolfin_solution[3];


  // FASP computed solution
  pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function fasp_solution(pnp_problem.fasp_solve());
  solution_file0 << fasp_solution[0];
  solution_file1 << fasp_solution[1];
  solution_file2 << fasp_solution[2];
  solution_file3 << fasp_solution[3];

  printf("Done\n\n"); fflush(stdout);

  return 0;
}
