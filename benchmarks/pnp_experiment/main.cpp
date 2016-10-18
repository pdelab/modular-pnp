/// Main file for solving the linearized PNP problem
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
      values[1] = x[0] > 0.0 ? 1.0 : 0.1;
      values[2] = x[0] > 0.0 ? 2.0 : 0.1;
      values[3] = x[0] > 0.0 ? 4.0 : 0.1;
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

std::size_t num_spheres = 20;
double xc[86] = { -2.36111111111,2.5,3.47222222222,-0.138888888889,2.77777777778,1.94444444444,3.05555555556,0.277777777778,3.19444444444,-0.277777777778,0.416666666667,
  -1.80555555556,3.33333333333,-0.555555555556,3.05555555556,2.91666666667,-1.52777777778,-3.33333333333,-1.25,-2.5,-1.94444444444,
  -0.277777777778,0.138888888889,2.36111111111,2.22222222222,-1.25,2.91666666667,-3.33333333333,-0.138888888889,-0.972222222222,-1.94444444444,
  1.66666666667,1.11111111111,-1.11111111111,1.11111111111,0.416666666667,0.972222222222,-2.77777777778,3.47222222222,-3.05555555556,-1.80555555556,
  -1.94444444444,-2.91666666667,2.91666666667,-0.416666666667,3.05555555556,2.91666666667,-0.416666666667,0.833333333333,3.19444444444,1.66666666667,
  0.0,-0.833333333333,-0.833333333333,-1.66666666667,-0.972222222222,-0.555555555556,-3.33333333333,2.36111111111,0.138888888889,-3.05555555556,
  -2.08333333333,-2.91666666667,-3.05555555556,3.05555555556,-3.33333333333,-2.63888888889,-2.08333333333,2.08333333333,-0.277777777778,1.25,
  -0.694444444444,2.63888888889,-3.19444444444,2.08333333333,2.5,-2.22222222222,2.36111111111,1.25,2.63888888889,-0.972222222222,
  -2.77777777778,-1.80555555556,-0.416666666667,-2.22222222222,-1.52777777778};

double yc[86] = { -2.22222222222,1.52777777778,-2.08333333333,-0.416666666667,-2.77777777778,-0.277777777778,2.77777777778,2.77777777778,2.08333333333,-2.22222222222,0.0,
  2.5,-3.33333333333,0.694444444444,-2.22222222222,-2.08333333333,-2.08333333333,0.833333333333,1.25,3.19444444444,3.05555555556,
  2.63888888889,-1.66666666667,-0.416666666667,-3.19444444444,0.555555555556,3.33333333333,3.05555555556,-3.05555555556,0.555555555556,-1.25,
  -1.25,-2.08333333333,-3.19444444444,-2.08333333333,1.94444444444,-0.277777777778,0.833333333333,0.277777777778,-3.19444444444,-1.94444444444,
  -3.47222222222,3.05555555556,-2.63888888889,1.25,-2.08333333333,2.77777777778,0.555555555556,2.08333333333,-1.66666666667,2.36111111111,
  2.36111111111,0.277777777778,1.25,2.77777777778,2.5,0.972222222222,3.05555555556,2.22222222222,2.22222222222,1.38888888889,
  0.694444444444,-3.19444444444,-2.36111111111,1.38888888889,-2.08333333333,2.77777777778,3.33333333333,-1.11111111111,-0.972222222222,0.277777777778,
  -0.694444444444,-3.05555555556,1.66666666667,-2.22222222222,-1.80555555556,-1.52777777778,-3.05555555556,-0.416666666667,-3.47222222222,0.277777777778,
  0.416666666667,1.38888888889,-0.416666666667,-3.33333333333,-3.05555555556};

double zc[86] = { 0.972222222222,-2.63888888889,-2.08333333333,3.33333333333,-2.63888888889,-2.77777777778,0.138888888889,3.33333333333,1.94444444444,1.52777777778,-2.91666666667,
  -0.694444444444,1.25,-0.972222222222,2.91666666667,2.08333333333,2.77777777778,2.36111111111,-3.19444444444,1.80555555556,1.11111111111,
  1.94444444444,2.63888888889,-0.555555555556,-1.66666666667,-0.694444444444,-3.33333333333,-0.833333333333,0.277777777778,1.38888888889,-0.138888888889,
  -2.77777777778,-3.33333333333,-2.5,-2.5,2.22222222222,3.33333333333,-3.05555555556,1.80555555556,0.972222222222,-2.36111111111,
  -2.36111111111,-0.972222222222,0.0,0.0,1.94444444444,0.972222222222,1.11111111111,-3.05555555556,-3.19444444444,1.11111111111,
  0.277777777778,0.833333333333,3.05555555556,0.833333333333,2.5,0.277777777778,2.77777777778,0.833333333333,1.52777777778,0.694444444444,
  3.05555555556,-0.833333333333,1.38888888889,0.138888888889,-3.05555555556,1.66666666667,-2.08333333333,-3.33333333333,-0.972222222222,-2.63888888889,
  -2.08333333333,-1.11111111111,3.05555555556,-0.416666666667,-2.22222222222,2.5,-2.36111111111,-2.63888888889,1.25,-1.25,
  -2.5,-1.11111111111,2.22222222222,1.80555555556,3.33333333333};

double rc[86] = { 1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,
  1.25,1.25,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,
  0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,
  0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,
  0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.972222222222,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,
  0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,
  0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,
  0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.833333333333,
  0.833333333333,0.833333333333,0.833333333333,0.833333333333,0.694444444444};

class BC_Interpolant : public dolfin::Expression {
public:
  BC_Interpolant() : dolfin::Expression(4) {}
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
    values[0] = x[0] < -5.0 + 1.0e-8 ? -1.0 : (x[0] > 5.0 - 1.0e-8 ? 1.0 : 0.0);
    values[1] = x[0] < -5.0 + 1.0e-8 ? std::log(1.0) : (x[0] > 5.0 - 1.0e-8 ? std::log(2.0) : std::log(1.5));
    values[2] = x[0] < -5.0 + 1.0e-8 ? std::log(1.5) : (x[0] > 5.0 - 1.0e-8 ? std::log(1.0) : std::log(1.0));
    values[3] = x[0] < -5.0 + 1.0e-8 ? std::log(0.5) : (x[0] > 5.0 - 1.0e-8 ? std::log(2.0) : std::log(1.5));
  }
};

class SpheresSubDomain : public dolfin::SubDomain {
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const {
    for (std::size_t i = 0; i < num_spheres; i++) {
      if (on_boundary && (
        std::pow(x[0] - xc[i], 2) + std::pow(x[1] - yc[i], 2) + std::pow(x[2] - zc[i], 2) < std::pow(rc[i], 2) + 0.1
      )) { return true; }
    }

    return false;
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
  char domain_param_filename[] = "./benchmarks/pnp_experiment/domain.dat";
  printf("\tdomain... %s\n", domain_param_filename);
  domain_param domain;
  domain_param_input(domain_param_filename, &domain);
  std::shared_ptr<dolfin::Mesh> mesh;
  mesh.reset(new dolfin::Mesh("./benchmarks/pnp_experiment/mesh.xml"));
  // *mesh = domain_build(domain);
  // print_domain_param(&domain);


  char fasp_params[] = "./benchmarks/pnp_experiment/bsr.dat";
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
    dolfin::File permittivity_file("./benchmarks/pnp_experiment/output/permittivity.pvd");
    dolfin::File charges_file("./benchmarks/pnp_experiment/output/charges.pvd");
    dolfin::File diffusivity_file("./benchmarks/pnp_experiment/output/diffusivity.pvd");
    dolfin::File valency_file("./benchmarks/pnp_experiment/output/valency.pvd");
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
  dolfin::File solution_file0("./benchmarks/pnp_experiment/output/1solution.pvd");
  dolfin::File solution_file1("./benchmarks/pnp_experiment/output/2solution.pvd");
  dolfin::File solution_file2("./benchmarks/pnp_experiment/output/3solution.pvd");
  dolfin::File solution_file3("./benchmarks/pnp_experiment/output/4solution.pvd");

  // initial guess for prescibed Dirichlet
  printf("Record interpolant for given Dirichlet BCs (initial guess for solution)\n");
  std::vector<std::size_t> components = {0, 0, 0, 0};
  std::vector<std::vector<double>> bcs;
  bcs.push_back({0.0,  1.0});
  bcs.push_back({std::log(1.0), std::log(2.0)});
  bcs.push_back({std::log(1.5), std::log(1.0)});
  bcs.push_back({std::log(0.5), std::log(2.0)});
  pnp_problem.set_DirichletBC(components, bcs);
  dolfin::Function solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];
  printf("\n");

  // add Dirichlet boundary conditions by subdomain
  auto sphere_boundary_ptr = std::make_shared<SpheresSubDomain>();
  std::vector<std::size_t> bc_fn_component = {0, 1, 2, 3};
  std::vector<std::shared_ptr<dolfin::SubDomain>> bc_vector (4, sphere_boundary_ptr);
  pnp_problem.add_DirichletBC(bc_fn_component, bc_vector);

  BC_Interpolant bc_interpolant_expr;
  dolfin::Function bc_interpolant_fn(pnp_problem.get_solution().function_space());
  bc_interpolant_fn.interpolate(bc_interpolant_expr);
  pnp_problem.set_solution(bc_interpolant_fn);
  solutionFn = pnp_problem.get_solution();
  solution_file0 << solutionFn[0];
  solution_file1 << solutionFn[1];
  solution_file2 << solutionFn[2];
  solution_file3 << solutionFn[3];

  //------------------------
  // Start nonlinear solver
  //------------------------
  printf("Initializing nonlinear solver\n");

  // set nonlinear solver parameters
  const std::size_t max_newton = 5;
  const double max_residual_tol = 1.0e-15;
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
    solution_file3 << solutionFn[3];
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
