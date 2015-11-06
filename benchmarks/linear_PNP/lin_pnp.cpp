/*! \file test_eafe.cpp
 *
 *  \brief Main to test EAFE functionality on linear convection reaction problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "linear_pnp.h"
#include "newton.h"
#include "newton_functs.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
  INT fasp_solver_dcsr_krylov (dCSRmat *A,
   dvector *b,
   dvector *x,
   itsolver_param *itparam
  );
#define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

double lower_cation_val = 0.1;  // 1 / m^3
double upper_cation_val = 1.0;  // 1 / m^3
double lower_anion_val = 1.0;  // 1 / m^3
double upper_anion_val = 0.1;  // 1 / m^3
double lower_potential_val = -1.0;  // V
double upper_potential_val = 1.0;  // V

class analyticCationExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_cation_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_cation_val * (x[0] + 5.0) / 10.0;
    values[0] += 2.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
  }
};

class analyticAnionExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_anion_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_anion_val * (x[0] + 5.0) / 10.0;
    values[0] += 2.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
  }
};

class analyticPotentialExpression : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]  = lower_potential_val * (5.0 - x[0]) / 10.0;
    values[0] += upper_potential_val * (x[0] + 5.0) / 10.0;
    values[0] += 0.0  * (5.0 - x[0]) * (x[0] + 5.0) / 100.0;
  }
};


int main()
{

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n Solving the linearized Poisson-Nernst-Planck system           "); fflush(stdout);
  printf("\n of a single cation and anion                                  "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  //*************************************************************
  //  Initialization
  //*************************************************************
  printf("Initialize the problem\n"); fflush(stdout);
  // read domain parameters
  printf("\tdomain...\n"); fflush(stdout);
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/linear_PNP/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  // print_domain_param(&domain_par);

  // build mesh
  printf("\tmesh...\n"); fflush(stdout);
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);

  // read coefficients and boundary values
  printf("\tcoefficients...\n"); fflush(stdout);
  coeff_param coeff_par;
  char coeff_param_filename[] = "./benchmarks/linear_PNP/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  // print_coeff_param(&coeff_par);

  // open files for outputting solutions
  File cationFile("./benchmarks/linear_PNP/output/cation.pvd");
  File anionFile("./benchmarks/linear_PNP/output/anion.pvd");
  File potentialFile("./benchmarks/linear_PNP/output/potential.pvd");

  // Initialize variational forms
  printf("\tvariational forms...\n"); fflush(stdout);
  linear_pnp::FunctionSpace V(mesh);
  linear_pnp::BilinearForm a_pnp(V,V);
  linear_pnp::LinearForm L_pnp(V);
  Constant eps(coeff_par.relative_permittivity);
  Constant Dp(coeff_par.cation_diffusivity);
  Constant Dn(coeff_par.anion_diffusivity);
  Constant qn(coeff_par.cation_mobility);
  Constant qp(coeff_par.anion_mobility);
  Constant zero(0.0);
  a_pnp.eps = eps; L_pnp.eps = eps;
  a_pnp.Dp = Dp; L_pnp.Dp = Dp;
  a_pnp.Dn = Dn; L_pnp.Dn = Dn;
  a_pnp.qp = qp; L_pnp.qp = qp;
  a_pnp.qn = qn; L_pnp.qn = qn;

  // analytic solution
  Function analyticSolutionFunction(V);
  Function analyticCation(analyticSolutionFunction[0]);
  Function analyticAnion(analyticSolutionFunction[1]);
  Function analyticPotential(analyticSolutionFunction[2]);
  analyticCationExpression cationExpression;
  analyticAnionExpression anionExpression;
  analyticPotentialExpression potentialExpression;
  analyticCation.interpolate(cationExpression);
  analyticAnion.interpolate(anionExpression);
  analyticPotential.interpolate(potentialExpression);
  L_pnp.cation = analyticCation;
  L_pnp.anion = analyticAnion;
  L_pnp.potential = analyticPotential;

  // Set Dirichlet boundaries
  printf("\tboundary conditions...\n"); fflush(stdout);
  unsigned int dirichlet_coord = 0;
  Constant zero_vec(0.0, 0.0, 0.0);
  SymmBoundaries boundary(dirichlet_coord, -domain_par.length_x/2.0, domain_par.length_x/2.0);
  dolfin::DirichletBC bc(V, zero_vec, boundary);

  // Initialize analytic expressions
  printf("\tanalytic expressions...\n"); fflush(stdout);
  LogCharge Cation(lower_cation_val, upper_cation_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);
  LogCharge Anion(lower_anion_val, upper_anion_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);
  Voltage Volt(lower_potential_val, upper_potential_val, -domain_par.length_x/2.0, domain_par.length_x/2.0, dirichlet_coord);

  // Interpolate analytic expressions
  Function solutionFunction(V);
  Function cationSolution(solutionFunction[0]);
  cationSolution.interpolate(Cation);
  Function anionSolution(solutionFunction[1]);
  anionSolution.interpolate(Anion);
  ivector cation_dofs;
  ivector anion_dofs;
  ivector potential_dofs;
  get_dofs(&solutionFunction, &cation_dofs, 0);
  get_dofs(&solutionFunction, &anion_dofs, 1);
  get_dofs(&solutionFunction, &potential_dofs, 2);

  // Solve for consistent voltage : not yet implemented
  Function potentialSolution(solutionFunction[2]);
  potentialSolution.interpolate(Volt);

  // print to file
  cationFile << cationSolution;
  anionFile << anionSolution;
  potentialFile << potentialSolution;

  // initialize linear system
  printf("\tlinear algebraic objects...\n"); fflush(stdout);
  EigenMatrix A_pnp;
  EigenVector b_pnp;
  dCSRmat A_fasp;
  dvector b_fasp, solu_fasp;

  // Setup FASP solver
  printf("\tsetup FASP solver...\n"); fflush(stdout);
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./benchmarks/linear_PNP/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  //ivector cation_fasp_dofs; map_dofs_for_fasp(&cation_dofs, &cation_fasp_dofs);
  //ivector anion_fasp_dofs; map_dofs_for_fasp(&anion_dofs, &anion_fasp_dofs);
  //ivector potential_fasp_dofs; map_dofs_for_fasp(&potential_dofs, &potential_fasp_dofs);
  INT status = FASP_SUCCESS;


  //*************************************************************
  //  Initialize Newton solver
  //*************************************************************
  // Setup newton parameters and compute initial residual
  // printf("\tnewton solver...\n"); fflush(stdout);
  Function solutionUpdate(V);

  // compute initial residual and Jacobian
  printf("\tconstruct linear system...\n"); fflush(stdout);
  a_pnp.CatCat = cationSolution;
  a_pnp.AnAn = anionSolution;
  a_pnp.EsEs = potentialSolution;
  assemble(A_pnp, a_pnp);

  L_pnp.CatCat = cationSolution;
  L_pnp.AnAn = anionSolution;
  L_pnp.EsEs = potentialSolution;
  assemble(b_pnp, L_pnp);
  //compute_residual(b_pnp);

  printf("\tinitialized succesfully!\n\n"); fflush(stdout);



  //*************************************************************
  //  Solve : this will be inside Newton loop
  //*************************************************************
  printf("Solve the system\n"); fflush(stdout);
  // Convert to fasp
  printf("\tconvert to FASP and solve...\n"); fflush(stdout);
  EigenVector_to_dvector(&b_pnp,&b_fasp);
  EigenMatrix_to_dCSRmat(&A_pnp,&A_fasp);
  fasp_dvec_alloc(b_fasp.row, &solu_fasp);
  fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);
  status = fasp_solver_dcsr_krylov(&A_fasp, &b_fasp, &solu_fasp, &itpar);

  // map solu_fasp into solutionUpdate
  printf("\tconvert FASP solution to function...\n"); fflush(stdout);
  copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &cation_dofs, &cation_dofs);
  copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &anion_dofs, &anion_dofs);
  copy_dvector_to_vector_function(&solu_fasp, &solutionUpdate, &potential_dofs, &potential_dofs);

  // update solution and reset solutionUpdate
  printf("\tupdate solution...\n"); fflush(stdout);
  *(solutionFunction.vector()) += *(solutionUpdate.vector());

  // write computed solution to file
  printf("\tsolved successfully!\n"); fflush(stdout);
  cationFile << cationSolution;
  anionFile << anionSolution;
  potentialFile << potentialSolution;

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
