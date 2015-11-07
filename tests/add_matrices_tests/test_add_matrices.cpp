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
#include "VecSpace.h"
#include "VecSpace2.h"
#include "Space.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "newton.h"
#include "newton_functs.h"
#include "funcspace_to_vecspace.h"
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
double pi=DOLFIN_PI;


class BigExp : public Expression
{
public:
  BigExp() : Expression(3) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
  }
};

// Exact solution
class Solution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
  }
};
class Solution3 : public Expression
{
public:
  Solution3() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
    values[1] = sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
  }
};
class Solution2 : public Expression
{
public:
  Solution2() : Expression(3) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
    values[1] = sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
    values[2] = sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
  }
};
// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] =  3*4*pow(pi,2)*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[2]);
  }
};
// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && ( x[0] < -1.0+DOLFIN_EPS || x[0] > 1.0 -DOLFIN_EPS ||
    x[1] < -1.0+DOLFIN_EPS || x[1] > 1.0 -DOLFIN_EPS || x[2] < -1.0+DOLFIN_EPS || x[2] > 1.0 -DOLFIN_EPS );
  }
};

int main()
{

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n Test of dofs                                                  "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  // read domain parameters
  domain_param domain_par;
  char domain_param_filename[] = "./tests/add_matrices_tests/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);

  // Function space
  VecSpace::FunctionSpace V(mesh);
  Space::FunctionSpace V0(mesh);
  VecSpace2::FunctionSpace V2(mesh);

  // Solve
  FacetFunction<std::size_t> markers(mesh, 1);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V0, u0, boundary);
  Space::BilinearForm a(V0, V0);
  Space::LinearForm L(V0);
  Source f;
  L.f = f;
  // Assembl Matrix and RHS
  EigenMatrix A;
  assemble(A,a); bc.apply(A);
  EigenVector b;
  assemble(b,L); bc.apply(b);
  EigenVector Solu_vec;
  solve(A, Solu_vec, b, "cg");
  Solution ExactSolu;
  dolfin::Function solu_ex(V0);
  dolfin::Function solu(V0);
  solu_ex.interpolate(ExactSolu);
  dolfin::File file1("./tests/add_matrices_tests/output/ExactSolu_V0.pvd");
  file1 << solu_ex;
  *(solu.vector())=Solu_vec;
  dolfin::File file2("./tests/add_matrices_tests/output/Solu_V0.pvd");
  file2 << solu;
  *(solu.vector())-=*(solu_ex.vector());
  dolfin::File file2b("./tests/add_matrices_tests/output/Error_V0.pvd");
  file2b << solu;

  // Define boundary condition
  Constant u02(0.0,0.0,0.0);
  DirichletBC bc2(V, u02, boundary);
  VecSpace::BilinearForm a2(V, V);
  VecSpace::LinearForm L2(V);
  Source f2;
  Constant f1(1.0);
  Constant f3(1.0);
  L2.f1 = f2;
  L2.f2 = f2;
  L2.f3 = f2;
  Constant Ep1(1.0);
  Constant Ep2(1.0);
  Constant Ep3(1.0);
  a2.Ep1 = Ep1;
  a2.Ep2 = Ep2;
  a2.Ep3 = Ep1;
  L2.Ep1 = Ep3;
  L2.Ep2 = Ep2;
  L2.Ep3 = Ep3;
  // Assembl Matrix and RHS
  EigenMatrix A2;
  assemble(A2,a2); bc2.apply(A2);
  EigenVector b2;
  assemble(b2,L2); bc2.apply(b2);
  EigenVector Solu_vec2;

  printf("\tA2 size = %ld x %ld\n",A2.size(0),A2.size(1));
  printf("\tA size = %ld x %ld\n",A.size(0),A.size(1));
  printf("\tV dim = %ld \n",V.dim());
  printf("\tV0 dim = %ld \n",V0.dim());
  printf("\tA nnz = %ld \n",A2.nnz());
  printf("\tA0 nnz = %ld \n",A.nnz());

  add_matrix(3,0, &V, &V0, &A2, &A);
  add_matrix(3,2, &V, &V0, &A2, &A);

  solve(A2, Solu_vec2, b2, "bicgstab");
  Solution2 ExactSolu2;
  dolfin::Function solu_ex2(V);
  dolfin::Function solu2(V);
  solu_ex2.interpolate(ExactSolu2);
  dolfin::File file3a("./tests/add_matrices_tests/output/ExactSolu1_V.pvd");
  file3a << solu_ex2[0];
  dolfin::File file3b("./tests/add_matrices_tests/output/ExactSolu0_V.pvd");
  file3b << solu_ex2[1];
  dolfin::File file3c("./tests/add_matrices_tests/output/ExactSolu2_V.pvd");
  file3c << solu_ex2[2];
  *(solu2.vector())=Solu_vec2;
  dolfin::File file4("./tests/add_matrices_tests/output/Solu0_V.pvd");
  file4 << solu2[0];
  dolfin::File file5("./tests/add_matrices_tests/output/Solu1_V.pvd");
  file5 << solu2[1];
  dolfin::File file6("./tests/add_matrices_tests/output/Solu2_V.pvd");
  file6 << solu2[2];
  *(solu2.vector())-=*(solu_ex2.vector());
  dolfin::File file4b("./tests/add_matrices_tests/output/Error0_V.pvd");
  file4b << solu2[0];
  dolfin::File file5b("./tests/add_matrices_tests/output/Error1_V.pvd");
  file5b << solu2[1];
  dolfin::File file6b("./tests/add_matrices_tests/output/Error2_V.pvd");
  file6b << solu2[2];

  Function solu_0(solu2[0]);
  Function solu_1(solu2[1]);
  Function solu_2(solu2[2]);

  printf("Errors:\n");
  printf("\tSoluon V0 = %f\n",solu.vector()->norm("l2"));
  printf("\tSolu[0] = %f\n",solu_0.vector()->norm("l2"));
  printf("\tSolu[1] = %f\n",solu_1.vector()->norm("l2"));
  printf("\tSolu[2] = %f\n",solu_2.vector()->norm("l2"));


  // Define boundary condition
  Constant u03(0.0,0.0);
  DirichletBC bc3(V2, u03, boundary);
  VecSpace2::BilinearForm a3(V2, V2);
  VecSpace2::LinearForm L3(V2);
  L3.f1 = f2;
  L3.f2 = f2;
  Constant Ep1_3(1.0);
  Constant Ep2_3(10.0);
  a3.Ep1 = Ep1_3;
  a3.Ep2 = Ep2_3;
  L3.Ep1 = Ep1_3;
  L3.Ep2 = Ep1_3;
  // Assembl Matrix and RHS
  EigenMatrix A3;
  assemble(A3,a3); bc3.apply(A3);
  EigenVector b3;
  assemble(b3,L3); bc3.apply(b3);
  EigenVector Solu_vec3;

  add_matrix(2,1, &V2, &V0, &A3, &A);

  solve(A3, Solu_vec3, b3, "bicgstab");
  Solution3 ExactSolu3;
  dolfin::Function solu_ex3(V2);
  dolfin::Function solu3(V2);
  solu_ex3.interpolate(ExactSolu3);
  *(solu3.vector())=Solu_vec3;
  *(solu3.vector())-=*(solu_ex3.vector());

  Function solu3_0(solu3[0]);
  Function solu3_1(solu3[1]);

  printf("Errors:\n");
  printf("\tSoluon V0 = %f\n",solu.vector()->norm("l2"));
  printf("\tSolu3[0] = %f\n",solu3_0.vector()->norm("l2"));
  printf("\tSolu3[1] = %f\n",solu3_1.vector()->norm("l2"));



  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
