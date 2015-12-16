/*! \file test_matrices.cpp
 *
 *  \brief Main to test add_matrix and replace_matrix from funcspace_to_vecspace.cpp
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "Space.h"
#include "VecSpace2.h"
#include "VecSpace3.h"
#include "L2Error.h"
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
bool DEBUG = false;


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
class Solution1 : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
  }
};
class Solution2 : public Expression
{
public:
  Solution2() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
    values[1] = sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
  }
};
class Solution3 : public Expression
{
public:
  Solution3() : Expression(3) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
    values[1] = sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
    values[2] = sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
  }
};
// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] =  3.0*4.0*pow(pi,2)*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
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

int main(int argc, char** argv)
{
  if (argc >1)
  {
    if (std::string(argv[1])=="DEBUG") DEBUG = true;
  }
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### The test of replace_matrix with DEBUG=TRUE              #### \n";
    std::cout << "################################################################# \n\n";
  }

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  // build mesh
  Point p0( -1.0, -1.0, -1.0);
  Point p1( 1.0, 1.0, 1.0);
  double mesh_size=20;
  BoxMesh mesh(p0, p1, mesh_size, mesh_size, mesh_size);

  // Function space/Vector space
  Space::FunctionSpace V1(mesh);
  VecSpace2::FunctionSpace V2(mesh);
  VecSpace3::FunctionSpace V3(mesh);

  // Constants
  Constant Ep1(1.0);
  Constant Ep10(10.0);
  Source f;
  DirichletBoundary boundary;

  // Assembling the system on the function space V1
  Constant u1(0.0);
  DirichletBC bc(V1, u1, boundary);
  Space::BilinearForm a1(V1, V1);
  Space::LinearForm L1(V1);
  L1.f = f;
  EigenMatrix A1;
  assemble(A1,a1); bc.apply(A1);
  EigenVector b1;
  assemble(b1,L1); bc.apply(b1);

  // Assembling the system on the vector space V2 (dim=2)
  Constant u2(0.0,0.0);
  DirichletBC bc2(V2, u2, boundary);
  VecSpace2::BilinearForm a2(V2, V2);
  VecSpace2::LinearForm L2(V2);
  L2.f1 = f;
  L2.f2 = f;
  a2.Ep1 = Ep10;
  a2.Ep2 = Ep10;
  L2.Ep1 = Ep1;
  L2.Ep2 = Ep1;
  // Assembl Matrix and RHS
  EigenMatrix A2;
  assemble(A2,a2); bc2.apply(A2);
  EigenVector b2;
  assemble(b2,L2); bc2.apply(b2);

  // Assembling the system on the vector space V3 (dim=3)
  Constant u3(0.0,0.0,0.0);
  DirichletBC bc3(V3, u3, boundary);
  VecSpace3::BilinearForm a3(V3, V3);
  VecSpace3::LinearForm L3(V3);
  L3.f1 = f;
  L3.f2 = f;
  L3.f3 = f;
  a3.Ep1 = Ep10;
  a3.Ep2 = Ep1;
  a3.Ep3 = Ep10;
  L3.Ep1 = Ep1;
  L3.Ep2 = Ep1;
  L3.Ep3 = Ep1;
  EigenMatrix A3;
  assemble(A3,a3); bc3.apply(A3);
  EigenVector b3;
  assemble(b3,L3); bc3.apply(b3);

  if (DEBUG){
    printf("Function Space V1:\n");
    printf("\tV1.dim() = %ld\n",V1.dim());
    printf("\tA1 size = %ld x %ld\n",A1.size(0),A1.size(1));
    printf("Vector Space V2:\n");
    printf("\tV2.dim() = %ld\n",V2.dim());
    printf("\tA2 size = %ld x %ld\n",A2.size(0),A2.size(1));
    printf("Vector Space V3:\n");
    printf("\tV3.dim() = %ld\n",V3.dim());
    printf("\tA3 size = %ld x %ld\n",A3.size(0),A3.size(1));
  }

  replace_matrix(3,0, &V3, &V1, &A3, &A1);
  replace_matrix(3,2, &V3, &V1, &A3, &A1);
  replace_matrix(2,0, &V2, &V1, &A2, &A1);
  replace_matrix(2,1, &V2, &V1, &A2, &A1);

  // Solving
  EigenVector Solu_vec1;
  EigenVector Solu_vec2;
  EigenVector Solu_vec3;
  solve(A1, Solu_vec1, b1, "bicgstab");
  solve(A2, Solu_vec2, b2, "bicgstab");
  solve(A3, Solu_vec3, b3, "bicgstab");

  Function solu_ex1(V1); Function solu1(V1);
  Function solu_ex2(V2); Function solu2(V2);
  Function solu_ex3(V3); Function solu3(V3);

  Solution1 S1;
  Solution2 S2;
  Solution3 S3;

  solu_ex1.interpolate(S1);
  solu_ex2.interpolate(S2);
  solu_ex3.interpolate(S3);
  *(solu1.vector())=Solu_vec1;
  *(solu2.vector())=Solu_vec2;
  *(solu3.vector())=Solu_vec3;

  double error_norm1 = 0.0;
  *(solu_ex1.vector())-=Solu_vec1;
  L2Error::Form_M L2error1(mesh,solu_ex1);
  error_norm1 = assemble(L2error1);
  if (DEBUG) printf("L2 Error on V1 is:\t%e\n", error_norm1);
  double error_norm2_1 = 0.0;
  double error_norm2_2 = 0.0;
  *(solu_ex2.vector())-=Solu_vec2;
  L2Error::Form_M L2error2_1(mesh,solu_ex2[0]);
  L2Error::Form_M L2error2_2(mesh,solu_ex2[1]);
  error_norm2_1 = assemble(L2error2_1);
  error_norm2_2 = assemble(L2error2_2);
  if (DEBUG) printf("L2 Error on of V2 are:\t%e\t%e\n", error_norm2_1, error_norm2_2);
  double error_norm3_1 = 0.0;
  double error_norm3_2 = 0.0;
  double error_norm3_3 = 0.0;
  *(solu_ex3.vector())-=Solu_vec3;
  L2Error::Form_M L2error3_1(mesh,solu_ex3[0]);
  L2Error::Form_M L2error3_2(mesh,solu_ex3[1]);
  L2Error::Form_M L2error3_3(mesh,solu_ex3[2]);
  error_norm3_1 = assemble(L2error3_1);
  error_norm3_2 = assemble(L2error3_2);
  error_norm3_3 = assemble(L2error3_3);
  if (DEBUG) printf("L2 Error on of V3 are:\t%e\t%e\t%e\n\n", error_norm3_1, error_norm3_2, error_norm3_3);

  double EPS = 1E-8;
  if ( (std::fabs(error_norm2_1-error_norm1) < EPS) && (std::fabs(error_norm2_2-error_norm1) < EPS) &&
       (std::fabs(error_norm3_1-error_norm1) < EPS) && (std::fabs(error_norm3_2-error_norm1) < EPS) && (std::fabs(error_norm3_3-error_norm1) < EPS)  )
 {
   std::cout << "Success... replace_matrix is working\n";
 }
 else {
   printf("***\tERROR IN REPLACE_MATRIX TEST\n");
   printf("***\n***\n***\n");
   printf("***\tSOLVER TEST:\n");
   printf("***\tThe computed solution is wrong\n");
   printf("***\n***\n***\n");
   printf("***\tERROR IN REPLACE_MATRIX TEST\n");
   fflush(stdout);
 }
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of the test of replace_matrix with DEBUG=TRUE       #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;
}
