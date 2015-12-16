/*! \file test_matrices.cpp
 *
 *  \brief Main to test add_matrix and replace_matrix from funcSpace_b_to_vecSpace_b.cpp
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "Space_b.h"
#include "VecSpace2_b.h"
#include "VecSpace3_b.h"
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
class PhiExp : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
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
class Source: public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 3.0*4.0*pow(pi,2)*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2]);
  }
};
// Source term (right-hand side)
class Source_b : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -4.0*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*(cos(2*pi*x[0])-3*pi*x[0]*sin(2*pi*x[0]));
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
    std::cout << "#### The test of replace_matrix2 with DEBUG=TRUE             #### \n";
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

  // Function Space_b/Vector Space_b
  Space_b::FunctionSpace V1(mesh);
  VecSpace2_b::FunctionSpace V2(mesh);
  VecSpace3_b::FunctionSpace V3(mesh);

  // Constants
  Constant Ep1(1.0);
  Constant Ep10(100.0);
  Source f;
  Source_b f1;
  PhiExp phi;
  DirichletBoundary boundary;

  // Assembling the system on the function Space_b V1
  Constant u1(0.0);
  DirichletBC bc(V1, u1, boundary);
  Space_b::BilinearForm a1(V1, V1);
  Space_b::LinearForm L1(V1);
  L1.f = f1;
  a1.phi = phi;
  EigenMatrix A1;
  assemble(A1,a1); bc.apply(A1);
  EigenVector b1;
  assemble(b1,L1); bc.apply(b1);

  // Assembling the system on the vector Space_b V2 (dim=2)
  Constant u2(0.0,0.0);
  DirichletBC bc2(V2, u2, boundary);
  VecSpace2_b::BilinearForm a2(V2, V2);
  VecSpace2_b::LinearForm L2(V2);
  L2.f1 = f1;
  L2.f2 = f;
  a2.Ep1 = Ep10;
  a2.Ep2 = Ep1;
  a2.phi = phi;
  L2.Ep1 = Ep1;
  L2.Ep2 = Ep1;
  // Assembl Matrix and RHS
  EigenMatrix A2;
  assemble(A2,a2); bc2.apply(A2);
  EigenVector b2;
  assemble(b2,L2); bc2.apply(b2);

  // Assembling the system on the vector Space_b V3 (dim=3)
  Constant u3(0.0,0.0,0.0);
  DirichletBC bc3(V3, u3, boundary);
  VecSpace3_b::BilinearForm a3(V3, V3);
  VecSpace3_b::LinearForm L3(V3);
  L3.f1 = f1;
  L3.f2 = f;
  L3.f3 = f;
  a3.phi = phi;
  a3.Ep1 = Ep10;
  a3.Ep2 = Ep1;
  a3.Ep3 = Ep1;
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
  replace_matrix(2,0, &V2, &V1, &A2, &A1);


  // Solving
  EigenVector Solu_vec1;
  EigenVector Solu_vec2;
  EigenVector Solu_vec3;

  if (DEBUG) printf("setup FASP solver...\n");
  input_param inpar;
  itsolver_param itpar;
  AMG_param amgpar;
  ILU_param ilupar;
  char inputfile[] = "./tests/matrices_tests/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  if (DEBUG) printf("convert to FASP and solve...\n");
  dCSRmat A1_fasp, A2_fasp, A3_fasp;
  dvector b1_fasp, b2_fasp, b3_fasp;
  EigenVector_to_dvector(&b1,&b1_fasp);
  EigenVector_to_dvector(&b2,&b2_fasp);
  EigenVector_to_dvector(&b3,&b3_fasp);
  EigenMatrix_to_dCSRmat(&A1,&A1_fasp);
  EigenMatrix_to_dCSRmat(&A2,&A2_fasp);
  EigenMatrix_to_dCSRmat(&A3,&A3_fasp);
  dvector solu1_fasp, solu2_fasp, solu3_fasp;
  fasp_dvec_alloc(b1_fasp.row, &solu1_fasp);
  fasp_dvec_alloc(b2_fasp.row, &solu2_fasp);
  fasp_dvec_alloc(b3_fasp.row, &solu3_fasp);
  fasp_dvec_set(b1_fasp.row, &solu1_fasp, 0.0);
  fasp_dvec_set(b2_fasp.row, &solu2_fasp, 0.0);
  fasp_dvec_set(b3_fasp.row, &solu3_fasp, 0.0);
  status = fasp_solver_dcsr_krylov(&A1_fasp, &b1_fasp, &solu1_fasp, &itpar);
  status = fasp_solver_dcsr_krylov(&A2_fasp, &b2_fasp, &solu2_fasp, &itpar);
  status = fasp_solver_dcsr_krylov(&A3_fasp, &b3_fasp, &solu3_fasp, &itpar);

  Function solu_ex1(V1); Function solu1(V1);
  Function solu_ex2(V2); Function solu2(V2);
  Function solu_ex3(V3); Function solu3(V3);

  Solution1 S1;
  Solution2 S2;
  Solution3 S3;

  solu_ex1.interpolate(S1);
  solu_ex2.interpolate(S2);
  solu_ex3.interpolate(S3);
  copy_dvector_to_Function(&solu1_fasp,&solu1);
  copy_dvector_to_Function(&solu2_fasp,&solu2);
  copy_dvector_to_Function(&solu3_fasp,&solu3);

  double error_norm1 = 0.0;
  *(solu_ex1.vector())-=*(solu1.vector());
  L2Error::Form_M L2error1(mesh,solu_ex1);
  error_norm1 = assemble(L2error1);
  if (DEBUG) printf("L2 Error on V1 is:\t%e\n", error_norm1);
  double error_norm2_1 = 0.0;
  double error_norm2_2 = 0.0;
  *(solu_ex2.vector())-=*(solu2.vector());
  L2Error::Form_M L2error2_1(mesh,solu_ex2[0]);
  L2Error::Form_M L2error2_2(mesh,solu_ex2[1]);
  error_norm2_1 = assemble(L2error2_1);
  error_norm2_2 = assemble(L2error2_2);
  if (DEBUG) printf("L2 Error on of V2 are:\t%e\t%e\n", error_norm2_1, error_norm2_2);
  double error_norm3_1 = 0.0;
  double error_norm3_2 = 0.0;
  double error_norm3_3 = 0.0;
  *(solu_ex3.vector())-=*(solu3.vector());
  L2Error::Form_M L2error3_1(mesh,solu_ex3[0]);
  L2Error::Form_M L2error3_2(mesh,solu_ex3[1]);
  L2Error::Form_M L2error3_3(mesh,solu_ex3[2]);
  error_norm3_1 = assemble(L2error3_1);
  error_norm3_2 = assemble(L2error3_2);
  error_norm3_3 = assemble(L2error3_3);
  if (DEBUG) printf("L2 Error on of V3 are:\t%e\t%e\t%e\n\n", error_norm3_1, error_norm3_2, error_norm3_3);

  double EPS = 1E-5;
  if ( (std::fabs(error_norm2_1-error_norm1) < EPS) && (std::fabs(error_norm3_1-error_norm1) < EPS)  )
 {
   std::cout << "Success... replace_matrix (test 2) is working\n";
 }
 else {
   printf("***\tERROR IN REPLACE_MATRIX TEST 2\n");
   printf("***\n***\n***\n");
   printf("***\tSOLVER TEST:\n");
   printf("***\tThe computed solution is wrong\n");
   printf("***\n***\n***\n");
   printf("***\tERROR IN REPLACE_MATRIX TEST 2\n");
   fflush(stdout);
 }
  if (DEBUG) {
    std::cout << "################################################################# \n";
    std::cout << "#### End of the test of replace_matrix2 with DEBUG=TRUE      #### \n";
    std::cout << "################################################################# \n";
  }

  return 0;
}
