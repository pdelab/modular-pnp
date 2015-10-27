/*! \file test_faspfenics.cpp
 *
 *  \brief Main to test EAFE functionality on linear convection reaction problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "fasp_to_fenics.h"
#include "Testcase.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
  INT fasp_solver_dcsr_krylov (dCSRmat *A,
   dvector *b,
   dvector *x,
   itsolver_param *itparam);
#define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;

class Advection : public Expression
{
public:
  Advection(double b_x, double b_y, double b_z): Expression(),BX(b_x),BY(b_y),BZ(b_z) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
   values[0] = BX*x[0] + BY*x[1] + BZ*x[2];
  }
private:
  double BX, BY, BZ;
};
class exp_phi: public Expression
{
public:
  exp_phi(double b_x, double b_y, double b_z): Expression(),BX(b_x),BY(b_y),BZ(b_z) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
   values[0] = std::exp(BX*x[0] + BY*x[1] + BZ*x[2]);
  }
private:
  double BX, BY, BZ;
};


class testRHS : public Expression
{
public:
  testRHS(double alpha, double eta, double gamma, double b_x, double b_y, double b_z): Expression(),A(alpha),E(eta),G(gamma),BX(b_x),BY(b_y),BZ(b_z) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    // u = x*y*(x-1)*(y-1)
    // u_x = y*(y-1)*(2*x-1), u_y = x*(x-1)*(2*y-1), u_z = 0
    // u_xx = 2*y*(y-1), u_yy = 2*x*(x-1), u_zz = 0

    // diffusion = -alpha*exp(eta)*(u_xx + u_yy + u_zz)
    double diffusion = -2.0*A*std::exp(E)*( x[1]*(x[1]-1.0) + x[0]*(x[0]-1.0) );

    // advection = -alpha*exp(eta)*(b_x*u_x + b_y*u_y + b_z*u_z)
    double advection = -A*std::exp(E)*( BX*x[1]*(x[1]-1.0)*(2.0*x[0]-1.0) + BY*x[0]*(x[0]-1.0)*(2.0*x[1]-1.0) );

    // reaction = gamma*exp(eta)*u
    double reaction = G*std::exp(E)*x[0]*x[1]*(x[0]-1.0)*(x[1]-1.0);

    values[0] = diffusion + advection + reaction;
  }
private:
  double A, E, G, BX, BY, BZ;
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
public:
  DirichletBoundary(): SubDomain() {}
  bool inside(const Array<double>& x, bool on_boundary) const
  {
  // x and y boundaries
      return on_boundary and (
        x[0] < DOLFIN_EPS
        or x[0] > 1.0 - DOLFIN_EPS
        or x[1] < DOLFIN_EPS
        or x[1] > 1.0 - DOLFIN_EPS
      );
    }
};

int main()
{
  // state problem
  printf("Solving the linear PDE with EAFE:\n");
  printf("\t-div( exp(phi)*( grad(u) + grad(phi)*u ) ) = 1\n");
  printf("And comparing it to :\n");
  printf("\t-div(grad(exp(phi)u)) = 1\n");
  printf("\n"); fflush(stdout);

  // read in coefficients
  char filenm[] = "./problems/test_eafe2/coefficients.dat";
  char buffer[100];
  int val;
  int mesh_size, FineMesh_size;
  double phi_x_double, phi_y_double, phi_z_double;
  FILE *fp = fopen(filenm,"r");
  if (fp==NULL) {
    printf("### ERROR: Could not open file %s...\n", filenm);
  }
  bool state = true;
  while ( state ) {
    int ibuff;
    double dbuff;
    char *fgetsPtr;

    val = fscanf(fp,"%s",buffer);
    if (val==EOF) break;
    if (val!=1){ state = false; break; }
    if (buffer[0]=='[' || buffer[0]=='%' || buffer[0]=='|') {
        fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        continue;
    }

    // match keyword and scan for value
    if (strcmp(buffer,"mesh_size")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%d",&ibuff);
      if (val!=1) { state = false; break; }
      mesh_size = ibuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    if (strcmp(buffer,"FineMesh_size")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%d",&ibuff);
      if (val!=1) { state = false; break; }
      FineMesh_size = ibuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"phi_x_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      phi_x_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"phi_y_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      phi_y_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"phi_z_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      phi_z_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else {
        state = true;
        printf(" Bad read-in: %s property unknown \n", buffer); fflush(stdout);
    }
  }
  fclose(fp);

  printf("Solving the problem with f = 1.\n");
  printf("Coefficients read in are:\n");
  printf("\tphi_x:   \t%e\n",phi_x_double);
  printf("\tphi_y:   \t%e\n",phi_y_double);
  printf("\tphi_z:   \t%e\n",phi_z_double);
  printf("\n"); fflush(stdout);


  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  printf("Create mesh %d x %d x %d \n",mesh_size,mesh_size,mesh_size); fflush(stdout);
  bool print_matrices = (mesh_size<5)? true : false;
  dolfin::Point p0( 0.0, 0.0, 0.0);
  dolfin::Point p1( 1.0, 1.0, 1.0);
  dolfin::BoxMesh mesh(p0, p1, mesh_size, mesh_size,mesh_size);
  printf("Create FineMesh %d x %d x %d \n",FineMesh_size,FineMesh_size,FineMesh_size); fflush(stdout);
  dolfin::BoxMesh FineMesh(p0, p1, FineMesh_size, FineMesh_size, FineMesh_size);

  // dolfin::UnitCubeMesh mesh(mesh_size, mesh_size, mesh_size);
  EAFE::FunctionSpace CG(mesh);
  Testcase::FunctionSpace FineCG(FineMesh);

  // Define boundary condition
  printf("Define boundary condition\n"); fflush(stdout);
  dolfin::Function u0(CG);
  DirichletBoundary boundary;
  dolfin::Constant zero(0.0);
  u0.interpolate(zero);

  dolfin::DirichletBC bc(CG, u0, boundary);
  dolfin::DirichletBC Finebc(FineCG, u0, boundary);

  printf("\tSave mesh in VTK format\n"); fflush(stdout);
  dolfin::FacetFunction<std::size_t> markedMesh(mesh);
  markedMesh.set_all(1);
  boundary.mark(markedMesh,2);
  dolfin::File fileMesh("./problems/test_eafe/output/mesh.pvd");
  fileMesh << markedMesh;


  // Define analytic expressions
  printf("Define analytic expressions\n"); fflush(stdout);
  dolfin::Constant alpha(1.0);
  dolfin::Constant gamma(0.0);
  dolfin::Function phi(CG);
  Advection betaExpression(phi_x_double,phi_y_double,phi_z_double);
  phi.interpolate(betaExpression);
  dolfin::Function Finephi(FineCG);
  Finephi.interpolate(betaExpression);

  // set RHS
  dolfin::Function f(CG);
  dolfin::Function Finef(FineCG);
  dolfin::Constant unity(1.0);
  f.interpolate(unity);
  Finef.interpolate(unity);

  // Save solution in VTK format
  printf("\tSave RHS in VTK format\n"); fflush(stdout);
  dolfin::File fileRHS("./problems/test_eafe2/output/RHS.pvd");
  fileRHS << f;




  /// EAFE convection problem
  printf("Solve convection problem using EAFE formulation\n"); fflush(stdout);
  // Define variational forms
  printf("\tDefine variational forms\n"); fflush(stdout);
  EAFE::BilinearForm a_eafe(CG,CG);
  EAFE::LinearForm L(CG);
  a_eafe.alpha = alpha;
  a_eafe.beta = phi;
  a_eafe.gamma = gamma;
  a_eafe.eta = phi;
  L.f = f;
  printf("\n");

  printf("Solve test problem\n"); fflush(stdout);
  // Define variational forms
  printf("\tDefine variational forms\n"); fflush(stdout);
  Testcase::BilinearForm a(FineCG,FineCG);
  Testcase::LinearForm  FineL(FineCG);
  a.phi = Finephi;
  FineL.f = Finef;
  printf("\n");


  /// Solve for solutions
  // Compute standard solution via linear solver
  printf("Compute solutions\n"); fflush(stdout);

  // Compute EAFE solution via linear solver
  printf("\tEAFE formulation\n"); fflush(stdout);
  dolfin::EigenMatrix A_eafe;
  dolfin::EigenVector u_eafe_vector;
  dolfin::EigenVector b_eafe;
  assemble(b_eafe,L); bc.apply(b_eafe);
  assemble(A_eafe,a_eafe); bc.apply(A_eafe); A_eafe.compress();
  solve(A_eafe, u_eafe_vector, b_eafe, "bicgstab");
  // convert to Function
  dolfin::Function u_eafe(CG);
  *(u_eafe.vector()) = u_eafe_vector;
  // Save solution in VTK format
  printf("\tSave EAFE solution in VTK format\n"); fflush(stdout);
  dolfin::File file_eafe("./problems/test_eafe2/output/EAFEConvection.pvd");
  file_eafe << u_eafe;

  printf("\tTestcase formulation\n"); fflush(stdout);
  dolfin::EigenMatrix A;
  dolfin::EigenVector u_vector;
  dolfin::EigenVector b;
  assemble(b,FineL); Finebc.apply(b);
  assemble(A,a); Finebc.apply(A); A.compress();
  // A.compress();
  solve(A, u_vector, b, "bicgstab");
  // convert to Function
  dolfin::Function u(FineCG);
  *(u.vector()) = u_vector;
  // Save solution in VTK format
  printf("\tSave standard solution in VTK format\n"); fflush(stdout);
  dolfin::File file("./problems/test_eafe2/output/Convection.pvd");
  file << u;
  printf("\n");

  printf("Compute the error...\n"); fflush(stdout);
  dolfin::Function Proj(CG);
  Proj.interpolate(u);
  dolfin::Function diff(CG);
  diff=u_eafe-Proj;
  // diff.abs();
  printf("\tSave the difference in VTK format\n"); fflush(stdout);
  dolfin::File file2("./problems/test_eafe2/output/diff.pvd");
  file2 << diff;
  double n = norm(*(diff.vector()),"l1");
  printf("\tThe L2 error is %f\n",n); fflush(stdout);


  // exit successfully
  printf("Done\n"); fflush(stdout);
  return 0;
}
