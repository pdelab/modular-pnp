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
#include "Convection.h"
#include "EAFE.h"
using namespace dolfin;

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

class Solution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0]*x[1]*(x[0]-1.0)*(x[1]-1.0);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
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
  printf("Solving the linear PDE:\n");
  printf("\t-div( alpha*exp(eta)*( grad(u) + <b_x,b_y,b_z>*u ) ) + gamma*exp(eta)u = f\n");
  printf("\n"); fflush(stdout);

  // read in coefficients
  char filenm[] = "./problems/test_eafe/coefficients.dat";
  char buffer[100];
  int val;
  int mesh_size;
  double test_problem;
  double alpha_double, eta_double, gamma_double, b_x_double, b_y_double, b_z_double;
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

    else if (strcmp(buffer,"test_problem")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      test_problem = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"alpha_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      alpha_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }
    
    else if (strcmp(buffer,"eta_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      eta_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"gamma_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      gamma_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"b_x_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      b_x_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"b_y_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      b_y_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else if (strcmp(buffer,"b_z_double")==0) {
      val = fscanf(fp,"%s",buffer);
      if (val!=1 || strcmp(buffer,"=")!=0) {
          state = false; break;
      }
      val = fscanf(fp,"%lf",&dbuff);
      if (val!=1) { state = false; break; }
      b_z_double = dbuff;
      fgetsPtr = fgets(buffer,500,fp); // skip rest of line
    }

    else {
        state = true;
        printf(" Bad read-in: %s property unknown \n", buffer); fflush(stdout);
    }  
  }
  fclose(fp);

  if (test_problem>0.0)
    printf("Solving the test problem with u = x*y*(x-1)*(y-1)\n");
  else
    printf("Solving the problem with f = 1.\n");
  printf("Coefficients read in are:\n");
  printf("\talpha: \t%e\n",alpha_double);
  printf("\teta:   \t%e\n",eta_double);
  printf("\tgamma: \t%e\n",gamma_double);
  printf("\tb_x:   \t%e\n",b_x_double);
  printf("\tb_y:   \t%e\n",b_y_double);
  printf("\tb_z:   \t%e\n",b_z_double);
  printf("\n"); fflush(stdout);


  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  printf("Create mesh %d x %d x %d\n",mesh_size,mesh_size,mesh_size); fflush(stdout);
  bool print_matrices = (mesh_size<5)? true : false;
  dolfin::UnitCubeMesh mesh(mesh_size, mesh_size, mesh_size);
  Convection::FunctionSpace CG(mesh);

  // Define boundary condition
  printf("Define boundary condition\n"); fflush(stdout);
  dolfin::Constant u0(0.0);
  DirichletBoundary boundary;
  dolfin::DirichletBC bc(CG, u0, boundary);

  // Define analytic expressions
  printf("Define analytic expressions\n"); fflush(stdout);
  dolfin::Constant alpha(alpha_double);
  
  Advection betaExpression(b_x_double,b_y_double,b_z_double);
  dolfin::Function beta(CG);
  beta.interpolate(betaExpression);

  dolfin::Constant gamma(gamma_double);
  dolfin::Constant eta(eta_double);

  // set RHS
  dolfin::Function f(CG);
  if (test_problem>0.0) {
    testRHS f_rhs(alpha_double,eta_double,gamma_double,b_x_double,b_y_double,b_z_double);
    f.interpolate(f_rhs);
  }
  else {
    dolfin::Constant unity(1.0);
    f.interpolate(unity);
  }
  // Save solution in VTK format
  printf("\tSave RHS in VTK format\n"); fflush(stdout);
  dolfin::File fileRHS("./problems/test_eafe/output/RHS.pvd");
  fileRHS << f;

  // Save analytic solution in VTK format
  if (test_problem>0.0) {
    printf("\tSave true solution in VTK format\n");
    Solution trueSolution;
    dolfin::Function solution(CG);
    solution.interpolate(trueSolution);
    dolfin::File fileSolution("./problems/test_eafe/output/solution.pvd");
    fileSolution << solution;
  }
  printf("\n"); fflush(stdout);



  /// Standard convection problem
  printf("Solve convection problem using standard formulation\n"); fflush(stdout);
  // Define variational forms
  printf("\tDefine variational forms\n"); fflush(stdout);
  Convection::BilinearForm a(CG,CG);
  Convection::LinearForm L(CG);
  a.alpha = alpha;
  a.beta = beta;
  a.gamma = gamma;
  a.eta = eta;
  L.f = f;

  // Compute solution via linear solver
  printf("\tCompute solution\n"); fflush(stdout);
  dolfin::EigenMatrix A;
  dolfin::Vector b; 
  dolfin::Vector u_vector;
  assemble(A,a); bc.apply(A); A.compress();
  assemble(b,L); bc.apply(b);
  solve(A, u_vector, b, "bicgstab");

  // convert to Function
  dolfin::Function u(CG);
  *(u.vector()) = u_vector;

  // Save solution in VTK format
  printf("\tSave solution in VTK format\n\n"); fflush(stdout);
  dolfin::File file("./problems/test_eafe/output/Convection.pvd");
  file << u;



  /// EAFE convection problem
  printf("Solve convection problem using EAFE formulation\n"); fflush(stdout);
  // Define variational forms
  printf("\tDefine variational forms\n"); fflush(stdout);
  EAFE::BilinearForm a_eafe(CG,CG);
  a_eafe.alpha = alpha;
  a_eafe.beta = beta;
  a_eafe.gamma = gamma;
  a_eafe.eta = eta;

  // Compute solution
  printf("\tCompute solution\n"); fflush(stdout);
  dolfin::EigenMatrix A_eafe;
  dolfin::Vector u_eafe_vector;
  assemble(A_eafe,a_eafe); bc.apply(A_eafe); A_eafe.compress();
  solve(A_eafe, u_eafe_vector, b, "bicgstab");

  // convert to Function
  dolfin::Function u_eafe(CG);
  *(u_eafe.vector()) = u_eafe_vector;

  // Save solution in VTK format
  printf("\tSave solution in VTK format\n\n"); fflush(stdout);
  dolfin::File file_eafe("./problems/test_eafe/output/EAFEConvection.pvd");
  file_eafe << u_eafe;



  /// Print stiffness matrices
  if (print_matrices) {
    std::cout << "There are " << A.nnz() << " nonzero entries in the standard formulation\n";
    std::cout << "There are " << A_eafe.nnz() << " nonzero entries in the EAFE formulation\n\n";

    std::cout << "The standard stiffness matrix is:\n";  
    std::string A_string = A.str(true);
    std::cout << A_string << "\n\n";

    std::cout << "The EAFE stiffness matrix is:\n";  
    std::string A_eafe_string = A_eafe.str(true);  
    std::cout << A_eafe_string;
    printf("\n");
  }

  // exit successfully
  printf("Done\n"); fflush(stdout);
  return 0;
}
