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

// class Advection : public Expression
// {
//   void eval(Array<double>& values, const Array<double>& x) const
//   {
//     values[0] = 1.0e+0*x[0] + 1.0e+0*x[1] + 1.0e+0*x[2];
//   }
// };

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
  printf("\t-div( alpha*exp(eta)*( grad(u) + <b_x,b_y,b_z>*u ) ) + gamma*exp(eta)u = 1\n");
  printf("\n"); fflush(stdout);

  // read in coefficients
  char filenm[] = "./problems/test_eafe/coefficients.dat";
  char buffer[100];
  int val;
  double alpha_double, eta_double, gamma_double, b_x_double, b_y_double, b_z_double;
  FILE *fp = fopen(filenm,"r");
  if (fp==NULL) {
    printf("### ERROR: Could not open file %s...\n", filenm);
  }
  bool state = true;
  while ( state ) {
    double  dbuff;
    char   *fgetsPtr;
    
    val = fscanf(fp,"%s",buffer);
    if (val==EOF) break;
    if (val!=1){ state = false; break; }
    if (buffer[0]=='[' || buffer[0]=='%' || buffer[0]=='|') {
        fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        continue;
    }
    
    // match keyword and scan for value
    if (strcmp(buffer,"alpha_double")==0) {
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
  printf("Create mesh\n"); fflush(stdout);
  unsigned int mesh_size = 10;
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
  dolfin::Constant f(1.0);



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

  // Compute solution
  printf("\tCompute solution\n\t"); fflush(stdout);
  dolfin::Function u(CG);
  solve(a == L, u, bc);

  // Save solution in VTK format
  printf("\tSave solution in VTK format\n"); fflush(stdout);
  dolfin::File file("./problems/test_eafe/output/Convection.pvd");
  file << u;



  /// EAFE convection problem
  printf("Solve convection problem using EAFE formulation\n"); fflush(stdout);
  // Define variational forms
  EAFE::BilinearForm a_eafe(CG,CG);
  a_eafe.alpha = alpha;
  a_eafe.beta = beta;
  a_eafe.gamma = gamma;
  a_eafe.eta = eta;

  // Compute solution
  printf("\tCompute solution\n\t"); fflush(stdout);
  dolfin::Function u_eafe(CG);
  solve(a_eafe == L, u_eafe, bc);

  // Save solution in VTK format
  printf("\tSave solution in VTK format\n"); fflush(stdout);
  dolfin::File file_eafe("./problems/test_eafe/output/EAFEConvection.pvd");
  file_eafe << u_eafe;



  /// Print stiffness matrices
  if (print_matrices) {
    dolfin::EigenMatrix A; 
    assemble(A,a); A.compress();
    std::cout << "There are " << A.nnz() << " nonzero entries in the standard formulation\n";

    dolfin::EigenMatrix A_eafe; 
    assemble(A_eafe,a_eafe); A_eafe.compress();
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
