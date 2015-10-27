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
#include "fasp_to_fenics.h"
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
  DirichletBoundary(unsigned int test): SubDomain(),TEST(test) {}
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    if (TEST==2) {  // from EAFE paper
      return on_boundary and (
        (x[0] < 0.25+DOLFIN_EPS and x[1] < 0.00+DOLFIN_EPS)
        or (x[0] < 0.00+DOLFIN_EPS and x[1] < 0.25+DOLFIN_EPS)
        or (x[0] > 0.75-DOLFIN_EPS and x[1] > 1.00-DOLFIN_EPS)
        or (x[0] > 1.00-DOLFIN_EPS and x[1] > 0.75-DOLFIN_EPS)
      );
    }
    else {  // x and y boundaries
      return on_boundary and (
        x[0] < DOLFIN_EPS 
        or x[0] > 1.0 - DOLFIN_EPS
        or x[1] < DOLFIN_EPS 
        or x[1] > 1.0 - DOLFIN_EPS
      );
    }
  }
private:
  unsigned int TEST;
};

/*** test_problem == 1 ***/
class Solution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0]*x[1]*(x[0]-1.0)*(x[1]-1.0);
  }
};

/*** test_problem == 2 ***/
class AdvectionFromPaper : public Expression
{
public:
  AdvectionFromPaper(double alpha, double eta): Expression(), A(alpha), E(eta) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double rho = std::sqrt(x[0]*x[0]+x[1]*x[1]);
    if ( rho + x[0] < 0.55) {
      values[0] = 0;
    }
    else if ( rho + x[0] < 0.65) {
      values[0] = 2.0*(rho-0.55) / (A*std::exp(E));
    }
    else {
      values[0] = 0.2 / (A*std::exp(E));
    }
  }
private:
  double A, E;
};

class BCFromPaper : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    if ( x[0] < 0.5) {
      values[0] = 0;
    }
    else {
      values[0] = 2.1;
    }
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
  int mesh_size, test_problem;
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
      val = fscanf(fp,"%d",&ibuff);
      if (val!=1) { state = false; break; }
      test_problem = ibuff;
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

  if (test_problem==1)
    printf("Solving the test problem with u = x*y*(x-1)*(y-1)\n");
  else if (test_problem==2)
    printf("Solving the test problem from the EAFE paper\n");
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
  printf("Create mesh %d x %d x 3\n",mesh_size,mesh_size); fflush(stdout);
  bool print_matrices = (mesh_size<5)? true : false;
  dolfin::Point p0( 0.0, 0.0, 0.0);
  dolfin::Point p1( 1.0, 1.0, 3.0/((double)mesh_size) );
  dolfin::BoxMesh mesh(p0, p1, mesh_size, mesh_size, 3);

  // dolfin::UnitCubeMesh mesh(mesh_size, mesh_size, mesh_size);
  Convection::FunctionSpace CG(mesh);

  // Define boundary condition
  printf("Define boundary condition\n"); fflush(stdout);
  dolfin::Function u0(CG);
  DirichletBoundary boundary(test_problem);
  if (test_problem==2) {
    BCFromPaper u0FromPaper;
    u0.interpolate(u0FromPaper);
  }
  else {
    dolfin::Constant zero(0.0);
    u0.interpolate(zero);
  }
  dolfin::DirichletBC bc(CG, u0, boundary);

  printf("\tSave mesh in VTK format\n"); fflush(stdout);
  dolfin::FacetFunction<std::size_t> markedMesh(mesh);
  markedMesh.set_all(1);
  boundary.mark(markedMesh,2);
  dolfin::File fileMesh("./problems/test_eafe/output/mesh.pvd");
  fileMesh << markedMesh;


  // Define analytic expressions
  printf("Define analytic expressions\n"); fflush(stdout);
  dolfin::Constant alpha(alpha_double);
  
  dolfin::Function beta(CG);
  if (test_problem==2) {
    AdvectionFromPaper betaFromPaper(alpha_double,eta_double);
    beta.interpolate(betaFromPaper);
  }
  else {
    Advection betaExpression(b_x_double,b_y_double,b_z_double);
    beta.interpolate(betaExpression);
  }

  dolfin::Constant gamma(gamma_double);
  dolfin::Constant eta(eta_double);

  // set RHS
  dolfin::Function f(CG);
  if (test_problem==1) {
    testRHS f_rhs(alpha_double,eta_double,gamma_double,b_x_double,b_y_double,b_z_double);
    f.interpolate(f_rhs);
  }
  else if (test_problem==2) {
    dolfin::Constant zero(0.0);
    f.interpolate(zero);
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
  if (test_problem==1) {
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

  /// EAFE convection problem
  printf("Solve convection problem using EAFE formulation\n"); fflush(stdout);
  // Define variational forms
  printf("\tDefine variational forms\n"); fflush(stdout);
  EAFE::BilinearForm a_eafe(CG,CG);
  a_eafe.alpha = alpha;
  a_eafe.beta = beta;
  a_eafe.gamma = gamma;
  a_eafe.eta = eta;
  printf("\n");


  /// Solve for solutions
  // Compute standard solution via linear solver
  printf("Compute solutions\n"); fflush(stdout);
  dolfin::EigenVector b;
  assemble(b,L); bc.apply(b);

  // Compute EAFE solution via linear solver
  printf("\tEAFE formulation\n"); fflush(stdout);
  dolfin::EigenMatrix A_eafe;
  dolfin::EigenVector u_eafe_vector;
  assemble(A_eafe,a_eafe); bc.apply(A_eafe); A_eafe.compress();
  solve(A_eafe, u_eafe_vector, b, "bicgstab");
  // convert to Function
  dolfin::Function u_eafe(CG);
  *(u_eafe.vector()) = u_eafe_vector;
  // Save solution in VTK format
  printf("\tSave EAFE solution in VTK format\n"); fflush(stdout);
  dolfin::File file_eafe("./problems/test_eafe/output/EAFEConvection.pvd");
  file_eafe << u_eafe;

  printf("\tStandard formulation\n"); fflush(stdout);
  dolfin::EigenMatrix A;
  dolfin::EigenVector u_vector;
  assemble(A,a); bc.apply(A);
  // A.compress();
  solve(A, u_vector, b, "bicgstab");
  // convert to Function
  dolfin::Function u(CG);
  *(u.vector()) = u_vector;
  // Save solution in VTK format
  printf("\tSave standard solution in VTK format\n"); fflush(stdout);
  dolfin::File file("./problems/test_eafe/output/Convection.pvd");
  file << u;
  printf("\n");


  /// solve using FASP
  printf("\tEAFE/FASP formulation\n"); fflush(stdout);
  dCSRmat adaptA_fasp;
  EigenMatrix_to_dCSRmat(&A_eafe, &adaptA_fasp);
  // fasp_dcoo_write("A_fasp.dat", &adaptA_fasp);
  
  dvector adaptb_fasp;
  EigenVector_to_dvector(&b, &adaptb_fasp);
  dvector adaptsoluvec;
  EigenVector_to_dvector(&u_vector, &adaptsoluvec) ;
  fasp_dvec_alloc(adaptb_fasp.row, &adaptsoluvec);
  fasp_dvec_set(adaptb_fasp.row, &adaptsoluvec, 0.0);
  printf("\t...initialize solver parameters\n"); fflush(stdout);
  // initialize solver parameters
  input_param inpar;  // parameters from input files
  itsolver_param itpar;  // parameters for itsolver
  AMG_param amgpar; // parameters for AMG
  ILU_param ilupar; // parameters for ILU
  char inputfile[] = "./problems/test_eafe/bsr.dat";
  fasp_param_input(inputfile, &inpar);
  fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
  INT status = FASP_SUCCESS;
  status = fasp_solver_dcsr_krylov(&adaptA_fasp, &adaptb_fasp, &adaptsoluvec, &itpar);
  dolfin::Function u_fasp(CG);
  copy_dvector_to_Function(&adaptsoluvec, &u_fasp);
  printf("\tSave EAFE/FASP solution in VTK format\n"); fflush(stdout);
  dolfin::File file_FASP("./problems/test_eafe/output/FASPConvection.pvd");
  file_FASP << u_fasp;
  printf("\n");

  /// Print stiffness matrices
  if (print_matrices) {
    printf("Printing stiffness matrices\n"); fflush(stdout);
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
