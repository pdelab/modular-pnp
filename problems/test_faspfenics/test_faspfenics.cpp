/*! \file test_faspfenics.cpp
 *
 *  \brief Main to test FASP/FENICS interface using the Poisson problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <string>
#include <dolfin.h>
#include "Poisson.h"
#include "fasp_to_fenics.h"

using namespace dolfin;
//using namespace std;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{

  int i;

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen"; // or uBLAS
  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  UnitSquareMesh mesh(3, 3);
  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);

  dolfin::EigenMatrix EA; assemble(EA,a);

  Source f;
  dUdN g;
  L.f = f;
  L.g = g;
  dolfin::EigenVector EV; assemble(EV,L);

  std::cout << "############################################################ \n";
  std::cout << "#### Beginning of test of EigenMatrixTOdCSRmat function #### \n";
  std::cout << "#### the EigenMatrix and the dCSRmat should be the same #### \n";

  std::string s=EA.str(true);
  std::cout << "#### EigenMatrix is\n";
  std::cout << "Nummber of none zero elements = "<< EA.nnz() << "\n";
  std::cout << s;


  dCSRmat bsr_A = EigenMatrix_to_dCSRmat(&EA);
  std::cout << "#### dCSRmat is  \n";
  std::cout << "number number of none zero elements = "<< bsr_A.nnz << "\n";
  std::cout << "number rows ="<< bsr_A.row << "\n";
  std::cout << "number cols ="<< bsr_A.col << "\n";
  std::cout << "values:" << "\t";
  for (i=0;i<bsr_A.nnz;i++)
  {
    std::cout << bsr_A.val[i] << "\t";
  }
  std::cout << "\n";
  std::cout << "JA:" << "\t";
  for (i=0;i<bsr_A.nnz;i++)
  {
    std::cout << bsr_A.JA[i] << "\t";
  }
  std::cout << "\n";
  std::cout << "IA:" << "\t";
  for (i=0;i<bsr_A.row+1;i++)
  {
    std::cout << bsr_A.IA[i] << "\t";
  }
  std::cout << "\n";

  std::cout << "#### End of test of EigenMatrixTOdCSRmat function       #### \n";
  std::cout << "############################################################ \n";
  std::cout << "############################################################ \n";
  std::cout << "#### Beginning of test of EigenVectorTOdVector function #### \n";
  std::cout << "#### the EigenVector and the dvector should be the same #### \n";

  //
  dvector dV = EigenVector_to_dvector(&EV);
  std::string s2=EV.str(true);
  std::cout << "#### Eigen vector is\n";
  std::cout << "Number of cols: \t"<< EV.size() << "\n";
  std::cout << s2 << "\n";
  std::cout << "#### dVector vector is \n";
  for (i=0;i< dV.row  ;i++)
  {
    std::cout <<dV.val[i] << "\t";
  }
  std::cout << "\n";

  std::cout << "#### End of test of EigenVectorTOdVector function       #### \n";
  std::cout << "############################################################ \n";

  free(bsr_A.IA);

  return 0;
}
