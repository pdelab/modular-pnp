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
  std::cout << "#### Test of the interface between FASP and FENiCS      #### \n";
  std::cout << "############################################################ \n";

  dCSRmat dcsr_A;
  EigenMatrix_to_dCSRmat(&EA, &dcsr_A);
  std::cout << "Test of EigenMatrix_to_dCSRmat\n";
  if (dcsr_A.nnz-EA.nnz()==0){
    std::cout << "\tNumber or non-zeros...Success\n";
  }
  else{
    std::cout << "\tNumber or non-zeros...Failure\n";
  }
  if (dcsr_A.row-EA.size(0)==0){
    std::cout << "\tNumber or rows...Success\n";
  }
  else{
    std::cout << "\tNumber or rows...Failure\n";
  }
  if (dcsr_A.col-EA.size(1)==0){
    std::cout << "\tNumber or columes...Success\n";
  }
  else{
    std::cout << "\tNumber or columes...Failure\n";
  }
  int *IA = (int*) std::get<0>(EA.data());
  int* JA= (int*) std::get<1>(EA.data());
  double* vals= (double*) std::get<2>(EA.data());

  if (dcsr_A.IA-IA==0){
    std::cout << "\tIA...Success\n";
  }
  else{
    std::cout << "\tIA...Failure\n";
  }
  if (dcsr_A.JA-JA==0){
    std::cout << "\tJA...Success\n";
  }
  else{
    std::cout << "\tJA...Failure\n";
  }
  if (dcsr_A.val-vals==0){
    std::cout << "\tValues...Success\n";
  }
  else{
    std::cout << "\tValues...Failure\n";
  }

  dvector d_vec;
  EigenVector_to_dvector(&EV,&d_vec);
  std::cout << "Test of EigenVector_to_dvector\n";

  if (d_vec.row-EV.size()==0){
    std::cout << "\tRow...Success\n";
  }
  else{
    std::cout << "\tRow...Failure\n";
  }
  if (d_vec.val-EV.data()==0){
    std::cout << "\tValues...Success\n";
  }
  else{
    std::cout << "\tValues...Failure\n";
  }

  EigenVector EV2(d_vec.row);
  copy_dvector_to_EigenVector(&d_vec, &EV2);
  std::cout << "Test of copy_dvector_to_EigenVector\n";
  if (d_vec.row-EV2.size()==0){
    std::cout << "\tRow...Success\n";
  }
  else{
    std::cout << "\tRow...Failure\n";
  }
  int flag=0;
  for (int i=0;i<5;i++)
  {
    if (std::fabs(d_vec.val[i]-EV2.data()[i])>1E-6){ flag=1;}
  }
  if (flag==0){
    std::cout << "\tValues...Success\n";
  }
  else{
    std::cout << "\tValues...Failure\n";
  }
  EV2.data()[0]=EV2.data()[0]-5.0;
  if (std::fabs(d_vec.val[0]-EV2.data()[0])>1E-6){
    std::cout << "\tHard Copy...Sucess\n";
  }
  else{
    std::cout << "\tHard Copy...Failure\n";
  }

  dolfin::Function F(V);
  copy_dvector_to_Function(&d_vec, &F);
  std::cout << "Test of copy_dvector_to_Function\n";
  if (d_vec.row-F.vector()->size()==0){
    std::cout << "\tRow...Success\n";
  }
  else{
    std::cout << "\tRow...Failure\n";
  }
  int flag2=0;
  std::vector<double> val3(F.vector()->local_size(), 0);
  F.vector()->get_local(val3);
  for (int i=0;i<5;i++)
  {
    if (std::fabs(d_vec.val[i]-val3[i])>1E-6){ flag2=1;}
  }
  if (flag2==0){
    std::cout << "\tValues...Success\n";
  }
  else{
    std::cout << "\tValues...Failure\n";
  }
  d_vec.val[0]=d_vec.val[0]-5.0;
  if (std::fabs(d_vec.val[0]-val3[0])>1E-6){
    std::cout << "\tHard Copy...Sucess\n";
  }
  else{
    std::cout << "\tHard Copy...Failure\n";
  }

  std::cout << "############################################################## \n";
  std::cout << "#### End of the test                                      #### \n";
  std::cout << "############################################################## \n";

  return 0;
}
