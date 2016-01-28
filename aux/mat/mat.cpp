/*! \file pnp_adaptive.cpp
 *
 *  \brief Setup and solve the PNP equations using adaptive meshing and FASP
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "es.h"
#include "es_cat.h"
#include "cat_es.h"
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "newton.h"
#include "newton_functs.h"
extern "C"
{
  #include "fasp.h"
  #include "fasp_functs.h"
  #define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;




int main(int argc, char** argv)
{

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  domain_build(&domain_par, &mesh, &subdomains, &surfaces);


  EAFE::FunctionSpace V(mesh);

  Function Cation(V);
  Function Anion(V);
  Function Potential(V);
  Function CatCatFunction(V);
  Function AnAnFunction(V);
  Function EsEsFunction(V);
  Function CatBetaFunction(V);
  Function AnBetaFunction(V);

  Constant eps(1.0);
  Constant Dp(1.0);
  Constant Dn(1.0);
  Constant qp(1.0);
  Constant qn(-1.0);
  Constant zero(0.0);

  EAFE::BilinearForm a_cat(V,V);
  a_cat.alpha = Dp;
  a_cat.gamma = zero;
  EAFE::BilinearForm a_an(V,V);
  a_an.alpha = Dn;
  a_an.gamma = zero;
  EigenMatrix A_cat, A_an;

  CatCatFunction.interpolate(anion);
  CatBetaFunction.interpolate(Potential);
  *(CatBetaFunction.vector()) *= 1.0;
  *(CatBetaFunction.vector()) += *(CatCatFunction.vector());
  AnAnFunction.interpolate(anion);
  AnBetaFunction.interpolate(Potential);
  *(AnBetaFunction.vector()) *= -1.0;
  *(AnBetaFunction.vector()) += *(AnAnFunction.vector());
  a_cat.eta = CatCatFunction;
  a_cat.beta = CatBetaFunction;
  a_an.eta = AnAnFunction;
  a_an.beta = AnBetaFunction;
  assemble(A_cat, a_cat);
  assemble(A_an, a_an);

  es::BilinearForm a_es(V,V);
  a_es.eps = eps;



}
