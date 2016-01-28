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


dBSRmat EigenMatrix_to_dBSRmat (EigenMatrix **A,const INT nb);


int main(int argc, char** argv)
{

  //*************************************************************
  //  Initialization
  //*************************************************************
  printf("Initialize the problem\n"); fflush(stdout);
  // read domain parameters
  printf("\tdomain...\n"); fflush(stdout);
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/PNP/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
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
  EigenMatrix A_cat, A_an, A_es,A_es_cat,A_cat_es,A_es_an,A_an_es;

  CatCatFunction.interpolate(Cation);
  CatBetaFunction.interpolate(Potential);
  *(CatBetaFunction.vector()) *= 1.0;
  *(CatBetaFunction.vector()) += *(CatCatFunction.vector());
  AnAnFunction.interpolate(Anion);
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
  a_es.eps = eps;
  assemble(A_es, a_es);

  es_cat::BilinearForm a_es_cat(V,V);
  a_es_cat.q = qp;
  a_es_cat.D = Dp;
  a_es_cat.CatCat = Cation;
  assemble(A_es_cat, a_es_cat);

  cat_es::BilinearForm a_cat_es(V,V);
  a_cat_es.q = qp;
  a_cat_es.CatCat = Cation;
  assemble(A_cat_es, a_cat_es);

  es_cat::BilinearForm a_es_an(V,V);
  a_es_an.q = qp;
  a_es_an.D = Dp;
  a_es_an.CatCat = Anion;
  assemble(A_es_an, a_es_an);

  cat_es::BilinearForm a_an_es(V,V);
  a_an_es.q = qp;
  a_an_es.CatCat = Anion;
  assemble(A_an_es, a_an_es);

  int nrows = A_cat.size(0);
  int ncols = A_cat.size(1);
  int nnz = A_cat.nnz();

  int nrows2 = A_es.size(0);
  int ncols2 = A_es.size(1);
  int nnz2 = A_es.nnz();

  int nrows3 = A_cat_es.size(0);
  int ncols3 = A_cat_es.size(1);
  int nnz3 = A_cat_es.nnz();

  int* JA = (int*) std::get<1>(A_cat.data());
  int *IA = (int*) std::get<0>(A_cat.data());
  double* vals = (double*) std::get<2>(A_cat.data());

  int* JA2 = (int*) std::get<1>(A_es.data());
  int *IA2 = (int*) std::get<0>(A_es.data());
  double* vals2 = (double*) std::get<2>(A_es.data());

  int* JA3 = (int*) std::get<1>(A_cat_es.data());
  int *IA3 = (int*) std::get<0>(A_cat_es.data());
  double* vals3 = (double*) std::get<2>(A_cat_es.data());


  for (int i=0;i<5;i++)
  {
    std::cout << IA[i] << "\t" << IA2[i] << "\t" << IA3[i] << std::endl;
  }


}

dBSRmat EigenMatrix_to_dBSRmat (EigenMatrix **A,const INT nb)
{
  INT i, j, k, ii, jj, kk, l, mod;
  INT row   = A[0]->size(0);
  INT col   = A[0]->size(1);
  INT nb2   = nb*nb;
  INT *IA   = (int*) std::get<0>(A[0]->data());
  INT *JA   =(int*) std::get<0>(A[1]->data());
  INT nnz = A[0]->nnz();

  double *val[nb];
  for (i=0;i<nb;i++){
    val[i]=(double*) std::get<2>(A[0]->data());
  }

  dBSRmat B;
  B.ROW = row;
  B.COL = col;
  B.nb  = nb;
  B.storage_manner = 0;

  // allocate ia for B
  INT *ia = (INT *) fasp_mem_calloc(row+1, sizeof(INT));
  // allocate ja and bval
  INT *ja = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
  REAL *bval = (REAL*)fasp_mem_calloc(nnz*nb2, sizeof(REAL));

  // Get ia for BSR format
  for (i=0; i<row+1; i++) {
    ia[i] = IA[i];
  }

  // Get ja for BSR format
  for (i=0; i<nnz; i++) {
    ia[i] = JA[i];
  }

  // Get non-zeros of BSR
  for (i=0; i<row; i++) {
    for(j=0; j<nb2; j++) {
      for(k=IA[i]; k<IA[i+1]; k++) {
          mod = JA[k];
          bval[l*nb2+j*nb+mod] = val[i*nb+j][k];
      }
    }
  }


  B.NNZ = nnz;
  B.IA = ia;
  B.JA = ja;
  B.val = bval;


  return B;
}
