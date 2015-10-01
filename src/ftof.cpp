/*! \file ftof.cpp
 *
 * \brief Contains functions to transfer EigenMatrix and EigenVector to FASP
 *  or to tranfer FASP to Fenics
 */

#include "ftof.h"

using namespace dolfin;

// Link the dolfin::EigenMatrix Mat_A to dCSRmat format from FASP
void EigenMatrixTOdCSRmat( dCSRmat* dCSR_A, const EigenMatrix* mat_A)
{
  int i,j,k=0;
  int nrows = mat_A->size(0);
  int ncols = mat_A->size(1);
  int nnz = mat_A->nnz();
  double* vals;
  int *JA;
  int *el_number; // el_number refer the index in val of the first element in each row
  int *IA;

  el_number= (int *)get<0>(mat_A->data());
  JA = (int *)get<1>(mat_A->data());
  vals = (double *)get<2>(mat_A->data());


  // Consturction of IA
  IA = (int*)fasp_mem_calloc(nrows+1, sizeof(int));
  for (i=0;i<nrows;i++)
  {
    IA[i]=JA[el_number[i]];
  }
  IA [nrows]=nnz;

  dCSR_A->nnz=nnz;
  dCSR_A->row=nrows;
  dCSR_A->col=ncols;

  dCSR_A->IA=IA;
  dCSR_A->JA=JA;
  dCSR_A->val=vals;

}

// Link the dolfin::EigenVector Vec_A to dvector format from FASP
void EigenVectorTOdvector( dvector* dVec_A, const EigenVector* Vec_A)
{

  int i,n;
  double *val;

  n = Vec_A->size();
  val = (double *)Vec_A->data();

  dVec_A->row=n;
  dVec_A->val=val;

}
