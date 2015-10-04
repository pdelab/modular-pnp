/*! \file fasp_to_fenics.cpp
 *
 * \brief Contains functions to transfer EigenMatrix and EigenVector to FASP
 *  or to tranfer FASP to Fenics
 */

#include <iostream>
#include <dolfin.h>
#include "fasp_to_fenics.h"

using namespace std;
using namespace dolfin;

// Link the dolfin::EigenMatrix Mat_A to dCSRmat format from FASP
dCSRmat EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A)
{
  // dimensions of matrix
  int nrows = mat_A->size(0);
  int ncols = mat_A->size(1);
  int nnz = mat_A->nnz();

  // point to JA array
  int* JA;
  JA = (int*) std::get<1>(mat_A->data());

  // construct IA
  int* el_number;
  int* IA;
  el_number = (int*) std::get<0>(mat_A->data());
  IA = (int*) fasp_mem_calloc(nrows+1, sizeof(int));
  for (int i=0; i<nrows; i++)  {
    IA[i] = JA[el_number[i]];
  }
  IA[nrows] = nnz;

  // point to values array
  double* vals;
  vals = (double*) std::get<2>(mat_A->data());

  // assign to dCSRmat
  dCSRmat dCSR_A;
  dCSR_A.nnz=nnz;
  dCSR_A.row=nrows;
  dCSR_A.col=ncols;
  dCSR_A.IA=IA;
  dCSR_A.JA=JA;
  dCSR_A.val=vals;
  return dCSR_A;
}

// Link the dolfin::EigenVector Vec_A to dvector format from FASP
dvector EigenVector_to_dvector(const dolfin::EigenVector* vec_A)
{
  dvector dVec_A;
  dVec_A.row = vec_A->size();
  dVec_A.val = (double*) vec_A->data();
  return dVec_A;
}
