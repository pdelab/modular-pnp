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

/**
 * \fn dCSRmat EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A)
 *
 * \brief Link the dolfin::EigenMatrix mat_A to dCSRmat format from FASP
 *
 * \param mat_A     EigenMatrix to be converted
 *
 * \return          dCSRmat if conversion successful; otherwise error information.
 */
dCSRmat EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A)
{
  // dimensions of matrix
  int nrows = mat_A->size(0);
  int ncols = mat_A->size(1);
  int nnz = mat_A->nnz();
  // check for uninitialized EigenMatrix
  if ( nrows<1 || ncols<1 || nnz<1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenMatrix_to_dCSRmat");
  }

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

/**
 * \fn dvector EigenVector_to_dvector(const dolfin::EigenVector* vec_b)
 *
 * \brief Link the dolfin::EigenVector vec_b to dvector format from FASP
 *
 * \param vec_b     EigenVector to be converted
 *
 * \return          dvector if conversion successful; otherwise error information.
 */
dvector EigenVector_to_dvector(const dolfin::EigenVector* vec_b)
{
  // check for uninitialized EigenMatrix
  int length = (int) vec_b->size();
  if ( length<1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenVector_to_dvector");
  }

  dvector dVec_b;
  dVec_b.row = length;
  dVec_b.val = (double*) vec_b->data();
  return dVec_b;
}

/**
 * \fn void dvector_to_EigenVector(dolfin::vector* VecSolu, const dvector* vec_b)
 *
 * \brief Link the dolfin::vector vec_b to dvector format from FASP
 *
 * \param
 *        VecSolu  dolfin::function (output)
 *        vec_b    dvector to be converted
 *
 * \return         dolinf::vector if conversion successful; otherwise error information.
 */
EigenVector Copy_dvector_to_EigenVector(const dvector* vec_b)
{
  // check for uninitialized EigenMatrix
  int length = vec_b->row;
  if ( length<1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenVector_to_dvector");
  }
  // Get the array
  EigenVector EGVec(length);
  double * array = EGVec.data();
  for(std::size_t i=0; i<length; ++i) {
      array[i] = vec_b->val[i];
  }
  return EGVec;
}
