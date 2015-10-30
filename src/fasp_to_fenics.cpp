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
 * \fn void EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A, dCSRmat dCSR_A)
 *
 * \brief Link the dolfin::EigenMatrix mat_A to dCSRmat format from FASP
 *
 * \param mat_A     EigenMatrix to be converted
 */
void EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A, dCSRmat* dCSR_A)
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

  int *IA;
  IA = (int*) std::get<0>(mat_A->data());

  // point to values array
  double* vals;
  vals = (double*) std::get<2>(mat_A->data());

  // assign to dCSRmat
  dCSR_A->nnz = nnz;
  dCSR_A->row = nrows;
  dCSR_A->col = ncols;
  dCSR_A->IA = IA;
  dCSR_A->JA = JA;
  dCSR_A->val = vals;
}

/**
 * \fn void EigenVector_to_dvector(const dolfin::EigenVector* vec_b, dvector* dVec_b)
 *
 * \brief Link the dolfin::EigenVector vec_b to dvector format from FASP
 *
 * \param vec_b     EigenVector to be converted
 */
void EigenVector_to_dvector(const dolfin::EigenVector* vec_b, dvector* dVec_b)
{
  // check for uninitialized EigenMatrix
  int length = (int) vec_b->size();
  if ( length<1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenVector_to_dvector");
  }

  dVec_b->row = length;
  dVec_b->val = (double*) vec_b->data();
}

/**
 * \fn void copy_dvector_to_EigenVector(const dvector* vec_b, dolfin::EigenVector* EGVec)
 *
 * \brief Copy the dvector vec_b to dolfin::EigenVector
 *
 * \param VecSolu  dolfin::function (output)
 * \param vec_b    dvector to be converted
 */
void copy_dvector_to_EigenVector(const dvector* vec_b, dolfin::EigenVector* EGVec)
{
  // check for uninitialized dvector
  int length = vec_b->row;
  if ( length<1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_EigenVector");
  }
  if ( length!=EGVec->size() ) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_EigenVector");
  }

  //printf("%d is the length\n", length); fflush(stdout);
  // Get the array
  double* array = EGVec->data();
  for(std::size_t i=0; i < length; ++i)
  {
      array[i] = vec_b->val[i];
  }
}

/**
 * \fn void copy_dvector_to_Function(doflin::Function* F, const dvector* vec_b)
 *
 * \brief Link copy to dvector vec_b to dolfin::Function.vector()
 *
 * \param vec_b    dvector to be converted
 * \param F        dolfin::Function*  (output)
 */
void copy_dvector_to_Function(const dvector* vec_b, dolfin::Function* F)
{
  // check for uninitialized EigenMatrix
  int length = vec_b->row;
  if ( length < 1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_Function");
  }
  if (F->vector()->size() != length) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_Function");
  }

  std::vector<double> values(F->vector()->local_size(), 0);
  for (int i=0; i < length; i++)
  {
    values[i] = vec_b->val[i];
  }
  F->vector()->set_local(values);
}
