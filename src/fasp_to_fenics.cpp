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
 * \brief Convert the dolfin::EigenMatrix mat_A to dCSRmat format from FASP
 *        and set zero rows to have a unit on the diagonal
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

  // Check for rows of zeros and add a unit diagonal entry
  bool nonzero_entry = false;
  for ( uint rowInd=0; rowInd<nrows; rowInd++ ) {
    // Check for nonzero entry
    nonzero_entry = false;
    int diagColInd = -1;
    if ( IA[rowInd] < IA[rowInd+1] ) {
      for ( uint colInd=IA[rowInd]; colInd < IA[rowInd+1]; colInd++ ) {
        if ( vals[colInd] != 0.0 ) nonzero_entry = true;
        if ( JA[colInd] == rowInd ) diagColInd = colInd;
      }
    }
    if ( diagColInd < 0 ) {
      printf(" ERROR: diagonal entry not allocated!!\n\n Exiting... \n \n"); fflush(stdout);
      printf("      for row %d\n",rowInd); fflush(stdout);
      for ( uint colInd=IA[rowInd]; colInd < IA[rowInd+1]; colInd++ ) {
        printf("          %d\n",JA[colInd]);
      }
    }
    if ( nonzero_entry == false ) {
      printf(" Row %d has only zeros! Setting diagonal entry to 1.0 \n", rowInd); fflush(stdout);
      vals[diagColInd] = 1.0;
    }
  }

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

/**
 * \fn void get_dofs(dolfin::Function* vector_function, unsigned int* dof_array, unsigned int component)
 *
 * \brief get indices for degrees of freedom of a specified component of a vector function
 *
 * \param vector_function    vector-function to be have DoFs extracted by component
 * \param dof_array          array to store dofs
 * \param compenent          desired component of vector-function to be extracted
 */
void get_dofs(dolfin::Function* vector_function, ivector* dof_array, unsigned int component)
{
  dolfin::FunctionSpace W( *(vector_function->function_space()) );
  std::vector<dolfin::la_index> gidx_fn;
  const dolfin::la_index n0 = W.dofmap()->ownership_range().first;
  const dolfin::la_index n1 = W.dofmap()->ownership_range().second;
  const dolfin::la_index num_dofs = n1 - n0;
  std::vector<std::size_t> comp(1);
  comp[0] = component;
  std::shared_ptr<GenericDofMap> dofmap_fn  = W.dofmap()->extract_sub_dofmap(comp, *(W.mesh()));

  for ( CellIterator cell(*(W.mesh())); !cell.end(); ++cell)
  {
    ArrayView<const dolfin::la_index> cell_dofs_fn  = dofmap_fn->cell_dofs(cell->index());
    for (std::size_t i = 0; i < cell_dofs_fn.size(); ++i)
    {
      const std::size_t dof = cell_dofs_fn[i];
      if (dof >= n0 && dof < n1)
        gidx_fn.push_back(dof);
    }
  }
  std::sort(gidx_fn.begin(), gidx_fn.end());
  // Remove duplicates
  gidx_fn.erase(std::unique(gidx_fn.begin(), gidx_fn.end()), gidx_fn.end());

  fasp_ivec_alloc(gidx_fn.size(), dof_array);
  for(std::size_t i=0; i<dof_array->row; i++)
    dof_array->val[i] = gidx_fn[i];
}

/**
 * \fn void copy_dvector_to_vector_function(const dvector* vector, dolfin::Function* F, ivector* vector_dofs, ivector* function_dofs)
 *
 * \brief Copy components of a vector to a Function corresponding to a set of DoFs
 */
void copy_dvector_to_vector_function(const dvector* vector, dolfin::Function* F, ivector* vector_dofs, ivector* function_dofs)
{
  // check for uninitialized vector
  int length = vector->row;
  if ( length < 1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_vector_function");
  }
  if (F->vector()->size() != length) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_vector_function");
  }

  // check for uninitialized or mismatching DoF arrays
  int dof_length = vector_dofs->row;
  if ( dof_length < 1 ) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_vector_function");
  }
  if (function_dofs->row != dof_length) {
    fasp_chkerr(ERROR_INPUT_PAR, "copy_dvector_to_vector_function");
  }
  // convert vector to double*
  for (int i=0; i < dof_length; i++)
    F->vector()->setitem(function_dofs->val[i], vector->val[vector_dofs->val[i]]);
}

void divide_vec(EigenVector *vec1, EigenVector *vec2)
{
  if (vec1->size()!=vec2->size()) printf("Error in divide_vec\n");
  double * dvec1 = vec1->data();
  double * dvec2 = vec2->data();
  for (int i=0;i<vec1->size();i++)
  {
    dvec1[i]=pow(dvec1[i],2)/dvec2[i];
  }
}
