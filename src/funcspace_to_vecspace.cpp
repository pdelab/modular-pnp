/*! \file funcspace_to_vecspace.cpp
 *
 * \brief Contains functions add matrices from function space to vector space
 */

#include <iostream>
#include <dolfin.h>
#include "funcspace_to_vecspace.h"

using namespace std;
using namespace dolfin;

/**
 * \fn void add_matrix(int l, dolfin::FunctionSpace *V, dolfin::FunctionSpace *V_l, dolfin::EigenMatrix *A, dolfin::EigenMatrix *A_l)
 *
 * \brief add A_l the coeffcients of the matrix A_l to the matrix A when A[i,j] exists
 *    A_l =  is the matrix of the l^th component
 *    A = matrix of the whole system with nc components
 *    l =  l^th component 0<=l<=nc-1
 *    nc = number of components
 *    V = VectorFunction space of dimensin nc
 *    V = Function space of dimension 1
 */
void add_matrix(
  int nc,
  int l, dolfin::FunctionSpace *V,
  dolfin::FunctionSpace *V_l,
  dolfin::EigenMatrix *A,
  dolfin::EigenMatrix *A_l
) {
  // dimensions of matrices
  int nrows_l = A_l->size(0);
  int ncols_l = A_l->size(1);
  int nrows = A->size(0);
  int ncols = A->size(1);
  int flag=0;

  // Test of dimensions
  if ( (V->dim()!=nc*V_l->dim()) || (ncols!=nc*ncols_l) || (nrows!=nc*nrows_l) || (nrows!=ncols) || (nrows!=V->dim())){
    printf("add_matrix ERROR: dimensions do not match\n");
  }

  // pointers to JA/IA/Values of the EigenMatrices
  int* JA_l=(int*) std::get<1>(A_l->data());
  int *IA_l=(int*) std::get<0>(A_l->data());
  double* vals_l = (double*) std::get<2>(A_l->data());
  int* JA=(int*) std::get<1>(A->data());
  int *IA=(int*) std::get<0>(A->data());
  double* vals =  (double*) std::get<2>(A->data());

  // DoF and Vertices index for each space
  std::vector<dolfin::la_index> v_d = vertex_to_dof_map(*V);
  std::vector<long unsigned int> d_v = dof_to_vertex_map(*V);
  std::vector<dolfin::la_index> v_d_l = vertex_to_dof_map(*V_l);
  std::vector<long unsigned int> d_v_l = dof_to_vertex_map(*V_l);

  int k,i,j;
  int ii,jj,kk,jj_temp;
  double el;
  for (i=0;i<nrows_l;i++)
  {
    // i = row index of A_l
    for (k=IA_l[i];k<IA_l[i+1];k++)
    {

      j=JA_l[k]; // j column index for A_l
      el=vals_l[k]; // el=A_l[i,j]
      // dof i = Vertex d_v_l[i] (on V_l)
      // Vertex d_v_l[i] = dof v_d[nc*d_v_l[i]]+l (on V)
      // A[ v_d[nc*d_v_l[i]] , v_d[nc*d_v_l[i]] ] += A_l[i,j]
      ii=v_d[nc*d_v_l[i]]+l;  // row index for A
      jj=v_d[nc*d_v_l[j]]+l;  // colum index for A
      if ( (ii>nrows) || (jj>ncols) ) printf("add_matrix ERROR: vertex/dof are wrong\n");
      flag=0;
      for (kk=IA[ii];kk<IA[ii+1];kk++)
      {
        jj_temp=JA[kk];
        if (jj_temp==jj)
        {
          vals[kk]+=el;
          flag=1;
          break;
        }
      }
      if (flag==0) printf("add_matrix ERROR: could not find the columns -> use add_matrix2\n");
    }
  }

}

/**
 * \fnvoid replace_matrix(int l, dolfin::FunctionSpace *V, dolfin::FunctionSpace *V_l, dolfin::EigenMatrix *A, dolfin::EigenMatrix *A_l)
 *
 * \brief replace A_l the coeffcients of the matrix A_l to the matrix A when A[i,j] exists
 *    A_l =  is the matrix of the l^th component
 *    A = matrix of the whole system with nc components
 *    l =  l^th component 0<=l<=nc-1
 *    nc = number of components
 *    V = VectorFunction space of dimensin nc
 *    V = Function space of dimension 1
 */
void replace_matrix(
  int nc,
  int l,
  dolfin::FunctionSpace *V,
  dolfin::FunctionSpace *V_l,
  dolfin::EigenMatrix *A,
  dolfin::EigenMatrix *A_l
) {
  // dimensions of matrices
  int nrows_l = A_l->size(0);
  int ncols_l = A_l->size(1);
  int nrows = A->size(0);
  int ncols = A->size(1);
  int flag=0;

  // Test of dimensions
  if ( (V->dim()!=nc*V_l->dim()) || (ncols!=nc*ncols_l) || (nrows!=nc*nrows_l) || (nrows!=ncols) || (nrows!=V->dim())){
    printf("replace_matrix ERROR: dimensions do not match\n");
  }

  // pointers to JA/IA/Values of the EigenMatrices
  int* JA_l=(int*) std::get<1>(A_l->data());
  int *IA_l=(int*) std::get<0>(A_l->data());
  double* vals_l = (double*) std::get<2>(A_l->data());
  int* JA=(int*) std::get<1>(A->data());
  int *IA=(int*) std::get<0>(A->data());
  double* vals =  (double*) std::get<2>(A->data());

  // DoF and Vertices index for each space
  std::vector<dolfin::la_index> v_d = vertex_to_dof_map(*V);
  std::vector<long unsigned int> d_v = dof_to_vertex_map(*V);
  std::vector<dolfin::la_index> v_d_l = vertex_to_dof_map(*V_l);
  std::vector<long unsigned int> d_v_l = dof_to_vertex_map(*V_l);

  int k,i,j;
  int ii,jj,kk,jj_temp;
  double el;
  for (i=0;i<nrows_l;i++)
  {
    // i = row index of A_l
    for (k=IA_l[i];k<IA_l[i+1];k++)
    {

      j=JA_l[k]; // j column index for A_l
      el=vals_l[k]; // el=A_l[i,j]
      // dof i = Vertex d_v_l[i] (on V_l)
      // Vertex d_v_l[i] = dof v_d[nc*d_v_l[i]]+l (on V)
      // A[ v_d[nc*d_v_l[i]] , v_d[nc*d_v_l[i]] ] += A_l[i,j]
      ii=v_d[nc*d_v_l[i]]+l;  // row index for A
      jj=v_d[nc*d_v_l[j]]+l;  // colum index for A
      if ( (ii>nrows) || (jj>ncols) ) printf("replace_matrix ERROR: vertex/dof are wrong\n");
      flag=0;
      for (kk=IA[ii];kk<IA[ii+1];kk++)
      {
        jj_temp=JA[kk];
        if (jj_temp==jj)
        {
          vals[kk]=el;
          flag=1;
          break;
        }
      }
      if (flag==0) printf("replace_matrix ERROR: could not find the columns -> use add_matrix2\n");
    }
  }

}

void replace_row(
  int row_index,
  dolfin::EigenMatrix* A
) {
  int* JA = (int*) std::get<1>(A->data());
  int* IA = (int*) std::get<0>(A->data());
  double* vals = (double*) std::get<2>(A->data());

  for (int k = IA[row_index]; k < IA[row_index+1]; k++) {
    if (JA[k]==row_index)
      vals[k]=1.0;
    else
      vals[k]=0.0;
  }
}
