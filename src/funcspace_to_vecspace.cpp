/*! \file fasp_to_fenics.cpp
 *
 * \brief Contains functions to transfer EigenMatrix and EigenVector to FASP
 *  or to tranfer FASP to Fenics
 */

#include <iostream>
#include <dolfin.h>
#include "funcspace_to_vecspace.h"

using namespace std;
using namespace dolfin;

/**
 * \fnvoid add_matrix(int l, dolfin::FunctionSpace *V, dolfin::FunctionSpace *V_l, dolfin::EigenMatrix *A, dolfin::EigenMatrix *A_l)
 *
 * \brief add A_l to A for l=0,1,2 is the lth component
 *
 */
void add_matrix(int l, dolfin::FunctionSpace *V, dolfin::FunctionSpace *V_l, dolfin::EigenMatrix *A, dolfin::EigenMatrix *A_l)
{
  // dimensions of matrices
  int nrows_l = A_l->size(0);
  int ncols_l = A_l->size(1);
  int nrows = A->size(0);
  int ncols = A->size(1);

  // point to JA/IA/Values array
  int* JA_l=(int*) std::get<1>(A_l->data());
  int *IA_l=(int*) std::get<0>(A_l->data());
  double* vals_l = (double*) std::get<2>(A_l->data());
  int* JA=(int*) std::get<1>(A->data());
  int *IA=(int*) std::get<0>(A->data());
  double* vals =  (double*) std::get<2>(A->data());

  // DoF and Vertices
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
      // j column index for A_l
      j=JA_l[k];
      // el=A_l[i,j]
      el=vals_l[k];
      // dof i = Vertex d_v0[i] (on V_cat)
      // Vertex d_v0[i] = dof 3*v_d[3*d_v0[i]] (on V)
      // A_pnp[ dof 3*v_d[3*d_v0[i]] , dof 3*v_d[3*d_v0[j]] ] += A_cat[i,j]
      ii=3*v_d[3*d_v_l[i]]+3*l;
      jj=3*v_d[3*d_v_l[j]]+3*l;
      for (kk=IA[ii];kk<IA[ii+1];kk++)
      {
        jj_temp=JA[kk];
        if (jj_temp==jj)
        {
          // printf("Found one\n");
          vals[kk]+=el;
        }
      }
    }
  }

}
