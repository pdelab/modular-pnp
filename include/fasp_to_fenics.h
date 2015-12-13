/*! \file fasp_to_fenics.h
 *  \brief Main header file for FASP/FEniCS interface
 */


#ifndef __FASPTOFENICS_H
#define __FASPTOFENICS_H

#include <iostream>
#include <dolfin.h>
#include <math.h>
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
    INT fasp_solver_bdcsr_krylov_block_3(block_dCSRmat *A,
                                       dvector *b,
                                       dvector *x,
                                       itsolver_param *itparam,
                                       AMG_param *amgparam,
                                       dCSRmat *A_diag);
#define FASP_BSR     ON  /** use BSR format in fasp */
}

/*------------- In file: fasp_to_fenics.cpp --------------*/

void EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A, dCSRmat* dCSR_A);

void EigenVector_to_dvector(const dolfin::EigenVector* vec_b, dvector* dVec_b);

void copy_dvector_to_EigenVector(const dvector* vec_b, dolfin::EigenVector* EGVec);

void copy_dvector_to_Function(const dvector* vec_b, dolfin::Function* F);

void get_dofs(dolfin::Function* vector_function, ivector* dof_array, unsigned int component);

void copy_dvector_to_vector_function(const dvector* vector, dolfin::Function* F, ivector* vector_dofs, ivector* function_dofs);

void divide_vec(dolfin::EigenVector *vec1, dolfin::EigenVector *vec2);

#endif
