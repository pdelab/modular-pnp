/*! \file fasp_to_fenics.h
 *  \brief Main header file for FASP/FEniCS interface
 */


#ifndef __FASPTOFENICS_H
#define __FASPTOFENICS_H

#include <iostream>
#include <dolfin.h>
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
#include "fasp4ns.h"
#include "fasp4ns_functs.h"

    INT fasp_solver_bdcsr_krylov_block_3(block_dCSRmat *A,
                                       dvector *b,
                                       dvector *x,
                                       itsolver_param *itparam,
                                       AMG_param *amgparam,
                                       dCSRmat *A_diag);
    INT fasp_solver_bdcsr_krylov_navier_stokes_with_pressure_mass (block_dCSRmat *Mat,
                                                                   dvector *b,
                                                                   dvector *x,
                                                                   itsolver_ns_param *itparam,
                                                                   AMG_ns_param *amgnsparam,
                                                                   ILU_param *iluparam,
                                                                   Schwarz_param *schparam,
                                                                   dCSRmat *Mp);
#define FASP_BSR     ON  /** use BSR format in fasp */
#define FASP_NS_MASS ON  /** use NS solver with pressure mass matrix */
}

/*------------- In file: fasp_to_fenics.cpp --------------*/

void EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A, dCSRmat* dCSR_A);

void EigenVector_to_dvector(const dolfin::EigenVector* vec_b, dvector* dVec_b);

void copy_dvector_to_EigenVector(const dvector* vec_b, dolfin::EigenVector* EGVec);

void copy_dvector_to_Function(const dvector* vec_b, dolfin::Function* F);

void get_dofs(dolfin::Function* vector_function, ivector* dof_array, unsigned int component);

void copy_dvector_to_vector_function(const dvector* vector, dolfin::Function* F, ivector* vector_dofs, ivector* function_dofs);

void copy_EigenMatrix_to_block_dCSRmat(dolfin::EigenMatrix* A, block_dCSRmat* A_block, ivector* dof_u, ivector* dof_p);

void copy_EigenVector_to_block_dvector(dolfin::EigenVector* b, dvector* b_block, ivector* dof_u, ivector* dof_p);

#endif
