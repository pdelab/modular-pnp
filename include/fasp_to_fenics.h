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
    INT fasp_solver_bdcsr_krylov_block_3(block_dCSRmat *A,
                                       dvector *b,
                                       dvector *x,
                                       itsolver_param *itparam,
                                       AMG_param *amgparam,
                                       dCSRmat *A_diag);
#define FASP_BSR     ON  /** use BSR format in fasp */
}

/*------------- In file: fasp_to_fenics.cpp --------------*/

dCSRmat EigenMatrix_to_dCSRmat(const dolfin::EigenMatrix* mat_A);

dvector EigenVector_to_dvector(const dolfin::EigenVector* vec_A);

dolfin::EigenVector Copy_dvector_to_EigenVector(const dvector* vec_b);

void Copy_dvector_to_Function(dolfin::Function* F, const dvector* vec_b);

#endif
