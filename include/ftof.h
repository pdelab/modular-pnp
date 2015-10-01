/*! \file ftof.h
 *  \brief Main header file for Fast/Fenics interface
 */


#ifndef __FTOF_H
#define __FTOF_H

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

void EigenMatrixTOdCSRmat( dCSRmat* dCSR_A, const dolfin::EigenMatrix* mat_A);

void EigenVectorTOdvector( dvector* dVec_A, const dolfin::EigenVector* Vec_A);

#endif
