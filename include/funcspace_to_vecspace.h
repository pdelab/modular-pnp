/*! \file fasp_to_fenics.h
 *  \brief Main header file for FASP/FEniCS interface
 */


#ifndef __FUNCSPACETOVECSPACE_H
#define __FUNCSPACETOVECSPACE_H

#include <iostream>
#include <dolfin.h>

void add_matrix(int l, dolfin::FunctionSpace *V, dolfin::FunctionSpace *V_l, dolfin::EigenMatrix *A, dolfin::EigenMatrix *A_l);

#endif
