/*! \file newton_solver.h
 *  \brief Main header file for (quasi-)Newton solvers
 *  \note Only define macros and data structures, no function decorations.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include "newton.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
}

/*------------- In file: newton.cpp --------------*/

REAL newton_compute_residual (dvector *residual);

REAL newton_compute_relative_residual (dvector *residual,
									   dvector *initial_residual);

/*------------- In file: params.cpp --------------*/

SHORT newton_param_input_init (newton_param *inparam);

SHORT newton_param_check (newton_param *inparam);

void newton_param_input (const char *filenm,
                      	 newton_param *inparam);

SHORT domain_param_input_init (domain_param *inparam);

SHORT domain_param_check (domain_param *inparam);

void domain_param_input (const char *filenm,
                      	 domain_param *inparam);

/*------------- In file: domains.cpp --------------*/

void domain_build (domain_param *domain_par,
				   dolfin::Mesh *mesh,
				   dolfin::MeshFunction<size_t> *subdomains,
				   dolfin::MeshFunction<size_t> *surfaces,
				   dolfin::File *mesh_output);

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
