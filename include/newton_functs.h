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

void print_newton_param (newton_param *inparam);


SHORT domain_param_input_init (domain_param *inparam);

SHORT domain_param_check (domain_param *inparam);

void domain_param_input (const char *filenm,
                      	 domain_param *inparam);

void print_domain_param (domain_param *inparam);


SHORT coeff_param_input_init (coeff_param *inparam);

SHORT coeff_param_check (coeff_param *inparam);

void coeff_param_input (const char *filenm,
                      	 coeff_param *inparam);

void print_coeff_param (coeff_param *inparam);

void non_dimesionalize_coefficients (domain_param *domain,
                                     coeff_param *coeffs,
                                     coeff_param *non_dim_coeffs);

/*------------- In file: domains.cpp --------------*/

void domain_build (domain_param *domain_par,
				   dolfin::Mesh *mesh,
				   dolfin::MeshFunction<size_t> *subdomains,
				   dolfin::MeshFunction<size_t> *surfaces,
				   dolfin::File *mesh_output);

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
