/*! \file newton_solver.h
 *  \brief Main header file for (quasi-)Newton solvers
 *  \note Only define macros and data structures, no function decorations.
 */

#ifndef __NEWTONFUNCTS_HEADER__   /*--- allow multiple inclusions ---*/
#define __NEWTONFUNCTS_HEADER__       /**< indicate newton.h has been included before */

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include "newton.h"
#include "gradient_recovery.h"
#include "poisson_cell_marker.h"
#include "fasp_to_fenics.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
}

/*------------- In file: newton.cpp --------------*/

void update_solution (dolfin::Function* iterate, dolfin::Function* update);

/*------------- In file: params.cpp --------------*/

SHORT newton_param_input_init (newton_param *inparam);

SHORT newton_param_check (newton_param *inparam);

void newton_param_input (const char *filenm, newton_param *inparam);

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

void non_dimesionalize_coefficients (
  domain_param *domain,
  coeff_param *coeffs,
  coeff_param *non_dim_coeffs
);

/*------------- In file: domains.cpp --------------*/

void domain_build (
  domain_param *domain_par,
  dolfin::Mesh *mesh,
  dolfin::MeshFunction<size_t> *subdomains,
  dolfin::MeshFunction<size_t> *surfaces
);

unsigned int check_local_entropy (
  dolfin::Function *cation,
  double cation_valency,
  dolfin::Function *anion,
  double anion_valency,
  dolfin::Function *voltage,
  dolfin::Mesh *target_mesh,
  double entropy_tol,
  uint max_elements,
  uint max_depth
);

unsigned int check_electric_field (
  dolfin::Function *voltage,
  dolfin::Mesh *target_mesh,
  double entropy_tol,
  uint max_elements,
  uint max_depth
);

INT mass_lumping_solver (
  dolfin::EigenMatrix* A,
  dolfin::EigenVector* b,
  dolfin::Function* solution
);


#endif /* end if for __NEWTONFUNCTS_HEADER__ */

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
