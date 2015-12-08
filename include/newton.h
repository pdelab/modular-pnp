/*! \file newton_solver.h
 *  \brief Main header file for (quasi-)Newton solvers
 *  \note Only define macros and data structures, no function decorations.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
}
#ifndef __NEWTON_HEADER__		/*--- allow multiple inclusions ---*/
#define __NEWTON_HEADER__       /**< indicate newton.h has been included before */

/*----------------------*/
/*-- Input parameters --*/
/*----------------------*/

/**
 * \struct newton_param
 * \brief Parameters for a Newton solver
 */
typedef struct {

    //! maximal iteration count
    INT max_it;

    //! a posteriori error tolerance
    REAL adapt_tol;

    //! Tolerance for nonlinear residual
    REAL tol;

    //! backtracking update damp factor
    REAL damp_factor;

    //! maximal iteration count of backtracking update
    INT damp_it;

} newton_param; /**< Parameters for Newton Solver */

/**
 * \struct domain_param
 * \brief Parameters for constructing the mesh
 */
typedef struct {
    //! reference length for domain
    REAL ref_length;
	//! dimension length along x-direction
    REAL length_x;
    //! dimension length along y-direction
    REAL length_y;
    //! dimension length along z-direction
    REAL length_z;
    //! dimension length along time direction
    REAL length_time;

    //! number of vertices along x-direction
    INT grid_x;
    //! number of vertices along y-direction
    INT grid_y;
    //! number of vertices along z-direction
    INT grid_z;
    //! number of vertices along time direction
    INT grid_time;

    //! string specifying location of mesh output file
    char mesh_output[128];
    //! string specifying location of mesh file
    char mesh_file[128];
    //! string specifying location of subdomain file
    char subdomain_file[128];
    //! string specifying location of surface file
    char surface_file[128];

} domain_param; /**< Parameters for constructing domain */


/**
 * \struct coeff_param
 * \brief Coefficients for the PDE
 */
typedef struct {
    //! reference scale for voltage
    REAL ref_voltage;

    //! reference scale for charge density
    REAL ref_density;

    //! temperature
    REAL temperature;

    //! relative permittivity coefficient
    REAL relative_permittivity;

    //! cation diffusivity coefficient
    REAL cation_diffusivity;

    //! cation diffusivity coefficient
    REAL cation_mobility;

    //! cation diffusivity coefficient
    REAL cation_valency;

    //! anion diffusivity coefficient
    REAL anion_diffusivity;

    //! anion diffusivity coefficient
    REAL anion_mobility;

    //! anion diffusivity coefficient
    REAL anion_valency;

} coeff_param; /**< Parameters for setting PDE coefficients */

#endif /* end if for __NEWTON_HEADER__ */

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
