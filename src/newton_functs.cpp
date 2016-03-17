/*! \file newton.cpp
 *
 * \brief Contains files for a newton solver
 *
 */

#include <iostream>
#include <dolfin.h>
#include "newton.h"
#include "newton_functs.h"
#include "mean_exp.h"

using namespace dolfin;

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void update_solution (dolfin::Function* iterate, dolfin::Function* update)
 *
 * \brief adds the update Function to the current iterate
 *
 * \param iterate   Function to be updated
 * \param update    update Function
 */
void update_solution (dolfin::Function* iterate, dolfin::Function* update)
{
    dolfin::FunctionSpace V( *(iterate->function_space()) );
    dolfin::Function update_interpolant(V);
    update_interpolant.interpolate(*update);
    *(iterate->vector()) += *(update_interpolant.vector());
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
