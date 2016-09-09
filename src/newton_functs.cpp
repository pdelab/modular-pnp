/*! \file newton.cpp
 *
 * \brief Contains files for a newton solver
 *
 */

#include <iostream>
#include <dolfin.h>
#include "newton.h"
#include "newton_functs.h"

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
void update_solution (std::shared_ptr<dolfin::Function> iterate,
                     std::shared_ptr<dolfin::Function> update)
{
    // auto V = std::make_shared<dolfin::FunctionSpace>(iterate->function_space());
    // std::shared_ptr<dolfin::FunctionSpace> V;
    // *V = *(iterate->function_space());
    auto update_interpolant = std::make_shared<dolfin::Function>(iterate->function_space());
    update_interpolant->interpolate(*update);
    *(iterate->vector()) += *(update_interpolant->vector());
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
