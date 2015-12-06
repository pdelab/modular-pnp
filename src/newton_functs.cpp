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
void update_solution (dolfin::Function* iterate, dolfin::Function* update)
{
    dolfin::FunctionSpace V( *(iterate->function_space()) );
    dolfin::Function update_interpolant(V);
    update_interpolant.interpolate(*update);
    *(iterate->vector()) += *(update_interpolant.vector());
}

/**
 * \fn void update_solution_pnp(dolfin::Function* iterate, dolfin::Function* update, double& relative_residual, const Form& L )
 *
 * \brief adds the update Function to the current iterate by testing the residual only for the nonlinear PNP equations
 *              where  L_pnp.CatCat = cationSolution, L_pnp.AnAn = anionSolution, L_pnp.EsEs = potentialSolution.
 *
 * \param iterate             Function to be updated
 * \param update              update Function
 * \param relative_residual   residual from the previous Newton iteration
 * \pram L                    RHS of the equations
 */
void update_solution_pnp(dolfin::Function* iterate, dolfin::Function* update, double& relative_residual, const Form& L, const dolfin::DirichletBC bc, dolfin::EigenVector* b )
{
    double new_residual;
    int i=1;
    dolfin::FunctionSpace V( *(iterate->function_space()) );
    dolfin::Function update_interpolant(V);
    update_interpolant.interpolate(*update);
    dolfin::Function _iterate(iterate);
    *(_iterate->vector()) += *(update_interpolant.vector());
    L.CatCat = _iterate[0];
    L.AnAn   = _iterate[1];
    L.EsEs   = _iterate[2];
    assemble(*b, L);
    bc.apply(*b);
    new_residual = (*b).norm("l2");
    while ( (new_residual>relative_residual) and i<10)
    {
      *(_iterate->vector()) = *(iterate.vector());
      *(_iterate->vector()) += (0.5)**i*(update_interpolant.vector());
      i++;
      L.CatCat = _iterate[0];
      L.AnAn   = _iterate[1];
      L.EsEs   = _iterate[2];
      assemble(*b, L);
      bc.apply(*b);
      new_residual = (*b).norm("l2");
    }
    *(iterate->vector()) = *(_iterate.vector());
    relative_residual = new_residual;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
