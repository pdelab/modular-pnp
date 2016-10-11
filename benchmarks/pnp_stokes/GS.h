#ifndef __GS_H
#define __GS_H

#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>
#include <dolfin.h>

#include "pnp_with_stokes.h"
#include "stokes_with_pnp.h"
#include "EAFE.h"
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "pnp_with_stokes.h"
#include "stokes_with_pnp.h"
#include "L2Error.h"
#include "energy.h"
#include "newton.h"
#include "newton_functs.h"

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <ufc.h>

extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
#include "fasp4ns.h"
#include "fasp4ns_functs.h"

    INT fasp_solver_bdcsr_krylov_block_3(block_dCSRmat *A,
                                       dvector *b,
                                       dvector *x,
                                       itsolver_param *itparam,
                                       AMG_param *amgparam,
                                       dCSRmat *A_diag);
    INT fasp_solver_bdcsr_krylov_navier_stokes_with_pressure_mass (block_dCSRmat *Mat,
                                                                   dvector *b,
                                                                   dvector *x,
                                                                   itsolver_ns_param *itparam,
                                                                   AMG_ns_param *amgnsparam,
                                                                   ILU_param *iluparam,
                                                                   Schwarz_param *schparam,
                                                                   dCSRmat *Mp);
#define FASP_BSR     ON  /** use BSR format in fasp */
#define FASP_NS_MASS ON  /** use NS solver with pressure mass matrix */
}


class FluidVelocity : public dolfin::Expression
{
public:
    FluidVelocity(double out_flow, double in_flow, double bc_dist, int bc_dir): Expression(3),outflow(out_flow),inflow(in_flow),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
    {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
        if ( std::fabs(x[0]) > 0.5 ) {
            values[bc_direction]  = outflow*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
            values[bc_direction] -=  inflow*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
        }
    }
private:
    double outflow, inflow, bc_distance;
    int bc_direction;
};


INT electrokinetic_block_guass_seidel (
  dBSRmat* A_pnp,
  block_dCSRmat* A_stokes,
  pnp_with_stokes::LinearForm* pnp_rhs_form,
  stokes_with_pnp::LinearForm* stokes_rhs_form,
  const dolfin::DirichletBC* pnp_bc,
  const dolfin::DirichletBC* stokes_bc,

  std::shared_ptr<dolfin::Function> dPNP,
  std::shared_ptr<dolfin::Function> dStokes,
  ivector* dof_u,
  ivector* dof_p,

  double relative_residual_tol,
  unsigned int max_bgs_it,

  itsolver_param* pnp_itpar,
  AMG_param* pnp_amgpar,
  itsolver_ns_param* stokes_itparam,
  AMG_ns_param* stokes_amgparam,
  ILU_param* stokes_iluparam,
  Schwarz_param* stokes_schparam

) {

  // initialize guess is zero
  auto zero = std::make_shared<dolfin::Constant>(0.0);
  auto zero_vec = std::make_shared<dolfin::Constant>(0.0, 0.0, 0.0);

  auto dCation = std::make_shared<dolfin::Function>((*dPNP)[0]);
  auto dAnion = std::make_shared<dolfin::Function>((*dPNP)[1]);
  auto dPhi = std::make_shared<dolfin::Function>((*dPNP)[2]);
  auto dU = std::make_shared<dolfin::Function>((*dStokes)[0]);
  auto dPressure = std::make_shared<dolfin::Function>((*dStokes)[1]);

  dCation->interpolate(*zero);
  dAnion->interpolate(*zero);
  dPhi->interpolate(*zero);
  dU->interpolate(*zero_vec);
  dPressure->interpolate(*zero);

  // update pnp_rhs_form
  pnp_rhs_form->dCat = dCation;
  pnp_rhs_form->dAn = dAnion;
  pnp_rhs_form->dPhi = dPhi;
  pnp_rhs_form->du = dU;
  dolfin::EigenVector pnp_rhs;
  assemble(pnp_rhs, *pnp_rhs_form);
  pnp_bc->apply(pnp_rhs);

  // update stokes_rhs_form
  stokes_rhs_form->dPhi =  dPhi;
  stokes_rhs_form->du = dU;
  stokes_rhs_form->dPress = dPressure;
  dolfin::EigenVector stokes_rhs;
  assemble(stokes_rhs, *stokes_rhs_form);
  stokes_bc->apply(stokes_rhs);

  int index_fix= dof_p->val[0];
  stokes_rhs[index_fix]=0.0;

  // compute initial residual
  double pnp_res = pnp_rhs.norm("l2");
  double stokes_res = stokes_rhs.norm("l2");
  double initial_residual = pnp_res * pnp_res + stokes_res * stokes_res;
  double relative_residual = 1.0;

  // initialize FASP arrays
  dvector pnp_rhs_fasp, pnp_soln_fasp;
  dvector stokes_rhs_fasp, stokes_soln_fasp;
  std::vector<double> pnp_value_vector;
  std::vector<double> stokes_value_vector;
  pnp_value_vector.reserve(pnp_rhs.size());
  stokes_value_vector.reserve(stokes_rhs.size());


  // Block Gauss-Seidel loop
  INT pnp_status = FASP_SUCCESS;
  INT stokes_status = FASP_SUCCESS;
  unsigned int index, iteration_count = 0;
  // printf("%d %d %f %f\n",iteration_count,max_bgs_it,relative_residual,relative_residual_tol);
  while ( (iteration_count < max_bgs_it) && (relative_residual > relative_residual_tol) ){

    EigenVector_to_dvector(&pnp_rhs, &pnp_rhs_fasp);
    fasp_dvec_alloc(pnp_rhs.size(), &pnp_soln_fasp);
    fasp_dvec_set(pnp_rhs_fasp.row, &pnp_soln_fasp, 0.0);
    // solve for pnp update
    pnp_status = fasp_solver_dbsr_krylov_amg (
      A_pnp,
      &pnp_rhs_fasp,
      &pnp_soln_fasp,
      pnp_itpar,
      pnp_amgpar
    );
    // printf("pnp status %d",pnp_status);
    if (pnp_status < 0)
      printf("\n### WARNING: PNP solver failed! Exit status = %d.\n\n", pnp_status);
    else
      printf("\tsolved PNP linearized system successfully...\n");

    // convert pnp update to functions
    for (index = 0; index < pnp_soln_fasp.row; index++) {
      pnp_value_vector[index] = pnp_soln_fasp.val[index];
    }
    (*dPNP).vector()->set_local(pnp_value_vector);

    // update pnp_rhs_form & stokes_rhs_form with pnp update
    auto dCation = std::make_shared<dolfin::Function>((*dPNP)[0]);
    auto dAnion = std::make_shared<dolfin::Function>((*dPNP)[1]);
    auto dPhi = std::make_shared<dolfin::Function>((*dPNP)[2]);
    pnp_rhs_form->dCat = dCation;
    pnp_rhs_form->dAn = dAnion;
    pnp_rhs_form->dPhi = dPhi;
    stokes_rhs_form->dPhi = dPhi;
    assemble(stokes_rhs, *stokes_rhs_form);
    stokes_bc->apply(stokes_rhs);
    stokes_rhs[index_fix]=0.0;



    // solve for stokes update and convert to functions
    // EigenVector_to_dvector(&stokes_rhs, &stokes_rhs_fasp);
    copy_EigenVector_to_block_dvector(&stokes_rhs, &stokes_rhs_fasp, dof_u, dof_p);
    fasp_dvec_alloc(stokes_rhs.size(), &stokes_soln_fasp);
    fasp_dvec_set(stokes_rhs_fasp.row, &stokes_soln_fasp, 0.0);
    stokes_status = fasp_solver_bdcsr_krylov_navier_stokes (
      A_stokes,
      &stokes_rhs_fasp,
      &stokes_soln_fasp,
      stokes_itparam,
      stokes_amgparam,
      stokes_iluparam,
      stokes_schparam
    );
    if (stokes_status < 0)
      printf("\n### WARNING: Stokes solver failed! Exit status = %d.\n\n", stokes_status);
    else
      printf("\tsolved Stokes system successfully....\n");

    // convert stokes update to functions
    for (index = 0; index < dof_u->row; index++) {
      stokes_value_vector[dof_u->val[index]] = stokes_soln_fasp.val[index];
    }
    for (index = 0; index < dof_p->row; index++) {
      stokes_value_vector[dof_p->val[index]] = stokes_soln_fasp.val[dof_u->row+index];
    }
    (*dStokes).vector()->set_local(stokes_value_vector);

    // update pnp_rhs_form & stokes_rhs_form with stokes update
    auto dU = std::make_shared<dolfin::Function>((*dStokes)[0]);
    auto dPressure = std::make_shared<dolfin::Function>((*dStokes)[1]);

    pnp_rhs_form->du = dU;
    stokes_rhs_form->du = dU;
    stokes_rhs_form->dPress = dPressure;

    assemble(pnp_rhs, *pnp_rhs_form);
    pnp_bc->apply(pnp_rhs);
    assemble(stokes_rhs, *stokes_rhs_form);
    stokes_bc->apply(stokes_rhs);

		fasp_dvec_free(&pnp_soln_fasp);
		fasp_dvec_free(&stokes_soln_fasp);
    fasp_dvec_free(&stokes_rhs_fasp);

    // update relative residual
    pnp_res = pnp_rhs.norm("l2");
    stokes_res = stokes_rhs.norm("l2");
    relative_residual = (pnp_res * pnp_res + stokes_res * stokes_res) / initial_residual;
    iteration_count+=1;
  }

  return FASP_SUCCESS;
};

#endif
