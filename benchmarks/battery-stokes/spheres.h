#ifndef __SPHERES_H
#define __SPHERES_H

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


double xc[86] = { -17.0,18.0,25.0,-1.0,20.0,14.0,22.0,2.0,23.0,-2.0,3.0,
	-13.0,24.0,-4.0,22.0,21.0,-11.0,-24.0,-9.0,-18.0,-14.0,
	-2.0,1.0,17.0,16.0,-9.0,21.0,-24.0,-1.0,-7.0,-14.0,
	12.0,8.0,-8.0,8.0,3.0,7.0,-20.0,25.0,-22.0,-13.0,
	-14.0,-21.0,21.0,-3.0,22.0,21.0,-3.0,6.0,23.0,12.0,
	0.0,-6.0,-6.0,-12.0,-7.0,-4.0,-24.0,17.0,1.0,-22.0,
	-15.0,-21.0,-22.0,22.0,-24.0,-19.0,-15.0,15.0,-2.0,9.0,
	-5.0,19.0,-23.0,15.0,18.0,-16.0,17.0,9.0,19.0,-7.0,
	-20.0,-13.0,-3.0,-16.0,-11.0};

double yc[86] = { -16.0,11.0,-15.0,-3.0,-20.0,-2.0,20.0,20.0,15.0,-16.0,0.0,
	18.0,-24.0,5.0,-16.0,-15.0,-15.0,6.0,9.0,23.0,22.0,
	19.0,-12.0,-3.0,-23.0,4.0,24.0,22.0,-22.0,4.0,-9.0,
	-9.0,-15.0,-23.0,-15.0,14.0,-2.0,6.0,2.0,-23.0,-14.0,
	-25.0,22.0,-19.0,9.0,-15.0,20.0,4.0,15.0,-12.0,17.0,
	17.0,2.0,9.0,20.0,18.0,7.0,22.0,16.0,16.0,10.0,
	5.0,-23.0,-17.0,10.0,-15.0,20.0,24.0,-8.0,-7.0,2.0,
	-5.0,-22.0,12.0,-16.0,-13.0,-11.0,-22.0,-3.0,-25.0,2.0,
	3.0,10.0,-3.0,-24.0,-22.0};

double zc[86] = { 7.0,-19.0,-15.0,24.0,-19.0,-20.0,1.0,24.0,14.0,11.0,-21.0,
	-5.0,9.0,-7.0,21.0,15.0,20.0,17.0,-23.0,13.0,8.0,
	14.0,19.0,-4.0,-12.0,-5.0,-24.0,-6.0,2.0,10.0,-1.0,
	-20.0,-24.0,-18.0,-18.0,16.0,24.0,-22.0,13.0,7.0,-17.0,
	-17.0,-7.0,0.0,0.0,14.0,7.0,8.0,-22.0,-23.0,8.0,
	2.0,6.0,22.0,6.0,18.0,2.0,20.0,6.0,11.0,5.0,
	22.0,-6.0,10.0,1.0,-22.0,12.0,-15.0,-24.0,-7.0,-19.0,
	-15.0,-8.0,22.0,-3.0,-16.0,18.0,-17.0,-19.0,9.0,-9.0,
	-18.0,-8.0,16.0,13.0,24.0};

double rc[86] = { 9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,
	9.0,9.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,
	7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,
	7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,
	7.0,7.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,5.0};

int Numb_spheres = 20;

class SpheresSubDomain : public dolfin::SubDomain
{
    bool inside(const dolfin::Array<double>& x, bool on_boundary) const
    {
      bool flag=false;
      for (int i=0;i<Numb_spheres;i++){
          if (on_boundary && (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0) )
                  flag=true;
                }
          return flag;
    }

};

/// Initialize expressions
class LogCharge_SPH : public dolfin::Expression
{
public:
  // constructor
  LogCharge_SPH(double lower_val, double upper_val,
    double lower, double upper, int bc_coord)
		{
			_lower_val = lower_val;
			_upper_val = upper_val;
			_lower = lower;
			_upper = upper;
			_bc_coord = bc_coord;
		}
  // evaluate LogCarge
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
	{
		values[0]  = std::log(_lower_val) * (_upper - x[_bc_coord]) / (_upper - _lower);
	  values[0] += std::log(_upper_val) * (x[_bc_coord] - _lower) / (_upper - _lower);
		// for (int i=0;i<Numb_spheres;i++){
		// 		if (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0)
		// 						values[0]=1.0;
		// }
	}
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};


class Potential_SPH: public dolfin::Expression
{
public:
  // constructor
  Potential_SPH(double lower_val, double upper_val,
    double lower, double upper, int bc_coord)
		{
		  _lower_val = lower_val;
		  _upper_val = upper_val;
		  _lower = lower;
		  _upper = upper;
		  _bc_coord = bc_coord;
		}
  // evaluate Voltage
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
	{
	  values[0]  = _lower_val * (_upper - x[_bc_coord]) / (_upper - _lower);
	  values[0] += _upper_val * (x[_bc_coord] - _lower) / (_upper - _lower);
		// for (int i=0;i<Numb_spheres;i++){
		// 		if (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0)
		// 						values[0]=1.0;
		// }
	}
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};


INT electrokinetic_block_guass_seidel (
  dBSRmat* A_pnp,
  block_dCSRmat* A_stokes,
  pnp_with_stokes::LinearForm* pnp_rhs_form,
  stokes_with_pnp::LinearForm* stokes_rhs_form,
  const dolfin::DirichletBC* pnp_bc,
  const dolfin::DirichletBC* stokes_bc,

  dolfin::Function* dPNP,
  dolfin::Function* dStokes,

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
  dolfin::Constant zero(0.0);
  dolfin::Constant zero_vector(0.0, 0.0, 0.0);
  // dolfin::Function dCation((*dPNP)[0]);
  // dolfin::Function dAnion((*dPNP)[1]);
  // dolfin::Function dPhi((*dPNP)[2]);
  // dolfin::Function dU((*dStokes)[0]);
  // dolfin::Function dPressure((*dStokes)[1]);
  (*dPNP)[0].interpolate(zero);
  (*dPNP)[1].interpolate(zero);
  (*dPNP)[2].interpolate(zero);
  (*dStokes)[0].interpolate(zero_vector);
  (*dStokes)[1].interpolate(zero);

  // update pnp_rhs_form
  pnp_rhs_form->dCat = (*dPNP)[0];
  pnp_rhs_form->dAn = (*dPNP)[1];
  pnp_rhs_form->dPhi = (*dPNP)[2];
  pnp_rhs_form->du = (*dStokes)[0];
  dolfin::EigenVector pnp_rhs;
  dolfin::assemble(pnp_rhs, *pnp_rhs_form);
  pnp_bc->apply(pnp_rhs);

  // update stokes_rhs_form
  stokes_rhs_form->dCat = (*dPNP)[0];
  stokes_rhs_form->dAn = (*dPNP)[1];
  stokes_rhs_form->dPhi = (*dPNP)[2];
  stokes_rhs_form->du = (*dStokes)[0];
  stokes_rhs_form->dPress = (*dStokes)[1];
  dolfin::EigenVector stokes_rhs;
  dolfin::assemble(stokes_rhs, *stokes_rhs_form);
  stokes_bc->apply(stokes_rhs);

  // compute initial residual
  double pnp_res = pnp_rhs.norm("l2");
  double stokes_res = stokes_rhs.norm("l2");
  double initial_residual = pnp_res * pnp_res + stokes_res * stokes_res;
  double relative_residual = 1.0;

  // initialize FASP arrays
  dvector pnp_rhs_fasp, pnp_soln_fasp;
  dvector stokes_rhs_fasp, stokes_soln_fasp;
  fasp_dvec_alloc(pnp_rhs.size(), &pnp_soln_fasp);
  fasp_dvec_alloc(stokes_rhs.size(), &stokes_soln_fasp);
  fasp_dvec_set(pnp_rhs_fasp.row, &pnp_soln_fasp, 0.0);
  fasp_dvec_set(stokes_rhs_fasp.row, &stokes_soln_fasp, 0.0);
  std::vector<double> pnp_value_vector;
  std::vector<double> stokes_value_vector;
  pnp_value_vector.reserve(pnp_rhs_fasp.row);
  stokes_value_vector.reserve(stokes_rhs_fasp.row);


  // Block Gauss-Seidel loop
  INT pnp_status = FASP_SUCCESS;
  INT stokes_status = FASP_SUCCESS;
  unsigned int index, iteration_count = 0;
  while (iteration_count++ < max_bgs_it && relative_residual < relative_residual_tol) {

    // solve for pnp update
    EigenVector_to_dvector(&pnp_rhs, &pnp_rhs_fasp);
    pnp_status = fasp_solver_dbsr_krylov_amg (
      A_pnp,
      &pnp_rhs_fasp,
      &pnp_soln_fasp,
      pnp_itpar,
      pnp_amgpar
    );
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
    pnp_rhs_form->dCat = (*dPNP)[0];
    pnp_rhs_form->dAn = (*dPNP)[1];
    pnp_rhs_form->dPhi = (*dPNP)[2];
    stokes_rhs_form->dCat = (*dPNP)[0];
    stokes_rhs_form->dAn = (*dPNP)[1];
    stokes_rhs_form->dPhi = (*dPNP)[2];
    assemble(stokes_rhs, *stokes_rhs_form);
    stokes_bc->apply(stokes_rhs);



    // solve for stokes update and convert to functions
    EigenVector_to_dvector(&stokes_rhs, &stokes_rhs_fasp);
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
    for (index = 0; index < stokes_soln_fasp.row; index++) {
      stokes_value_vector[index] = stokes_soln_fasp.val[index];
    }
    (*dStokes).vector()->set_local(stokes_value_vector);

    // update pnp_rhs_form & stokes_rhs_form with stokes update
    pnp_rhs_form->du = (*dStokes)[0];
    stokes_rhs_form->du = (*dStokes)[0];
    stokes_rhs_form->dPress = (*dStokes)[1];
    assemble(pnp_rhs, *pnp_rhs_form);
    pnp_bc->apply(pnp_rhs);
    assemble(stokes_rhs, *stokes_rhs_form);
    stokes_bc->apply(stokes_rhs);

		fasp_dvec_free(&pnp_soln_fasp);
		fasp_dvec_free(&stokes_soln_fasp);

    // update relative residual
    pnp_res = pnp_rhs.norm("l2");
    stokes_res = stokes_rhs.norm("l2");
    relative_residual = (pnp_res * pnp_res + stokes_res * stokes_res) / initial_residual;
  }

  return FASP_SUCCESS;
};

#endif
