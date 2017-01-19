#ifndef __LINEAR_PNP_H
#define __LINEAR_PNP_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "pde.h"
#include "domain.h"
#include "dirichlet.h"
#include "EAFE.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_linear_pnp_forms.h"

class Linear_PNP : public PDE {
  public:

    /// Create a PNP problem equipped with necessary
    /// methods for defining, updating, and solving
    /// the specific PNP problem.
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  domain (_domain_param_)
    ///    Parameters for domain dimensions and BCs
    ///  itsolver (_itsolver_param_)
    ///    Parameters for iterative linear solver
    ///  amg (_AMG_param_)
    ///    Parameters for AMG linear solver
    Linear_PNP (
      const std::shared_ptr<const dolfin::Mesh> mesh,
      const std::shared_ptr<dolfin::FunctionSpace> function_space,
      const std::shared_ptr<dolfin::Form> bilinear_form,
      const std::shared_ptr<dolfin::Form> linear_form,
      const std::map<std::string, std::vector<double>> coefficients,
      const std::map<std::string, std::vector<double>> sources,
      const itsolver_param &pnpitsolver,
      const AMG_param &pnpamg
      const itsolver_param &nsitsolver,
      const AMG_param &nsamg,
      const std::vector<std::string> variables
    );

    /// Destructor
    virtual ~Linear_PNP ();

    /// FASP interface
    void setup_fasp_linear_algebra ();

    dolfin::Function fasp_solve ();

    dolfin::EigenVector fasp_test_solver (
      const dolfin::EigenVector& target_vector
    );

    void free_fasp ();
    void get_dofs_fasp();

    void apply_eafe ();
    void use_eafe ();
    void no_eafe ();

    std::vector<std::shared_ptr<dolfin::Function>> split_mixed_function (
      std::shared_ptr<const dolfin::Function> mixed_function
    );

    dolfin::Function get_total_charge ();


    std::shared_ptr<dolfin::FunctionSpace> diffusivity_space;
    std::shared_ptr<dolfin::FunctionSpace> valency_space;
    std::shared_ptr<dolfin::FunctionSpace> permittivity_space;
    std::shared_ptr<dolfin::FunctionSpace> fixed_charge_space;

  private:
    // FASP
    // ivector _cation_dofs;
    // ivector _anion_dofs;
    // ivector _potential_dofs;
    // ivector _velocity_dofs;
    // ivector _pressure_dofs;
    ivector _pnp_dofs;
    ivector _stokes_dofs;
    itsolver_param _pnpitsolver, _nsitsolver;
    AMG_param _pnpamg, _nsamg;
    dCSRmat _fasp_matrix;
    block_dCSRmat _fasp_block_matrix;
    dvector _fasp_vector;
    dvector _fasp_soln;
    bool _faps_soln_unallocated = true;

    // EAFE
    bool _use_eafe = false;
    bool _eafe_uninitialized = true;
    std::shared_ptr<EAFE::BilinearForm> _eafe_bilinear_form;
    std::shared_ptr<dolfin::FunctionSpace> _eafe_function_space;

    std::vector<std::shared_ptr<dolfin::Function>> _split_diffusivity;
    std::vector<double> _valency_double;

    std::shared_ptr<dolfin::Function> eafe_beta, eafe_eta;
    std::shared_ptr<dolfin::EigenMatrix> _eafe_matrix;

};

#endif
