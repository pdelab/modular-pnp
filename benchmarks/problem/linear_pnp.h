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
      const itsolver_param &itsolver,
      const AMG_param &amg
    );

    /// Destructor
    virtual ~Linear_PNP ();

    /// FASP interface
    void setup_fasp_linear_algebra ();

    dolfin::Function fasp_solve ();

    void free_fasp ();

    std::shared_ptr<dolfin::FunctionSpace> diffusivity_space;
    std::shared_ptr<dolfin::FunctionSpace> valency_space;
    std::shared_ptr<dolfin::FunctionSpace> permittivity_space;
    std::shared_ptr<dolfin::FunctionSpace> fixed_charge_space;

  private:
    itsolver_param _itsolver;
    AMG_param _amg;
    dCSRmat _fasp_matrix;
    dBSRmat _fasp_bsr_matrix;
    dvector _fasp_vector, _fasp_soln;
};

#endif
