#ifndef __PROBLEM_H
#define __PROBLEM_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

class Problem {
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
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///  itsolver (_itsolver_param_)
    ///    Parameters for iterative linear solver
    ///  amg (_AMG_param_)
    ///    Parameters for AMG linear solver
    Problem (
      std::shared_ptr<const dolfin::Mesh> mesh,
      domain_param domain,
      coeff_param coeff,
      itsolver_param itsolver,
      AMG_param amg
    );

    /// Destructor
    virtual ~Problem ();

    /// Define analytic functions from read-in files
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  domain (_domain_param_)
    ///    Parameters for domain dimensions and BCs
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///    May contain script defining coefficients
    void set_coefficients (
      std::shared_ptr<const dolfin::Mesh> mesh,
      domain_param domain,
      coeff_param coeff
    );

    /// Define initial guess for the Newton solver
    /// from read-in files
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  domain (_domain_param_)
    ///    Parameters for domain dimensions and BCs
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///
    /// *Returns*
    ///   std::shared_ptr<dolfin::Function>
    ///     Initial guess as a vector-valued function
    std::shared_ptr<dolfin::Function> initial_guess (
      std::shared_ptr<const dolfin::Mesh> mesh,
      domain_param domain,
      coeff_param coeff
    );

    /// Construct bilinear form corresponding to
    /// the linearized nonlinear problem about the iterate
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  iterate (dolfin::Function)
    ///    Iterate where form is linearized
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///
    /// *Returns*
    ///   dolfin::Form
    ///     Bilinear form describing linearized PDE
    dolfin::Form linearized_form (
      std::shared_ptr<const dolfin::Mesh> mesh,
      std::shared_ptr<dolfin::Function> iterate,
      coeff_param coeff
    );

    /// Construct bilinear form corresponding to the PDE
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///
    /// *Returns*
    ///   dolfin::Form
    ///     Bilinear form describing the PDE
    dolfin::Form bilinear_form (
      std::shared_ptr<const dolfin::Mesh> mesh,
      coeff_param coeff
    );

    /// Construct linear form corresponding to the residual
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///  iterate (dolfin::Function)
    ///    Iterate where form is linearized
    ///
    /// *Returns*
    ///   dolfin::Form
    ///     Linear form describing the PDE
    dolfin::Form linear_form (
      std::shared_ptr<const dolfin::Mesh> mesh,
      std::shared_ptr<dolfin::Function> iterate,
      coeff_param coeff
    );

    /// Construct linear form corresponding to the residual
    ///
    /// *Arguments*
    ///  mesh (dolfin::Mesh)
    ///    The mesh
    ///  coeff (_coeff_param_)
    ///    Parameters describing the PDE
    ///
    /// *Returns*
    ///   dolfin::Form
    ///     Linear form describing the PDE
    dolfin::Form linear_form (
      std::shared_ptr<const dolfin::Mesh> mesh,
      coeff_param coeff
    );

  private:
    /// quasi-Newton flag
    bool _quasi_newton = false;

    /// linear algebraic
    dolfin::EigenMatrix _jacobian;
    dolfin::EigenVector _residual;

    /// Dirichlet boundary conditions
    dolfin::DirichletBC _dirichletBC;
    dolfin::SubDomain _dirichletSubDomain;

    /// Linear solver
    itsolver_param* _itsolver;
    amg_param* _amg;
};

#endif
