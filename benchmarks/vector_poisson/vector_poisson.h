#ifndef __VECTOR_POISSON_H
#define __VECTOR_POISSON_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "domain.h"
#include "dirichlet.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "vector_poisson_forms.h"

class Vector_Poisson {
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
    Vector_Poisson (
      const std::shared_ptr<const dolfin::Mesh> mesh,
      const domain_param &domain,
      const std::map<std::string, std::vector<double>> coefficients,
      const itsolver_param &itsolver,
      const AMG_param &amg
    );

    /// Destructor
    virtual ~Vector_Poisson ();

    /// Update the mesh
    void update_mesh (
      const std::shared_ptr<const dolfin::Mesh>
    );

    /// Return mesh from the Poisson object
    dolfin::Mesh get_mesh ();

    /// Set quasi-Newton option to true
    void use_quasi_newton ();

    /// Set quasi-Newton option to false
    void use_exact_newton ();

    /// Print coefficient names to console
    void print_coefficients ();

    /// Remove DoFs for Dirichlet boundary condition
    void remove_Dirichlet_dof (
      std::vector<std::size_t> coordinate
    );

    /// Set Dirichlet Boundary condition
    void set_DirichletBC (
      std::vector<std::size_t> component,
      std::vector<std::vector<double>> boundary
    );

    /// Return the DirichletBC SubDomain
    std::vector<std::shared_ptr<dolfin::SubDomain>> get_Dirichlet_SubDomain();

    /// Set the solution to a constant value
    void set_solution (
      double value
    );

    /// Set the solution to a vector of constant values
    void set_solution (
      std::vector<double> value
    );

    /// Set the solution to interpolate an expression
    void set_solution (
      std::vector<Linear_Function> expression
    );

    /// Get the current solution
    dolfin::Function get_solution ();

    /// Solve the problem using dolfin
    dolfin::Function dolfin_solve ();

    /// Update solution given an update function
    void update_solution(
      dolfin::Function solution,
      const dolfin::Function& update
    );


    /// Define analytic functions from read-in files
    ///
    /// *Arguments*
    ///  coeff (_std::vector<double>_)
    ///    Parameters describing the PDE
    ///    May contain script defining coefficients
    void set_coefficients (
      std::map<std::string, std::vector<double>> values
    );

  private:
    /// Mesh
    std::shared_ptr<dolfin::Mesh> _mesh;
    std::vector<double> _mesh_max, _mesh_min;
    std::size_t _mesh_dim;
    double _mesh_epsilon;

    /// Function Space
    std::shared_ptr<dolfin::FunctionSpace> _function_space;

    /// Forms
    std::shared_ptr<vector_poisson_forms::Form_a> _bilinear_form;
    std::shared_ptr<vector_poisson_forms::Form_L> _linear_form;

    // Current solution
    std::shared_ptr<dolfin::Function> _solution_function;
    std::size_t _get_solution_dimension();

    /// Coefficients
    std::map<std::string, std::shared_ptr<const dolfin::Constant>> _bilinear_coefficient;
    std::map<std::string, std::shared_ptr<const dolfin::Constant>> _linear_coefficient;
    void _construct_coefficients ();

    // /// Dirichlet boundary conditions
    std::vector<std::shared_ptr<dolfin::DirichletBC>> _dirichletBC;
    std::vector<std::shared_ptr<dolfin::SubDomain>> _dirichlet_SubDomain;

    // /// quasi-Newton flag
    bool _quasi_newton;

    // /// Linear solver
    itsolver_param _itsolver;
    AMG_param _amg;
};

#endif
