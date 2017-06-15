#ifndef __LINEAR_PNP_NS_H
#define __LINEAR_PNP_NS_H

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
  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
}

#include "vector_linear_pnp_ns_forms.h"

class Linear_PNP_NS : public PDE {
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
    Linear_PNP_NS (
      const std::shared_ptr<const dolfin::Mesh> mesh,
      const std::shared_ptr<dolfin::FunctionSpace> function_space,
      const std::vector<std::shared_ptr<dolfin::FunctionSpace>> functions_space,
      const std::shared_ptr<dolfin::Form> bilinear_form,
      const std::shared_ptr<dolfin::Form> linear_form,
      const std::map<std::string, std::vector<double>> coefficients,
      const std::map<std::string, std::vector<double>> sources,
      const itsolver_param &itsolver,
      const itsolver_param &pnpitsolver,
      const AMG_param &pnpamg,
      const itsolver_ns_param &nsitsolver,
      const AMG_ns_param &nsamg,
      const std::vector<std::string> variables
    );

    /// Destructor
    virtual ~Linear_PNP_NS ();

    /// FASP interface
    void setup_fasp_linear_algebra ();

    std::vector<dolfin::Function> fasp_solve ();

    dolfin::EigenVector fasp_test_solver (
      const dolfin::EigenVector& target_vector
    );

    void free_fasp ();

    void get_dofs_fasp(
      std::vector<std::size_t> pnp_dimensions,
      std::vector<std::size_t> ns_dimensions);

    void EigenVector_to_dvector_block (
      std::shared_ptr<const dolfin::EigenVector> eigen_vector,
      dvector* vector);

    void apply_eafe ();
    void use_eafe ();
    void no_eafe ();
    void init_BC (double Lx,double Ly,double Lz);

    std::vector<std::shared_ptr<dolfin::Function>> split_mixed_function (
      std::shared_ptr<const dolfin::Function> mixed_function
    );

    // dolfin::Function get_total_charge ();


    std::shared_ptr<dolfin::FunctionSpace> diffusivity_space;
    std::shared_ptr<dolfin::FunctionSpace> valency_space;
    std::shared_ptr<dolfin::FunctionSpace> permittivity_space;
    std::shared_ptr<dolfin::FunctionSpace> fixed_charge_space;
    std::shared_ptr<dolfin::FunctionSpace> phib_space;
    std::shared_ptr<dolfin::FunctionSpace> ub_space;

    ivector _velocity_dofs;
    ivector _pressure_dofs;
    ivector _pnp_dofs;
    ivector _stokes_dofs;

  private:
    // FASP
    // ivector _cation_dofs;
    // ivector _anion_dofs;
    // ivector _potential_dofs;

    itsolver_param _itsolver;
    itsolver_param _pnpitsolver;
    itsolver_ns_param _nsitsolver;
    AMG_param _pnpamg;
    AMG_ns_param _nsamg;
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

static double rc = 0.4;

class SphereSubDomain : public dolfin::SubDomain
{
public:
    bool inside(const dolfin::Array<double>& x, bool on_boundary) const
    {
      return (on_boundary &&
        (std::pow(x[0]-0.0,2) +
        std::pow(x[1]-0.0,2) +
        std::pow(x[2]-0.0,2) < std::pow(rc,2)+1E-5) );
    }
};

class PhibExpression : public dolfin::Expression {
  public:
    PhibExpression(double Eps) : dolfin::Expression(),K(std::sqrt(2.0/Eps))  {};
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      double r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])-rc;
      double g =  std::exp(0.0)*( std::exp(1.0/2.0) - 1.0 )/( std::exp(1.0/2.0) + 1.0 );
      values[0] = 2.0*std::log( (1.0-g*std::exp(-r*K)) / (1.0+g*std::exp(-r*K)) );
    }
  private:
    double K;
};

class VelExpression : public dolfin::Expression {
public:
  VelExpression() : dolfin::Expression(3) {}
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
  values[0] = -x[1];
  values[1] = x[0];
  values[2] = 0.0;
  };
};

class ExactExpression : public dolfin::Expression {
  public:
    ExactExpression(double Eps) : dolfin::Expression(3),K(std::sqrt(2.0/Eps)) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
      double r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])-rc;
      double g =  std::exp(0.0)*( std::exp(1.0/2.0) - 1.0 )/( std::exp(1.0/2.0) + 1.0 );
      values[0] = 2.0*std::log( (1.0-g*std::exp(-r*K)) / (1.0+g*std::exp(-r*K)) );
      values[1] = -values[0];
      values[2] = values[0];
    }
  private:
    double K;
};

#endif
