#ifndef __ERROR_H
#define __ERROR_H


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

#include "L2Error.h"
#include "SemiH1error.h"

class Error {
  public:

    /// Constructor
    Error (
      std::shared_ptr<const dolfin::Function> exact_solution
    );

    /// Destructor
    virtual ~Error ();

    /// Change exact solution
    void update_exact_solution (
      std::shared_ptr<const dolfin::Function> exact_solution
    );

    /// Compute the error and measure
    dolfin::Function compute_error (
      std::shared_ptr<dolfin::Function> computed_solution
    );

    double compute_l2_error (
      std::shared_ptr<dolfin::Function> computed_solution
    );

    double compute_semi_h1_error (
      std::shared_ptr<dolfin::Function> computed_solution
    );

    double compute_h1_error (
      std::shared_ptr<dolfin::Function> computed_solution
    );

  private:
    std::shared_ptr<L2Error::Functional> _l2_form;
    std::shared_ptr<SemiH1error::Functional> _semi_h1_form;

    std::shared_ptr<const dolfin::Function> _exact_solution;
    std::shared_ptr<const dolfin::FunctionSpace> _function_space;

    std::size_t _num_subfunctions;

};

#endif
