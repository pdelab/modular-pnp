#ifndef __NEWTON_STATUS_H
#define __NEWTON_STATUS_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>

class Newton_Status {
  public:

    /// Constructor
    Newton_Status (
      const std::size_t max_iterations_in,
      const double initial_residual_in,
      const double rel_residual_tol_in,
      const double max_residual_tol_in
    );

    /// Destructor
    virtual ~Newton_Status ();

    /// update status
    void update_iteration ();

    void update_residuals (
      const double residual_in,
      const double max_residual_in
    );

    void update_max_residual (
      const double max_residual_in
    );

    void update_rel_residual (
      const double residual_in
    );

    /// check status
    bool needs_to_iterate ();
    bool converged ();
    void print_status ();

    /// compare solutions
    // void update_solution ();
    // void update_residual_vector ();
    // dolfin::Function damp_update ();

    std::size_t max_iterations;
    double initial_residual;
    double residual;
    double rel_residual_tol;
    double max_residual_tol;

    std::size_t iteration = 0;
    double max_residual;
    double relative_residual = 1.0;

  private:
    // std::shared_ptr<dolfin::Function> _solution;
    // std::shared_ptr<dolfin::GenericVector> _residual_vector;
};

#endif
