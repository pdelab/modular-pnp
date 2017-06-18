#ifndef __MESH_REFINER_H
#define __MESH_REFINER_H

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
#include "poisson_cell_marker.h"

class Mesh_Refiner {
  public:

    /// constructor
    Mesh_Refiner (
      const std::shared_ptr<const dolfin::Mesh> initial_mesh,
      const std::size_t max_elements_in,
      const std::size_t max_refine_depth_in,
      const double entropy_per_cell
    );

    /// Destructor
    virtual ~Mesh_Refiner ();

    /// Return mesh from the Poisson object
    std::shared_ptr<const dolfin::Mesh> get_mesh ();

    /// refine mesh recursively
    std::shared_ptr<const dolfin::Mesh> multilevel_refinement (
      std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_weight_vector
    );

    std::shared_ptr<const dolfin::Mesh> recursive_refinement (
      std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_weight_vector,
      double entropy_tolerance,
      std::size_t depth
    );

    /// mark for refinement
    std::size_t mark_for_refinement (
      std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_weight_vector,
      double entropy_tolerance
    );

    /// refine to get close to target size
    std::size_t mark_for_refinement_with_target_size (
      std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector,
      std::size_t target_size
    );

    /// compute interpolation error of entropy function
    dolfin::EigenVector compute_entropy_error_vector (
      std::vector<std::shared_ptr<const dolfin::Function>> diffusivity_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_potential_vector,
      std::vector<std::shared_ptr<const dolfin::Function>> entropy_log_weight_vector
    );

    /// mesh refinement
    std::shared_ptr<const dolfin::Mesh> refine_mesh ();

    std::shared_ptr<const dolfin::Mesh> refine_uniformly ();

    void mass_lumping_solver (
      std::shared_ptr<dolfin::EigenMatrix> A,
      std::shared_ptr<dolfin::EigenVector> b,
      std::shared_ptr<dolfin::Function> solution
    );

    dolfin::Function as_function(
      std::shared_ptr<dolfin::FunctionSpace> function_space,
      dolfin::EigenVector vec
    );

    /// iteration count
    std::size_t iteration;

    /// maximum mesh size
    std::size_t max_elements;

    /// solve flags
    bool needs_to_solve;
    bool needs_refinement;

  private:
    std::size_t max_refine_depth;
    double entropy_tolerance_per_cell;
    std::shared_ptr<const dolfin::Mesh> _mesh;
    std::shared_ptr<const L2Error::Functional> _l2_form;
    std::shared_ptr<const SemiH1error::Functional> _semi_h1_form;
    std::shared_ptr<dolfin::MeshFunction<bool>> _cell_marker;
};

#endif
