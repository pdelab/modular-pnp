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
#include "gradient_recovery.h"
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

    /// mark for refinement
    void mark_for_refinement (
      std::shared_ptr<const dolfin::Function> entropy_potential
    );

    /// mesh refinement
    std::shared_ptr<const dolfin::Mesh> refine_mesh ();

    std::shared_ptr<const dolfin::Mesh> refine_uniformly ();

    void mass_lumping_solver (
      std::shared_ptr<dolfin::EigenMatrix> A,
      std::shared_ptr<dolfin::EigenVector> b,
      std::shared_ptr<dolfin::Function> solution
    );

    /// iteration count
    std::size_t iteration;

    /// solve flags
    bool needs_to_solve;
    bool needs_refinement;

  private:
    std::size_t max_elements, max_refine_depth;
    double entropy_tolerance_per_cell;
    std::shared_ptr<const dolfin::Mesh> _mesh;
    std::shared_ptr<const L2Error::Functional> _l2_form;
    std::shared_ptr<const SemiH1error::Functional> _semi_h1_form;
    std::shared_ptr<dolfin::MeshFunction<bool>> _cell_marker;
};

#endif
