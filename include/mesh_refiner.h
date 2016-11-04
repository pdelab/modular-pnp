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

class Mesh_Refiner {
  public:

    /// constructor
    Mesh_Refiner (
      const std::shared_ptr<const dolfin::Mesh> initial_mesh
    );

    /// Destructor
    virtual ~Mesh_Refiner ();

    // /// add Dirichlet BCs
    // void add_Dirichlet_BC (
    //   std::vector<std::size_t> fn_component,
    //   std::vector<std::shared_ptr<dolfin::SubDomain>> boundary
    // );

    // /// return Dirichlet BCs
    // std::vector<std::shared_ptr<dolfin::DirichletBC>> get_Dirichlet_BCs ();

    // /// mark surfaces
    // // void add_marked_surfaces (
    // //   std::vector<std::size_t> surface_index,
    // //   std::vector<std::shared_ptr<dolfin::SubDomain>> surface
    // // );

    // /// Update the mesh
    // void mark_mesh (
    //   const std::shared_ptr<const dolfin::Mesh> mesh
    // );

    /// Return mesh from the Poisson object
    std::shared_ptr<const dolfin::Mesh> get_mesh ();

    // /// mark for refinement
    // void mark_for_refinement (
    //   std::vector<std::shared_ptr<const dolfin::Function>> weights,
    //   std::shared_ptr<const dolfin::Function> solution,
    //   std::string norm
    // );

    /// mesh refinement
    std::shared_ptr<const dolfin::Mesh> refine_mesh ();

    std::shared_ptr<const dolfin::Mesh> refine_uniformly ();

    /// iteration count
    std::size_t iteration;

  private:
    std::shared_ptr<const dolfin::Mesh> _mesh;
    std::shared_ptr<const L2Error::Functional> _l2_form;
    std::shared_ptr<const SemiH1error::Functional> _semi_h1_form;

    std::vector<std::shared_ptr<dolfin::DirichletBC>> _dirichletBC;
    std::vector<std::shared_ptr<dolfin::SubDomain>> _dirichlet_SubDomain;

    dolfin::MeshFunction<bool> _cell_marker;
};

#endif
