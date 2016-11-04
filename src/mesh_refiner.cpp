#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "domain.h"
#include "dirichlet.h"
#include "mesh_refiner.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

#include "L2Error.h"
#include "SemiH1error.h"

//--------------------------------
Mesh_Refiner::Mesh_Refiner (
  const std::shared_ptr<const dolfin::Mesh> initial_mesh,
  const std::size_t max_elements_in,
  const std::size_t max_refine_depth_in
) {
  _mesh.reset(new const dolfin::Mesh(*initial_mesh));

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  iteration = 0;
  Mesh_Refiner::needs_to_solve = true;
  Mesh_Refiner::needs_refinement = false;

  Mesh_Refiner::max_elements = max_elements_in;
  Mesh_Refiner::max_refine_depth = max_refine_depth_in;
};
//--------------------------------
Mesh_Refiner::~Mesh_Refiner () {};
//--------------------------------

//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::get_mesh () {
  return _mesh;
};
//--------------------------------
void Mesh_Refiner::mark_for_refinement (
  std::shared_ptr<const dolfin::Function> entropy
) {
  if (Mesh_Refiner::max_refine_depth == 0 || _mesh->num_cells() > Mesh_Refiner::max_elements) {
    printf("Mesh refinement is attempting to over-refine...\n");
    Mesh_Refiner::needs_refinement = false;
    Mesh_Refiner::needs_to_solve = false;
    return;
  }

  _cell_marker.reset(
    new dolfin::MeshFunction<bool>(_mesh, _mesh->topology().dim(), true)
  );

  Mesh_Refiner::needs_refinement = true;
};
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::refine_mesh () {
  auto refined_mesh = adapt(*_mesh, *_cell_marker);
  _mesh = refined_mesh;

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  Mesh_Refiner::needs_refinement = false;
  Mesh_Refiner::needs_to_solve = true;
  return _mesh;
};
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::refine_uniformly () {
  auto refined_mesh = adapt(*_mesh);
  _mesh = refined_mesh;

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  Mesh_Refiner::needs_refinement = false;
  Mesh_Refiner::needs_to_solve = true;
  return _mesh;
};
//--------------------------------
