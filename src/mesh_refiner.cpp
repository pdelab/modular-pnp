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
  const std::shared_ptr<const dolfin::Mesh> initial_mesh
) {
  _mesh.reset(new const dolfin::Mesh(*initial_mesh));

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));

  iteration = 0;
};
//--------------------------------
Mesh_Refiner::~Mesh_Refiner () {};
//--------------------------------

// //--------------------------------
// void Mesh_Refiner::add_Dirichlet_BC (
//   std::vector<std::size_t> fn_component,
//   std::vector<std::shared_ptr<dolfin::SubDomain>> boundary
// ) {
//   if (fn_component.size() != boundary.size()) {
//     printf("Incompatible boundary conditions... not applying BCs\n"); fflush(stdout);
//   }

//   std::shared_ptr<dolfin::Constant> zero_constant;
//   zero_constant.reset(new dolfin::Constant(0.0));
//   for (std::size_t bc = 0; bc < boundary.size(); bc++) {
//     _dirichlet_SubDomain.push_back(std::make_shared<dolfin::SubDomain>());
//     _dirichlet_SubDomain.back() = boundary[bc];

//     _dirichletBC.push_back(std::make_shared<dolfin::DirichletBC>(
//       (*_function_space)[bc],
//       zero_constant,
//       _dirichlet_SubDomain.back()
//     ));
//   }
// };
// //--------------------------------
// std::vector<std::shared_ptr<dolfin::DirichletBC>> Mesh_Refiner::get_Dirichlet_BCs () {
//   return _dirichlet_SubDomain;
// };
// //--------------------------------

// //--------------------------------
// // void Mesh_Refiner::add_marked_surfaces (
// //   std::vector<std::size_t> surface_index,
// //   std::vector<std::shared_ptr<dolfin::SubDomain>> surface
// // ) {};
// //--------------------------------

//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::get_mesh () {
  return _mesh;
};
// //--------------------------------
// void Mesh_Refiner::mark_for_refinement (
//   std::vector<std::shared_ptr<const dolfin::Function>> weights,
//   std::shared_ptr<const dolfin::Function> solution,
//   std::string norm
// ) {
//   _cell_marker.reset(new dolfin::MeshFunction<bool>(*_mesh, _mesh->topology().dim(), true));
// };
// //--------------------------------
// std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::refine_mesh () {
//   dolfin::Mesh refined_mesh(_mesh);
//   _mesh.reset(new const dolfin::Mesh(
//     std::make_shared<const dolfin::Mesh>(refine_mesh)
//   ));

//   _l2_form.reset(new L2Error::Functional(_mesh));
//   _semi_h1_form.reset(new SemiH1error::Functional(_mesh));
//   return _mesh;
// };
//--------------------------------
std::shared_ptr<const dolfin::Mesh> Mesh_Refiner::refine_uniformly () {
  auto refined_mesh = adapt(*_mesh);
  _mesh = refined_mesh;

  _l2_form.reset(new L2Error::Functional(_mesh));
  _semi_h1_form.reset(new SemiH1error::Functional(_mesh));
  return _mesh;
};
//--------------------------------
