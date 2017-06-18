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
#include "linear_pnp_ns.h"

using namespace std;

//--------------------------------------
Linear_PNP_NS::Linear_PNP_NS (
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
) : PDE (
  mesh,
  function_space,
  functions_space,
  bilinear_form,
  linear_form,
  coefficients,
  sources,
  variables
) {


  // g_space.reset(
  //   new vector_linear_pnp_ns_forms::CoefficientSpace_penalty1(mesh)
  // );


  _itsolver = itsolver;
  _pnpitsolver = pnpitsolver;
  _pnpamg = pnpamg;
  _nsitsolver = nsitsolver;
  _nsamg = nsamg;

   _fasp_block_matrix.brow = 2;
   _fasp_block_matrix.bcol = 2;
   _fasp_block_matrix.blocks = (dCSRmat **)calloc(4, sizeof(dCSRmat *));
   fasp_mem_check((void *)_fasp_block_matrix.blocks, "block matrix:cannot allocate memory!\n", ERROR_ALLOC_MEM);
   for (int i=0; i<4 ;i++) {
       _fasp_block_matrix.blocks[i] = (dCSRmat *)fasp_mem_calloc(1, sizeof(dCSRmat));
   }
}
//--------------------------------------
Linear_PNP_NS::~Linear_PNP_NS () {}
//--------------------------------------
void Linear_PNP_NS::get_dofs_fasp(std::vector<std::size_t> pnp_dimensions,std::vector<std::size_t> ns_dimensions){

  int i,d,k,count;
  // dof index for PNP
  d=0;
  for (i=0;i<pnp_dimensions.size();i++) { d+=_dof_map[pnp_dimensions[i]].size(); }
  fasp_ivec_alloc(d, &_pnp_dofs);
  // dof index for NS
  d=0;
  for (i=0;i<ns_dimensions.size();i++) { d+=_dof_map[ns_dimensions[i]].size(); }
  fasp_ivec_alloc(d, &_stokes_dofs);


  // PNP
  d = pnp_dimensions.size();
  for(k=0;k<pnp_dimensions.size();k++)
  {
    for (i=0; i<_dof_map[pnp_dimensions[k]].size(); i++)
        { _pnp_dofs.val[d*i+k] = _dof_map[pnp_dimensions[k]][i]; }
  }


  // form dof index for Navier-Stokes
  // Velocity
  fasp_ivec_alloc(_dof_map[ns_dimensions[0]].size(), &_velocity_dofs);
  count = 0;
  for (i=0; i<_dof_map[ns_dimensions[0]].size(); i++)
  {
        _stokes_dofs.val[count] = _dof_map[ns_dimensions[0]][i];
        _velocity_dofs.val[i] = _dof_map[ns_dimensions[0]][i];
        count++;
  }

  // Potential
  fasp_ivec_alloc(_dof_map[ns_dimensions[1]].size(), &_pressure_dofs);
  for (i=0; i<_dof_map[ns_dimensions[1]].size(); i++)
  {
      _stokes_dofs.val[count] = _dof_map[ns_dimensions[1]][i];
      _pressure_dofs.val[i] = _dof_map[ns_dimensions[1]][i];
      count++;
  }


}

//--------------------------------------
void Linear_PNP_NS::EigenVector_to_dvector_block (
  std::shared_ptr<const dolfin::EigenVector> eigen_vector,
  dvector* vector
) {
  int i;
  int row = (int) eigen_vector->size();
  double * val = (double*) eigen_vector->data();
  if (row < 1) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenVector_to_dvector");
  }
  // if( _fasp_vector.row > 0) fasp_dvec_free(&_fasp_vector);
  _fasp_vector.row = row;
  fasp_dvec_alloc(row, &_fasp_vector);
  for (i=0; i<_pnp_dofs.row; i++)
      _fasp_vector.val[i] = val[_pnp_dofs.val[i]];
  for (i=0; i<_stokes_dofs.row; i++)
      _fasp_vector.val[_pnp_dofs.row + i] = val[_stokes_dofs.val[i]];

}
//--------------------------------------
void Linear_PNP_NS::setup_fasp_linear_algebra () {
  Linear_PNP_NS::setup_linear_algebra();

  // if (_use_eafe) {
  //   Linear_PNP_NS::apply_eafe();
  //   for (std::size_t i = 0; i < _dirichletBC.size(); i++) {
  //     _dirichletBC[i]->apply(*_eigen_matrix);
  //   }
  // }

  std::size_t dimension = Linear_PNP_NS::get_solution_dimension();
  EigenMatrix_to_dCSRmat(_eigen_matrix, &_fasp_matrix);

  fasp_dcsr_getblk(&_fasp_matrix, _pnp_dofs.val,  _pnp_dofs.val,    _pnp_dofs.row,
      _pnp_dofs.row,    _fasp_block_matrix.blocks[0]);
  fasp_dcsr_getblk(&_fasp_matrix, _pnp_dofs.val,  _stokes_dofs.val, _pnp_dofs.row,
       _stokes_dofs.row, _fasp_block_matrix.blocks[1]);
  fasp_dcsr_getblk(&_fasp_matrix, _stokes_dofs.val, _pnp_dofs.val,    _stokes_dofs.row,
     _pnp_dofs.row,    _fasp_block_matrix.blocks[2]);
  fasp_dcsr_getblk(&_fasp_matrix, _stokes_dofs.val, _stokes_dofs.val, _stokes_dofs.row,
     _stokes_dofs.row, _fasp_block_matrix.blocks[3]);

  if (_faps_soln_unallocated) {
    fasp_dvec_alloc(_eigen_vector->size(), &_fasp_soln);
    _faps_soln_unallocated = false;
  }
  Linear_PNP_NS::EigenVector_to_dvector_block(_eigen_vector, &_fasp_vector);

  fasp_dvec_set(_fasp_vector.row, &_fasp_soln, 0.0);

  // fasp_dcoo_write("Matrix.txt",&_fasp_matrix);
  // fasp_dcoo_write("Matrix11.txt",_fasp_block_matrix.blocks[0]);
  // fasp_dcoo_write("Matrix12.txt",_fasp_block_matrix.blocks[1]);
  // fasp_dcoo_write("Matrix21.txt",_fasp_block_matrix.blocks[2]);
  // fasp_dcoo_write("Matrix22.txt",_fasp_block_matrix.blocks[3]);
  // fasp_ivec_write("pnp_dofs.txt",&(_pnp_dofs));
  // fasp_ivec_write("stokes_dofs.txt",&(_stokes_dofs));
  // fasp_dvec_write("RHS.txt",&(_fasp_vector));

}
//--------------------------------------
std::vector<dolfin::Function> Linear_PNP_NS::fasp_solve () {
  Linear_PNP_NS::setup_fasp_linear_algebra();
  std::vector<dolfin::Function> solutions(Linear_PNP_NS::get_solutions());

  printf("Solving linear system using FASP solver...\n"); fflush(stdout);
  INT status = fasp_solver_bdcsr_krylov_pnp_stokes(
    &_fasp_block_matrix,
    &_fasp_vector,
    &_fasp_soln,
    &_itsolver,
    &_pnpitsolver,
    &_pnpamg,
    &_nsitsolver,
    &_nsamg,
    _velocity_dofs.row,
    _pressure_dofs.row);

  if (status < 0) {
    printf("\n### WARNING: FASP solver failed! Exit status = %d.\n", status);
    fflush(stdout);
  }
  else {
    printf("Successfully solved the linear system\n");
    fflush(stdout);
  }
    dolfin::EigenVector solution_vector(_eigen_vector->mpi_comm(),_eigen_vector->size());
    double* array = solution_vector.data();

    std::size_t i;
    for (i=0; i<_pnp_dofs.row; i++)
        array[_pnp_dofs.val[i]] = _fasp_soln.val[i];
    for (i=0; i<_stokes_dofs.row;i++)
        array[_stokes_dofs.val[i]] = _fasp_soln.val[_pnp_dofs.row + i] ;

    dolfin::Function update (
      Linear_PNP_NS::_convert_EigenVector_to_Function(solution_vector)
    );

    // This need to change, it copies the same vector 3 times.
    auto dCation = std::make_shared<dolfin::Function>(update[0]);
    auto dAnion = std::make_shared<dolfin::Function>(update[1]);
    auto dPotential = std::make_shared<dolfin::Function>(update[2]);

    dolfin::Function dU = update[3];
    dolfin::Function dPressure = update[4];

    std::vector<std::shared_ptr<const dolfin::Function>> vvv = {dCation,dAnion,dPotential};
    auto update_pnp  = std::make_shared<dolfin::Function>(_functions_space[0]);
    assign(update_pnp , vvv);

    *(solutions[0].vector()) += *(update_pnp->vector());
    *(solutions[1].vector()) += *(dU.vector());
    *(solutions[2].vector()) += *(dPressure.vector());

  Linear_PNP_NS::set_solutions(solutions);

  return solutions;
}
//--------------------------------------
dolfin::EigenVector Linear_PNP_NS::fasp_test_solver (
  const dolfin::EigenVector& target_vector
) {
  Linear_PNP_NS::setup_fasp_linear_algebra();

  printf("Compute RHS...\n"); fflush(stdout);
  std::shared_ptr<dolfin::EigenVector> rhs_vector;
  rhs_vector.reset( new dolfin::EigenVector(target_vector.mpi_comm(),target_vector.size()) );


  _eigen_matrix->mult(target_vector, *rhs_vector);
  EigenVector_to_dvector(rhs_vector, &_fasp_vector);

  dolfin::Function solution(Linear_PNP_NS::get_solution());

  printf("Solving linear system using FASP solver...\n"); fflush(stdout);
  INT status = fasp_solver_bdcsr_krylov_pnp_stokes(
    &_fasp_block_matrix,
    &_fasp_vector,
    &_fasp_soln,
    &_itsolver,
    &_pnpitsolver,
    &_pnpamg,
    &_nsitsolver,
    &_nsamg,
    _velocity_dofs.row,
    _pressure_dofs.row);

  if (status < 0) {
    printf("\n### WARNING: FASP solver failed! Exit status = %d.\n", status);
    fflush(stdout);

    return target_vector;
  }

  printf("Successfully solved the linear system\n");
  fflush(stdout);

  dolfin::EigenVector solution_vector(_eigen_vector->mpi_comm(),_eigen_vector->size());
  double* array = solution_vector.data();
  for (std::size_t i = 0; i < _fasp_soln.row; ++i) {
    array[i] = _fasp_soln.val[i];
  }

  return solution_vector;
}
//--------------------------------------
void Linear_PNP_NS::free_fasp () {
  // fasp_dcsr_free(&_fasp_matrix);
  fasp_bdcsr_free(&_fasp_block_matrix);
  fasp_dvec_free(&_fasp_vector);
  fasp_dvec_free(&_fasp_soln);
}
//--------------------------------------

//--------------------------------------
void Linear_PNP_NS::init_BC (double Lx,double Ly,double Lz) {
  std::vector<std::size_t> v1 = {0};
  std::vector<double> v2 = {-Lx/2.0};
  std::vector<double> v3 = {Lx/2.0};
  auto BCdomain = std::make_shared<Dirichlet_Subdomain>(v1,v2,v3,1E-5);
  // Dirichlet_Subdomain BCdomain({component},{-5.0},{5.0},1E-5);

  std::vector<std::size_t> v4 = {0,1,2};
  std::vector<double> v5 = {-Lx/2.0,-Ly/20.0,-Lz/20.0};
  std::vector<double> v6 = {-Lx/2.0,-Ly/20.0,-Lz/20.0};
  auto BCdomainxyz = std::make_shared<Dirichlet_Subdomain>(v4,v5,v6,1E-5);

  auto zero=std::make_shared<dolfin::Constant>(0.0);
  auto zero_vec=std::make_shared<dolfin::Constant>(0.0, 0.0, 0.0);

  auto BC1 = std::make_shared<dolfin::DirichletBC>(_function_space->sub(0),zero,BCdomain);
  auto BC2 = std::make_shared<dolfin::DirichletBC>(_function_space->sub(1),zero,BCdomain);
  auto BC3 = std::make_shared<dolfin::DirichletBC>(_function_space->sub(2),zero,BCdomain);
  auto BC4 = std::make_shared<dolfin::DirichletBC>(_function_space->sub(3),zero_vec,BCdomainxyz);
  _dirichletBC.push_back(BC1);
  _dirichletBC.push_back(BC2);
  _dirichletBC.push_back(BC3);
  _dirichletBC.push_back(BC4);


  std:shared_ptr<const dolfin::Mesh> mesh = _function_space->mesh();
  auto sub_domains = std::make_shared<dolfin::MeshFunction<std::size_t>> \
      (mesh, mesh->topology().dim() - 1);
  sub_domains->set_all(0);
  sub_domains->set_value(0, 1);
  auto BC5 = std::make_shared<dolfin::DirichletBC>(_function_space->sub(4),zero,sub_domains,1);
  _dirichletBC.push_back(BC5);
}
//--------------------------------------


//--------------------------------------
void Linear_PNP_NS::init_measure (std::shared_ptr<const dolfin::Mesh> mesh,
  double Lx, double Ly, double Lz) {
  auto markers = std::make_shared<dolfin::FacetFunction<std::size_t>>(mesh, 1);
  markers->set_all(0);

  // X Boundaries
  std::vector<std::size_t> v1x = {0};
  std::vector<double> v2x = {-Lx/2.0};
  std::vector<double> v3x = {Lx/2.0};
  Dirichlet_Subdomain BCdomain_x(v1x,v2x,v3x,1E-5);
  BCdomain_x.mark(*markers,1);

  // Y and X Boundaries
  // std::vector<std::size_t> v1yz = {1,2};
  // std::vector<double> v2yz = {-Ly/2.0,-Lz/2.0};
  // std::vector<double> v3yz = { Ly/2.0,Lz/2.0};
  // Dirichlet_Subdomain BCdomain_yz(v1yz,v2yz,v3yz,1E-5);
  // BCdomain_yz.mark(*markers,2);

  _linear_form->set_exterior_facet_domains(markers);
}
//--------------------------------------
