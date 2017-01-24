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
}

#include "vector_linear_pnp_ns_forms.h"
#include "linear_pnp.h"

using namespace std;

//--------------------------------------
Linear_PNP_NS::Linear_PNP_NS (
  const std::shared_ptr<const dolfin::Mesh> mesh,
  const std::shared_ptr<dolfin::FunctionSpace> function_space,
  const std::shared_ptr<dolfin::Form> bilinear_form,
  const std::shared_ptr<dolfin::Form> linear_form,
  const std::map<std::string, std::vector<double>> coefficients,
  const std::map<std::string, std::vector<double>> sources,
  const itsolver_param &pnpitsolver,
  const AMG_param &pnpamg
  const itsolver_param &nsitsolver,
  const AMG_param &nsamg,
  const std::vector<std::string> variables
) : PDE (
  mesh,
  function_space,
  bilinear_form,
  linear_form,
  coefficients,
  sources,
  variables
) {

  diffusivity_space.reset(
    new vector_linear_pnp_ns_forms::CoefficientSpace_diffusivity(mesh)
  );

  valency_space.reset(
    new vector_linear_pnp_ns_forms::CoefficientSpace_valency(mesh)
  );

  fixed_charge_space.reset(
    new vector_linear_pnp_ns_forms::CoefficientSpace_fixed_charge(mesh)
  );

  permittivity_space.reset(
    new vector_linear_pnp_ns_forms::CoefficientSpace_permittivity(mesh)
  );

  penalty1_space.reset(
    new vector_linear_pnp_ns_forms::CoefficientSpace_penalty1(mesh)
  );

  penalty2_space.reset(
    new vector_linear_pnp_ns_forms::CoefficientSpace_penalty2(mesh)
  );

  _functions_space.push_back(new vector_linear_pnp_ns_forms::CoefficientSpace_cc(mesh));
  _functions_space.push_back(new vector_linear_pnp_ns_forms::CoefficientSpace_uu(mesh));
  _functions_space.push_back(new vector_linear_pnp_ns_forms::CoefficientSpace_pp(mesh));

  _pnpitsolver = pnpitsolver;
  _pnpamg = pnpamg;
  _nsitsolver = nsitsolver;
   _nsamg = nsamg;
}
//--------------------------------------
Linear_PNP_NS::~Linear_PNP_NS () {}
//--------------------------------------
void get_dofs_fasp(std::vector<std::size_t> pnp_dimensions,std::vector<std::size_t> ns_dimensions){

  int i,d,k,count;
  // form dof index for PNP
  d=0
  for (i=0;i<pnp_dimensions.size();i++) d+=_dof_map[pnp_dimensions[i]].size();
  fasp_ivec_alloc(d, &_pnp_dofs);
  d=0
  for (i=0;i<ns_dimensions.size();i++) d+=_dof_map[ns_dimensions[i]].size();
  fasp_ivec_alloc(velocity_dofs.row + pressure_dofs.row, &_stokes_dofs);

  d = pnp_dimensions.size()
  for(k=0;k<pnp_dimensions.size(),k++)
  {
    for (i=0; i<_dof_map[pnp_dimensions[k]].size(); i++)
        _pnp_dofs.val[d*i+k] = _dof_map[pnp_dimensions[k]][i];
  }

  // form dof index for Navier-Stokes
  // Velocity
  count = 0;
  for(k=0;k<ns_dimensions.size()-1,k++)
  {
    for (i=0; i<_dof_map[ns_dimensions[k]].size(); i++)
        _stokes_dofs.val[count] = _dof_map[ns_dimensions[k]][i];
        count++;
  }
  // Potential
  k = ns_dimensions.size()-1
  for (i=0; i<_dof_map[ns_dimensions[k]]; i++)
  {
      stokes_dofs.val[count] = _dof_map[ns_dimensions[k]][i];
      count++
  }

}

//--------------------------------------
void PDE::EigenVector_to_dvector_block (
  std::shared_ptr<const dolfin::EigenVector> eigen_vector,
  dvector* vector
) {
  int i;
  int row = (int) eigen_vector->size();
  double * val = (double*) eigen_vector->data()
  if (row < 1) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenVector_to_dvector");
  }

  fasp_dvec_free(&_fasp_vector);
  _fasp_vector->row = row;
  fasp_dvec_alloc(row,_fasp_vector->row = row;);
  for (i=0; i<pnp_dofs.row; i++)
      _fasp_vector->val[i] = val[pnp_dofs.val[i]];
  for (i=0; i<stokes_dofs.row; i++)
      _fasp_vector->val[pnp_dofs.row + i] = val[stokes_dofs.val[i]];

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
  // _fasp_bsr_matrix = fasp_format_dcsr_dbsr(&_fasp_matrix, dimension);
  fasp_bdcsr_free(&_fasp_block_matrix);
   _fasp_block_matrix.brow = 2;
   _fasp_block_matrix.bcol = 2;
   _fasp_block_matrix.blocks = (dCSRmat **)calloc(4, sizeof(dCSRmat *));
  fasp_dcsr_getblk(&_fasp_matrix, pnp_dofs.val,    pnp_dofs.val,    pnp_dofs.row,
      pnp_dofs.row,    _fasp_block_matrix[0]);
  fasp_dcsr_getblk(&_fasp_matrix, pnp_dofs.val,    stokes_dofs.val, pnp_dofs.row,
       stokes_dofs.row, _fasp_block_matrix[1]);
  fasp_dcsr_getblk(&_fasp_matrix, stokes_dofs.val, pnp_dofs.val,    stokes_dofs.row,
     pnp_dofs.row,    _fasp_block_matrix[2]);
  fasp_dcsr_getblk(&_fasp_matrix, stokes_dofs.val, stokes_dofs.val, stokes_dofs.row,
     stokes_dofs.row, _fasp_block_matrix[3]);


  if (_faps_soln_unallocated) {
    fasp_dvec_alloc(_eigen_vector->size(), &_fasp_soln);
    _faps_soln_unallocated = false;
  }
  EigenVector_to_dvector_block(_eigen_vector, &_fasp_vector) // to do;

  fasp_dvec_set(_fasp_vector.row, &_fasp_soln, 0.0);
}
//--------------------------------------
std::vector<dolfin::Function> Linear_PNP_NS::fasp_solve () {
  Linear_PNP_NS::setup_fasp_linear_algebra();
  std::vector<dolfin::Function> solution(Linear_PNP::get_solutions());

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
    velocity_dofs.row,
    pressure_dofs.row);

  if (status < 0) {
    printf("\n### WARNING: FASP solver failed! Exit status = %d.\n", status);
    fflush(stdout);
  }
  else {
    printf("Successfully solved the linear system\n");
    fflush(stdout);

    dolfin::EigenVector solution_vector(_eigen_vector->mpi_comm(),_eigen_vector->size());
    double* array = solution_vector.data();

    std::size_t i
    // for (std::size_t i = 0; i < _fasp_soln.row; ++i) {
    //   array[i] = _fasp_soln.val[i];
    for (i=0; i<_pnp_dofs.row; i++)
        array[_pnp_dofs.val[i]] = _fasp_soln.val[i];
    for (i=0; i<stokes_dofs.row;i++)
        array[_stokes_dofs.val[i]] = _fasp_soln.val[_pnp_dofs.row + i] ;

    dolfin::Function update (
      Linear_PNP::_convert_EigenVector_to_Function(solution_vector)
    );

    // This need to change, it copies the same vector 3 times.
    Function dAnion = update[0];
    Function dCation = update[1];
    Function dPotential = update[2];
    Function dU = update[3];
    Function dPressure = update[4];

    std::vector<std::shared_ptr<const Function>> vvv = {dAnion,dCation,dPotential};
    dolfin::Function update_pnp(_functions_space[0])
    assign(update_pnp , vvv);

    *(solution[0].vector()) += *(update_pnp.vector());
    *(solution[1].vector()) += *(dU.vector());
    *(solution[2].vector()) += *(dPressure.vector());
  }

  Linear_PNP::set_solutions(solutions);

  return solution;
}
//--------------------------------------
dolfin::EigenVector Linear_PNP::fasp_test_solver (
  const dolfin::EigenVector& target_vector
) {
  Linear_PNP::setup_fasp_linear_algebra();

  printf("Compute RHS...\n"); fflush(stdout);
  std::shared_ptr<dolfin::EigenVector> rhs_vector;
  rhs_vector.reset( new dolfin::EigenVector(target_vector.mpi_comm(),target_vector.size()) );


  _eigen_matrix->mult(target_vector, *rhs_vector);
  EigenVector_to_dvector(rhs_vector, &_fasp_vector);

  dolfin::Function solution(Linear_PNP::get_solution());

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
    velocity_dofs.row,
    pressure_dofs.row);

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
void Linear_PNP::free_fasp () {
  // fasp_dcsr_free(&_fasp_matrix);
  fasp_bdcsr_free(&_fasp_block_matrix);
  fasp_dvec_free(&_fasp_vector);
  fasp_dvec_free(&_fasp_soln);
}
//--------------------------------------


//--------------------------------------
void Linear_PNP::use_eafe () {
 _use_eafe = true;
}
//--------------------------------------
void Linear_PNP::no_eafe () {
 _use_eafe = false;
}
//--------------------------------------
void Linear_PNP::apply_eafe () {
  dolfin::Function solution_function(Linear_PNP::get_solution());

  if (_eafe_uninitialized) {

    _eafe_function_space = solution_function[0].function_space()->collapse();
    _eafe_bilinear_form.reset(
      new EAFE::Form_a(_eafe_function_space, _eafe_function_space)
    );
    _eafe_matrix.reset(new dolfin::EigenMatrix());

    std::shared_ptr<dolfin::Function> _diffusivity;
    _diffusivity.reset(new dolfin::Function(diffusivity_space));
    _diffusivity->interpolate(
      *(_bilinear_form->coefficient("diffusivity"))
    );
    _split_diffusivity = Linear_PNP::split_mixed_function(_diffusivity);


    dolfin::Function val_fn(valency_space);
    val_fn.interpolate(
      (*_bilinear_form->coefficient("valency"))
    );

    std::size_t vals = Linear_PNP::get_solution_dimension();
    _valency_double.reserve(vals);
    for (uint val_idx = 0; val_idx < vals; val_idx++) {
      _valency_double[val_idx] = (*(val_fn.vector()))[val_idx];
    }
    _valency_double[0] = 0.0;
  }

  std::shared_ptr<dolfin::Function> solution_ptr;
  solution_ptr.reset( new dolfin::Function(solution_function.function_space()) );
  *solution_ptr = solution_function;

  auto zero= std::make_shared<const dolfin::Constant>(0.0);
  std::vector<std::shared_ptr<dolfin::Function>> solution_vector;
  solution_vector = Linear_PNP::split_mixed_function(solution_ptr);
  std::shared_ptr<dolfin::Function> beta, eta, phi;

  std::size_t eqns = Linear_PNP::get_solution_dimension();
  for (uint eqn_idx = 1; eqn_idx < eqns; eqn_idx++) {
    beta.reset(new dolfin::Function(_eafe_function_space));
    eta.reset(new dolfin::Function(_eafe_function_space));
    phi.reset(new dolfin::Function(_eafe_function_space));

    beta->interpolate( *(solution_vector[eqn_idx]) );
    eta->interpolate( *(solution_vector[eqn_idx]) );
    phi->interpolate( *(solution_vector[0]) );

    if (_valency_double[eqn_idx] != 0.0) {
      *(phi->vector()) *= _valency_double[eqn_idx];
      *beta = *beta + *phi;
    }

    _eafe_bilinear_form->alpha = _split_diffusivity[eqn_idx];
    _eafe_bilinear_form->beta = beta;
    _eafe_bilinear_form->eta = eta;
    _eafe_bilinear_form->gamma = zero;

    dolfin::assemble(*_eafe_matrix, *_eafe_bilinear_form);

    const int* IA = (int*) std::get<0>(_eafe_matrix->data());
    const int* JA = (int*) std::get<1>(_eafe_matrix->data());
    const double* values = (double*) std::get<2>(_eafe_matrix->data());

    const int* global_IA = (int*) std::get<0>(_eigen_matrix->data());
    const int* global_JA = (int*) std::get<1>(_eigen_matrix->data());
    double* global_values = (double*) std::get<2>(_eigen_matrix->data());

    uint row = 0, new_i = 0;
    dolfin::la_index global_row, global_col;
    std::size_t local_size = _eafe_matrix->nnz();
    for (uint i = 0; i < local_size; i++) {

      // convert local row/col to global row/col
      while (IA[row + 1] < i + 1) row++;
      global_row = _dof_map[eqn_idx][row];
      global_col = _dof_map[eqn_idx][JA[i]];

      // find and replace corresponding row/col in global IA/JA
      new_i = global_IA[global_row];
      while (global_JA[new_i] < global_col) new_i++;
      global_values[new_i] = values[i];
      // if (global_JA[new_i] != global_col) {
        // printf("Error : %d != %d\n", global_JA[new_i], global_col); fflush(stdout);
      // }
    }
  }
}
//--------------------------------------
std::vector<std::shared_ptr<dolfin::Function>> Linear_PNP::split_mixed_function (
  std::shared_ptr<const dolfin::Function> mixed_function
) {
  //construct assignment arguments
  std::shared_ptr<const dolfin::FunctionSpace> mixed_space(
    mixed_function->function_space()
  );

  // construct receiving arguments
  std::vector<std::shared_ptr<dolfin::Function>> function_vector;
  std::vector<std::shared_ptr<const dolfin::FunctionSpace>> subspace_vector;

  // initialize vectors
  std::shared_ptr<dolfin::Function> subfunction;
  std::shared_ptr<const dolfin::FunctionSpace> subspace;
  std::size_t num_components = mixed_space->element()->num_sub_elements();

  for (std::size_t c = 0; c < num_components; c++) {
    subspace.reset(
      new const dolfin::FunctionSpace( *( (*mixed_space)[c]->collapse() ) )
    );
    subfunction.reset(new dolfin::Function(subspace));

    subspace_vector.push_back(subspace);
    function_vector.push_back(subfunction);
  }

  // assign
  dolfin::FunctionAssigner function_assigner(subspace_vector, mixed_space);
  function_assigner.assign(function_vector, mixed_function);

  return function_vector;
}

//-------------------------------------
dolfin::Function Linear_PNP::get_total_charge () {
  dolfin::Function total_charge(fixed_charge_space);
  total_charge.interpolate(
    *(_linear_form->coefficient("fixed_charge"))
  );

  dolfin::Function valencies(valency_space);
  valencies.interpolate(
    (*_bilinear_form->coefficient("valency"))
  );

  dolfin::Function solution(Linear_PNP::get_solution());
  std::size_t solution_size = Linear_PNP::get_solution_dimension();
  std::shared_ptr<dolfin::Function> solution_charge;

  double value;
  for (std::size_t charge = 1; charge < solution_size; charge++) {
    double q = (*(valencies.vector()))[charge];
    solution_charge.reset(new dolfin::Function(fixed_charge_space));
    solution_charge->interpolate(solution[charge]);
    for (std::size_t index = 0; index < solution_charge->vector()->size(); index++) {
      value = q * std::exp( (*(solution_charge->vector()))[index] );
      solution_charge->vector()->setitem(index, value);
    }

    total_charge = total_charge + (*solution_charge);
  }

  return total_charge;
}