#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>
#include <ufc.h>
#include "pde.h"
#include "domain.h"
#include "dirichlet.h"
extern "C" {
  #include "fasp.h"
  #include "fasp_functs.h"
}

using namespace std;

//--------------------------------------
PDE::PDE (
  const std::shared_ptr<const dolfin::Mesh> mesh,
  const std::shared_ptr<dolfin::FunctionSpace> function_space,
  const std::shared_ptr<dolfin::Form> bilinear_form,
  const std::shared_ptr<dolfin::Form> linear_form,
  const std::map<std::string, std::vector<double>> coefficients,
  const std::map<std::string, std::vector<double>> sources
) {
  PDE::update_mesh(mesh);
  _function_space = function_space;
  _bilinear_form = bilinear_form;
  _linear_form = linear_form;

  PDE::get_dofs();
  PDE::set_solution(0.0);

  PDE::set_coefficients(coefficients, sources);

  PDE::use_exact_newton();
}
//--------------------------------------
PDE::~PDE() {}
//--------------------------------------




//--------------------------------------
void PDE::update_mesh (
  const std::shared_ptr<const dolfin::Mesh> mesh
) {
  _mesh.reset(new dolfin::Mesh(*mesh));

  _mesh_dim = _mesh->topology().dim();
  _mesh_epsilon = 0.1 * _mesh->rmin();
  for (std::size_t d = 0; d < _mesh_dim; d++) {
    _mesh_max.push_back(-1E+20);
    _mesh_min.push_back(+1E+20);
  }

  double coord_value = 0.0;
  for (std::size_t vert = 0; vert < _mesh->num_vertices(); ++vert) {
    for (std::size_t d = 0; d < _mesh_dim; d++) {
      coord_value = _mesh->coordinates()[vert * _mesh_dim + d];
      _mesh_max[d] = coord_value > _mesh_max[d] ? coord_value : _mesh_max[d];
      _mesh_min[d] = coord_value < _mesh_min[d] ? coord_value : _mesh_min[d];
    }
  }
}
//--------------------------------------
dolfin::Mesh PDE::get_mesh () {
  return *(_mesh);
}
//--------------------------------------




//--------------------------------------
void PDE::use_quasi_newton () {
  _quasi_newton = true;
}
//--------------------------------------
void PDE::use_exact_newton () {
  _quasi_newton = false;
}
//--------------------------------------



//--------------------------------------
std::size_t PDE::get_solution_dimension() {
  return _function_space->element()->num_sub_elements();
}
//--------------------------------------
void PDE::set_solution (
  double value
) {
  std::shared_ptr<dolfin::Constant> constant_fn;
  _solution_function.reset( new dolfin::Function(_function_space) );
  std::size_t dimension = PDE::get_solution_dimension();

  if (dimension == 0) {
    constant_fn.reset( new dolfin::Constant(value) );
    _solution_function->interpolate(*constant_fn);
  }
  else {
    std::vector<double> values(dimension, value);
    constant_fn.reset( new dolfin::Constant(values) );
    _solution_function->interpolate(*constant_fn);
  }

  _bilinear_form->set_coefficient("uu", (_solution_function));
  _linear_form->set_coefficient("uu", (_solution_function));
}
//--------------------------------------
void PDE::set_solution (
  std::vector<double> value
) {
  _solution_function.reset( new dolfin::Function(_function_space) );
  std::shared_ptr<dolfin::Constant> constant_fn;
  std::size_t dimension = PDE::get_solution_dimension();

  if (dimension == value.size()) {
    constant_fn.reset( new dolfin::Constant(value) );
    _solution_function->interpolate(*constant_fn);
  }
  else {
    printf("Dimension mismatch!!\n");
    printf("\tsetting solution to zeros %lu \n", dimension); fflush(stdout);
    constant_fn.reset( new dolfin::Constant(dimension, 0.0) );
    _solution_function->interpolate(*constant_fn);
  }

  _bilinear_form->set_coefficient("uu", (_solution_function));
  _linear_form->set_coefficient("uu", (_solution_function));
}
//--------------------------------------
void PDE::set_solution (
  const std::vector<Linear_Function::Linear_Function> expression
) {

  std::size_t dimension = PDE::get_solution_dimension();
  if (dimension == expression.size()) {

    std::vector<std::shared_ptr<const dolfin::FunctionSpace>> vector_space;
    std::vector<std::shared_ptr<const dolfin::Function>> vector_expression;

    std::shared_ptr<const dolfin::FunctionSpace> sub_space;
    std::shared_ptr<const dolfin::Function> const_sub_expression;
    std::shared_ptr<dolfin::Function> sub_expression;

    for (std::size_t i = 0; i < dimension; i++) {
      sub_space.reset(
        new const dolfin::FunctionSpace( *((*_function_space)[i]->collapse()) )
      );
      sub_expression.reset( new dolfin::Function(sub_space) );
      sub_expression->interpolate(expression[i]);

      const_sub_expression.reset( new const dolfin::Function(*sub_expression) );
      vector_expression.push_back(const_sub_expression);
      vector_space.push_back(sub_space);
    }
    dolfin::FunctionAssigner function_assigner(_function_space, vector_space);
    function_assigner.assign(_solution_function, vector_expression);
  }
  else {
    printf("Dimension mismatch!!\n");
    printf("\tsetting solution to zeros %lu \n", dimension);
    std::shared_ptr<dolfin::Constant> constant_fn;
    constant_fn.reset( new dolfin::Constant(dimension, 0.0) );
    _solution_function->interpolate(*constant_fn);
  }

  _bilinear_form->set_coefficient("uu", _solution_function);
  _linear_form->set_coefficient("uu", _solution_function);
}
//--------------------------------------
void PDE::set_solution (
  const dolfin::Function& new_solution
) {

  std::size_t dimension = PDE::get_solution_dimension();
  std::size_t new_dim = new_solution.function_space()->element()->num_sub_elements();

  if (dimension == new_dim) {
    _solution_function.reset(new dolfin::Function(_function_space));
    *_solution_function = new_solution;
  }
  else {
    printf("Dimension mismatch!!\n");
    printf("\tsetting solution to zeros %lu \n", dimension);
    std::shared_ptr<dolfin::Constant> constant_fn;
    constant_fn.reset( new dolfin::Constant(dimension, 0.0) );
    _solution_function->interpolate(*constant_fn);
  }

  _bilinear_form->set_coefficient("uu", _solution_function);
  _linear_form->set_coefficient("uu", _solution_function);
}
//--------------------------------------
dolfin::Function PDE::get_solution () {
  return *(_solution_function);
}
//--------------------------------------



//--------------------------------------
void PDE::print_coefficients () {

  printf("The coefficients of the bilinear form are:\n");
  std::map<std::string, std::shared_ptr<const dolfin::Constant>>::iterator bc;
  for (bc = _bilinear_coefficient.begin(); bc != _bilinear_coefficient.end(); ++bc) {
    printf("\t%s\n", bc->first.c_str());
  }

  printf("The coefficients of the linear form are:\n");
  std::map<std::string, std::shared_ptr<const dolfin::Constant>>::iterator lc;
  for (lc = _linear_coefficient.begin(); lc != _linear_coefficient.end(); ++lc) {
    printf("\t%s\n", lc->first.c_str());
  }
}
//--------------------------------------
void PDE::set_coefficients (
  std::map<std::string, std::vector<double>> coefficients,
  std::map<std::string, std::vector<double>> sources
) {
  std::shared_ptr<dolfin::Constant> constant_fn;

  std::map<std::string, std::vector<double>>::iterator bc;
  for (bc = coefficients.begin(); bc != coefficients.end(); ++bc) {
    if (coefficients.find(bc->first)->second.size() == 1) {
      constant_fn.reset( new dolfin::Constant(coefficients.find(bc->first)->second[0]) );
    } else {
      constant_fn.reset( new dolfin::Constant(coefficients.find(bc->first)->second) );
    }

    _bilinear_form->set_coefficient(bc->first, constant_fn);
    _linear_form->set_coefficient(bc->first, constant_fn);

    _bilinear_coefficient.emplace(bc->first, constant_fn);
    _linear_coefficient.emplace(bc->first, constant_fn);
  }

  std::map<std::string, std::vector<double>>::iterator lc;
  for (lc = sources.begin(); lc != sources.end(); ++lc) {
    if (sources.find(lc->first)->second.size() == 1) {
      constant_fn.reset( new dolfin::Constant(sources.find(lc->first)->second[0]) );
    } else {
      constant_fn.reset( new dolfin::Constant(sources.find(lc->first)->second) );
    }

    _linear_form->set_coefficient(lc->first, constant_fn);
    _linear_coefficient.emplace(lc->first, constant_fn);
  }
}
//--------------------------------------




//--------------------------------------
void PDE::get_dofs() {
  std::size_t dof;
  std::vector<std::size_t> component(1);
  std::vector<dolfin::la_index> index_vector;
  const dolfin::la_index n_first = _function_space->dofmap()->ownership_range().first;
  const dolfin::la_index n_second = _function_space->dofmap()->ownership_range().second;

  for (std::size_t comp_index = 0; comp_index < PDE::get_solution_dimension(); comp_index++) {
    component[0] = comp_index;
    index_vector.clear();
    std::shared_ptr<dolfin::GenericDofMap> dofmap = _function_space->dofmap()
      ->extract_sub_dofmap(component, *_mesh);

    for (dolfin::CellIterator cell(*_mesh); !cell.end(); ++cell) {
      dolfin::ArrayView<const dolfin::la_index> cell_dof = dofmap->cell_dofs(cell->index());

      for (std::size_t i = 0; i < cell_dof.size(); ++i) {
        dof = cell_dof[i];
        if (dof >= n_first && dof < n_second)
          index_vector.push_back(dof);
      }
    }

    std::sort(index_vector.begin(), index_vector.end());
    index_vector.erase(std::unique(index_vector.begin(), index_vector.end()), index_vector.end());
    _dof_map[comp_index].swap(index_vector);
  }
}
//--------------------------------------
void PDE::remove_Dirichlet_dof (
  std::vector<std::size_t> coordinates
) {

  std::size_t dimension = PDE::get_solution_dimension();

  _dirichlet_SubDomain.clear();
  _dirichlet_SubDomain.resize(dimension);

  _dirichletBC.clear();
  _dirichletBC.resize(dimension);

  std::shared_ptr<dolfin::Constant> zero_constant;
  zero_constant.reset(new dolfin::Constant(0.0));

  for (std::size_t i = 0; i < dimension; i++) {
    _dirichlet_SubDomain[i].reset(
      new Dirichlet_Subdomain::Dirichlet_Subdomain(
        {coordinates[i]},
        _mesh_min,
        _mesh_max,
        _mesh_epsilon
      )
    );
    _dirichletBC[i].reset(
      new dolfin::DirichletBC((*_function_space)[i], zero_constant, _dirichlet_SubDomain[i])
    );
  }
}

//--------------------------------------
void PDE::set_DirichletBC (
  std::vector<std::size_t> component,
  std::vector<std::vector<double>> boundary
) {

  PDE::remove_Dirichlet_dof(component);

  std::vector<Linear_Function::Linear_Function> interpolant_vector;
  for (std::size_t i = 0; i < component.size(); i++) {
    Linear_Function::Linear_Function linear_interpolant(
     component[i],
     _mesh_min[component[i]],
     _mesh_max[component[i]],
     boundary[i][0],
     boundary[i][1]
    );
    interpolant_vector.push_back(linear_interpolant);
  }

  PDE::set_solution(interpolant_vector);
}
//--------------------------------------
std::vector<std::shared_ptr<dolfin::SubDomain>> PDE::get_Dirichlet_SubDomain () {
  return _dirichlet_SubDomain;
}
//--------------------------------------





//--------------------------------------
dolfin::Function PDE::dolfin_solve () {
  dolfin::Function solution_update(_function_space);
  dolfin::Equation equation(_bilinear_form, _linear_form);

  std::vector<const dolfin::DirichletBC *> dirichletBC_vector;
  for (std::size_t i = 0; i < _dirichletBC.size(); i++) {
    dirichletBC_vector.push_back( &(*(_dirichletBC.at(i))) );
  }

  dolfin::solve(equation, solution_update, dirichletBC_vector);
  *(_solution_function->vector()) += *(solution_update.vector());

  _bilinear_form->set_coefficient("uu", _solution_function);
  _linear_form->set_coefficient("uu", _solution_function);

  return *_solution_function;
}
//--------------------------------------






//--------------------------------------
void PDE::setup_linear_algebra () {
  dolfin::EigenMatrix eigen_matrix;
  dolfin::EigenVector eigen_vector;
  assemble(eigen_matrix, *_bilinear_form);
  assemble(eigen_vector, *_linear_form);

  if (_quasi_newton) {
    printf("WARNING: No support for quasi-Newton solver!!!\n");
  }

  for (std::size_t i = 0; i < _dirichletBC.size(); i++) {
    _dirichletBC[i]->apply(eigen_matrix);
    _dirichletBC[i]->apply(eigen_vector);
  }

  _eigen_matrix.reset( new const dolfin::EigenMatrix(eigen_matrix) );
  _eigen_vector.reset( new const dolfin::EigenVector(eigen_vector) );
}
//--------------------------------------
dolfin::Function PDE::_convert_EigenVector_to_Function (
  const dolfin::EigenVector &eigen_vector
) {
  dolfin::Function fn(_function_space);

  if (eigen_vector.size() != _solution_function->vector()->size()) {
    printf("Cannot convert EigenVector to Function...\n");
    printf("\tincompatible dimensions!\n");
  }

  dolfin::la_index dof_index;
  for (std::size_t component = 0; component < _dof_map.size(); component++) {
    for (std::size_t index = 0; index < _dof_map[component].size(); index++) {
      dof_index = _dof_map[component][index];
      fn.vector()->setitem(dof_index, eigen_vector[dof_index]);
    }
  }

  return fn;
}
//--------------------------------------
void PDE::EigenMatrix_to_dCSRmat (
  std::shared_ptr<const dolfin::EigenMatrix> eigen_matrix,
  dCSRmat* dCSR_matrix
) {

  int row = eigen_matrix->size(0);
  int col = eigen_matrix->size(1);
  int nnz = eigen_matrix->nnz();
  if (row < 1 || col < 1 || nnz < 1) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenMatrix_to_dCSRmat");
  }

  int* JA;
  int* IA;
  double* val;
  IA = (int*) std::get<0> (eigen_matrix->data());
  JA = (int*) std::get<1> (eigen_matrix->data());
  val = (double*) std::get<2> (eigen_matrix->data());

  // Check for rows of zeros and add a unit diagonal entry
  for (uint row_index = 0; row_index < row; row_index++) {
    bool nonzero_entry = false;
    int diagColInd = -1;

    if (IA[row_index] < IA[row_index+1]) {
      for (uint col_index = IA[row_index]; col_index < IA[row_index + 1]; col_index++) {
        if (val[col_index] != 0.0) {
          nonzero_entry = true;
        }

        if (JA[col_index] == row_index) {
          diagColInd = col_index;
        }
      }
    }

    if (diagColInd < 0) {
      printf(" ERROR: diagonal entry not allocated!!\n\n Exiting... \n \n");
    }

    if (nonzero_entry == false) {
      printf(" Row %d has only zeros! Setting diagonal entry to 1.0 \n", row_index);
      val[diagColInd] = 1.0;
    }
  }

  // assign to dCSRmat
  dCSR_matrix->nnz = nnz;
  dCSR_matrix->row = row;
  dCSR_matrix->col = col;
  dCSR_matrix->IA = IA;
  dCSR_matrix->JA = JA;
  dCSR_matrix->val = val;
}
//--------------------------------------
void PDE::EigenVector_to_dvector (
  std::shared_ptr<const dolfin::EigenVector> eigen_vector,
  dvector* vector
) {
  int row = (int) eigen_vector->size();
  if (row < 1) {
    fasp_chkerr(ERROR_INPUT_PAR, "EigenVector_to_dvector");
  }

  vector->row = row;
  vector->val = (double*) eigen_vector->data();
}
//--------------------------------------

