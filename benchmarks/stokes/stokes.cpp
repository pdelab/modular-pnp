#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "EAFE.h"
#include "funcspace_to_vecspace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "newton.h"
#include "newton_functs.h"
#include "L2Error.h"
#include "stokes.h"
#include <time.h>
#include <boost/filesystem.hpp>

using namespace dolfin;
// using namespace std;
extern "C"
{
  #include "fasp.h"
  #include "fasp_block.h"
  #include "fasp_functs.h"

  #include "fasp4ns.h"
  #include "fasp4ns_functs.h"
}

class zerovec4 : public dolfin::Expression
{
public:

  zerovec4() : Expression(4) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 0.0;
    values[1] = 0.0;
    values[2] = 0.0;
    values[3] = 0.0;
  }

};

class Bd_all : public dolfin::SubDomain
{
    bool inside(const dolfin::Array<double>& x, bool on_boundary) const
    {
        return on_boundary;
    }

};

class FluidVelocity : public dolfin::Expression
{
public:
    FluidVelocity(double out_flow, double in_flow, double bc_dist, int bc_dir): Expression(3),outflow(out_flow),inflow(in_flow),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
    {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
        if ( std::fabs(x[0]) > 0.5 ) {
            values[bc_direction]  = outflow*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
            values[bc_direction] -=  inflow*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
        }
    }
private:
    double outflow, inflow, bc_distance;
    int bc_direction;
};

int main()
{

  std::string Folder="./benchmarks/stokes//Data/";
  std::string FolderFig="./benchmarks/stokes/Figures/";

  // Deleting the folders:
  boost::filesystem::remove_all(Folder);
  boost::filesystem::remove_all(FolderFig);

  File FileU(FolderFig+"u.pvd");
  File FileP(FolderFig+"p.pvd");

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  //*************************************************************
  //  Initialization
  //*************************************************************
  printf("Initialize the problem\n"); fflush(stdout);
  // read domain parameters
  printf("domain...\n"); fflush(stdout);
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/pnp_stokes/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  printf("mesh...\n"); fflush(stdout);
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  // mesh
  dolfin::Point p0( -domain_par.length_x/2, -domain_par.length_y/2, -domain_par.length_z/2);
  dolfin::Point p1(  domain_par.length_x/2,  domain_par.length_y/2,  domain_par.length_z/2);
  auto mesh = std::make_shared<dolfin::BoxMesh>(p0, p1, domain_par.grid_x, domain_par.grid_y, domain_par.grid_z);

  // read coefficients and boundary values
  printf("coefficients...\n"); fflush(stdout);
  coeff_param coeff_par, non_dim_coeff_par;
  char coeff_param_filename[] = "./benchmarks/stokes/coeff_params.dat";
  coeff_param_input(coeff_param_filename, &coeff_par);
  // non_dimesionalize_coefficients(&domain_par, &coeff_par, &non_dim_coeff_par);
  print_coeff_param(&coeff_par);

  // initialize Newton solver parameters
  printf("Newton solver parameters...\n"); fflush(stdout);
  newton_param newtparam;
  char newton_param_file[] = "./benchmarks/stokes/newton_param.dat";
  newton_param_input (newton_param_file, &newtparam);
  print_newton_param(&newtparam);
  double initial_residual, relative_residual = 1.0;

  INT status = FASP_SUCCESS;

  // Setup FASP solver
  printf("FASP solver parameters for stokes..."); fflush(stdout);
  input_ns_param stokes_inpar;
  itsolver_ns_param stokes_itpar;
  AMG_ns_param  stokes_amgpar;
  ILU_param stokes_ilupar;
  Schwarz_param stokes_schpar;
  // char fasp_ns_params[] = "./nsbcsr.dat";
  char fasp_ns_params[] = "./benchmarks/stokes//ns.dat";
  fasp_ns_param_input(fasp_ns_params, &stokes_inpar);
  fasp_ns_param_init(&stokes_inpar, &stokes_itpar, &stokes_amgpar, &stokes_ilupar, &stokes_schpar);
  printf("done\n"); fflush(stdout);

  auto V = std::make_shared< stokes::FunctionSpace>(mesh);
  File FileMesh(Folder+"mesh.pvd");
  FileMesh << *mesh;

  auto zero = std::make_shared<Constant>(0.0);
  auto zero_vec = std::make_shared<Constant>(0.0,0.0);
  auto zero_vec3 = std::make_shared<Constant>(0.0,0.0,0.0);
  auto one_vec3 = std::make_shared<Constant>(1.0,1.0,1.0);
  auto zero_vec2=std::make_shared<Constant>(0.0, 0.0);
  auto mu=std::make_shared<Constant>(0.1);
  auto penalty=std::make_shared<Constant>(1.0e-2);

  stokes::BilinearForm a(V,V);
  stokes::LinearForm L(V);

  // Set Dirichlet boundaries
  printf("\tboundary conditions...\n"); fflush(stdout);
  auto boundary = std::make_shared<SymmBoundaries>(coeff_par.bc_coordinate, -domain_par.length_x/2.0, domain_par.length_x/2.0);
  auto bddd = std::make_shared<Bd_all>();
  dolfin::DirichletBC bc(V->sub(0), one_vec3, bddd);

  Function Solution(V);
  ivector dof_u;
  ivector dof_p;
  get_dofs(&Solution, &dof_u, 0);
  get_dofs(&Solution, &dof_p, 1);


  EigenMatrix A;
  EigenVector b;

  a.mu = mu;
  a.alpha = penalty;
  assemble(A,a);

  L.f = zero_vec3;
  assemble(b,L);
  bc.apply(A,b);

  // Fasp matrices and vectors
  block_dCSRmat A_fasp;
  dvector b_fasp, solu_fasp;
  std::vector<double> stokes_value_vector;
  stokes_value_vector.reserve(b.size());
  unsigned int index;

  copy_EigenVector_to_block_dvector(&b, &b_fasp, &dof_u, &dof_p);
  copy_EigenMatrix_to_block_dCSRmat(&A, &A_fasp, &dof_u, &dof_p);
  fasp_dvec_alloc(b.size(), &solu_fasp);
  fasp_dvec_set(b_fasp.row, &solu_fasp, 0.0);

  // FENiCS Solver
  // Create CG Krylov solver and turn convergence monitoring on
  // EigenKrylovSolver solver("gmres","ilu");
  // solver.solve(A,sol,b);
  // solver.parameters["relative_tolerance"] = 1e-8;
  // solver.parameters["maximum_iterations"] = 1000;

  // STOKES SOLVER
  status = fasp_solver_bdcsr_krylov_navier_stokes (
    &A_fasp,
    &b_fasp,
    &solu_fasp,
    &stokes_itpar,
    &stokes_amgpar,
    &stokes_ilupar,
    &stokes_schpar
  );
  // status = fasp_solver_dcsr_krylov(&A_fasp, &b_fasp, &solu_fasp, &itpar);
  if (status < 0)
    printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", status);
  // else
  //   printf("\tsolved linear system successfully...\n");

  // convert stokes update to functions
  for (index = 0; index < dof_u.row; index++) {
    stokes_value_vector[dof_u.val[index]] = solu_fasp.val[index];
  }
  for (index = 0; index < dof_p.row; index++) {
    stokes_value_vector[dof_p.val[index]] = solu_fasp.val[dof_u.row+index];
  }
  (Solution).vector()->set_local(stokes_value_vector);

  Function u = Solution[0];
  Function p = Solution[1];

  // copy_dvector_to_EigenVector(&solu_fasp, &sol);
  fasp_dvec_free(&solu_fasp);


  File FileData(Folder+"u.xml");
  FileData << u;

  FileU << u;
  FileP << p;

  return 1;

}
