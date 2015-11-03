/*! \file test_eafe.cpp
 *
 *  \brief Main to test EAFE functionality on linear convection reaction problem
 *
 *  \note Currently initializes the problem based on specification
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <dolfin.h>
#include "FESpace.h"
#include "fasp_to_fenics.h"
#include "boundary_conditions.h"
#include "newton.h"
#include "newton_functs.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
  INT fasp_solver_dcsr_krylov (dCSRmat *A,
   dvector *b,
   dvector *x,
   itsolver_param *itparam
  );
#define FASP_BSR     ON  /** use BSR format in fasp */
}
using namespace dolfin;
// using namespace std;


class BigExp : public Expression
{
public:
  BigExp() : Expression(3) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
  }
};

int main()
{

  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n Test of dofs                                                  "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  // Need to use Eigen for linear algebra
  parameters["linear_algebra_backend"] = "Eigen";
  parameters["allow_extrapolation"] = true;

  // read domain parameters
  domain_param domain_par;
  char domain_param_filename[] = "./benchmarks/PNP/domain_params.dat";
  domain_param_input(domain_param_filename, &domain_par);
  print_domain_param(&domain_par);

  // build mesh
  dolfin::Mesh mesh;
  dolfin::MeshFunction<std::size_t> subdomains;
  dolfin::MeshFunction<std::size_t> surfaces;
  dolfin::File meshOut(domain_par.mesh_output);
  domain_build(&domain_par, &mesh, &subdomains, &surfaces, &meshOut);



  // Function space for PNP (Cation,Anion,Phi)
  FESpace::FunctionSpace V(mesh);

  // Test on init function
  Function initFunc(V);
  BigExp BigFunc;
  Constant C1(1.0);
  initFunc.interpolate(BigFunc);



  // coordinates
  //    when  n=number of vertex, and 0<= k<n :
  //        x=coord[3*k],  y=coord[3*k+1],  z=coord[3*k+2]
  printf("Mesh & Coordinates\n");
  int n = V.dim();
  int d = mesh.geometry().dim();
  std::vector<double> coord = mesh.coordinates();
  int n_vert = coord.size()/d;
  printf("\tNumber of vertices = %d\n",n_vert);
  int k = 0;
  std::cout <<"\t x ,y z at 0"<< coord[3*k]<< ", " <<coord[3*k+1]<< ", " << coord[3*k+2] << std::endl;
  k = n_vert-1;
  std::cout <<"\t x ,y z at #vert"<< coord[3*k]<< ", " <<coord[3*k+1]<< ", " << coord[3*k+2] << std::endl;
  printf("\n"); fflush(stdout);

  // Function on MixedSpace is 3k=Component 1, 3k+1=Component 2, 3k+2=Component 3
  // values[3k+l] = values at of the component l at the dof
  // with 0<=k<number of vertex or (number of dof)/3
  printf("Function values\n");
  std::vector<double> values(initFunc.vector()->local_size(), 0);
  initFunc.vector()->get_local(values);
  std::cout <<"\tFirst component should be 1 ="<<  values[0] << std::endl;
  std::cout <<"\tSecond component should be 2 ="<< values[1] << std::endl;
  std::cout <<"\tThird component should be 3 ="<<  values[2] << std::endl;
  std::cout <<"\tFirst component should be 1 ="<<  values[3] << std::endl;
  printf("\n"); fflush(stdout);

  // Maping of the dofs
  printf("Coordinates the dofs\n");
  std::shared_ptr<const GenericDofMap> dof = V.dofmap();
  std::vector<double> dof_x = dof->tabulate_all_coordinates(mesh);
  // dof_x[9k+3l+0]= x of componant l
  // dof_x[9k+3l+1]= y of componant l
  // dof_x[9k+3l+2]= z of componant l
  // for l=0,1,2 and 0<=k<number of vertices or (number of dof)/3
  std::cout << "\t(x,y,z) of component 0 (dof=1)=" << dof_x[0] << ", " << dof_x[1] << ", " << dof_x[2]  << std::endl;
  std::cout << "\t(x,y,z) of component 1 (dof=2)=" << dof_x[3] << ", " << dof_x[4] << ", " << dof_x[5]  << std::endl;
  std::cout << "\t(x,y,z) of component 2 (dof=3)=" << dof_x[6] << ", " << dof_x[7] << ", " << dof_x[8]  << std::endl;

  printf("Mapping dofs<-->Vertices\n");
  int nn = V.dofmap()->num_entity_dofs(0);
  //V->dofmap()->dof_to_vertex_map(mesh).
  std::vector<dolfin::la_index> v_d = vertex_to_dof_map(V);
  std::vector<long unsigned int> d_v = dof_to_vertex_map(V);
  k=10;
  std::cout <<"\tvextex to dof:"<< std::endl;
  std::cout <<"\t\tcoord "<< coord[3*10/nn]<< ", " <<coord[3*10/nn+1]<< ", " << coord[3*10/nn+2] << std::endl;
  std::cout <<"\t\tdof_x "<< dof_x[v_d[10]]<< ", " << dof_x[v_d[10]+1]<< ", " << dof_x[v_d[10]+2] << std::endl;
  std::cout <<"\tdof to vertex:"<< std::endl;
  std::cout <<"\t\tdof_x "<< dof_x[11]<< ", " << dof_x[11+1]<< ", " << dof_x[11+2] << std::endl;
  // std::cout << d_v[10] << " " << nn << " " << d_v[10]/nn << std::endl ;
  std::cout <<"\t\tcoord "<< coord[d_v[10]/nn] << ", " <<coord[d_v[10]/nn+1]<< ", " << coord[d_v[10]/nn+2] << std::endl;
  printf("\n"); fflush(stdout);

  // di_dx = dof_x.tolist()
  // print di_dx[10], di_dx[125+10]
  // vertex_x = mesh.coordinates().reshape((-1, d))
  // vi_vx = vertex_x.tolist()
  // print vi_vx[10]
  //
  // coor = mesh.coordinates()
  // print np.size(coor[:,0]), np.size(coor[0,:])
  // print coor[int(d_v[10])/nn]
  // print di_dx[v_d[10]], coor[10/nn]

  // printf("\t %f\n",mesh.coordinates()[0]);
  printf("\n"); fflush(stdout);






  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
