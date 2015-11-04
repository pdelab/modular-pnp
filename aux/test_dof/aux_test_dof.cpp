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
#include "VecSpace.h"
#include "Space.h"
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
  VecSpace::FunctionSpace V(mesh);

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
  printf("Function values (Vector Space)\n");
  std::vector<double> values(initFunc.vector()->local_size(), 0);
  initFunc.vector()->get_local(values);
  std::cout <<"\tFirst component should be 1 ="<<  values[0] << std::endl;
  std::cout <<"\tSecond component should be 2 ="<< values[1] << std::endl;
  std::cout <<"\tThird component should be 3 ="<<  values[2] << std::endl;
  std::cout <<"\tFirst component should be 1 ="<<  values[3] << std::endl;
  printf("\n"); fflush(stdout);

  // Coordinates of the dofs for Vector Space (u1,u2,u3)
  printf("Coordinates the dofs (Vector Space)\n");
  std::vector<double> dof_coord= V.dofmap()->tabulate_all_coordinates(mesh);
  // dof_coord[9k+3l+0]= x of componant l
  // dof_coord[9k+3l+1]= y of componant l
  // dof_coord[9k+3l+2]= z of componant l
  // for l=0,1,2 and 0<=k<number of vertices or (number of dof)/3
  std::cout << "\t(x,y,z) of component 0 (dof=1) = " << dof_coord[0] << ", " << dof_coord[1] << ", " << dof_coord[2]  << std::endl;
  std::cout << "\t(x,y,z) of component 1 (dof=2) = " << dof_coord[3] << ", " << dof_coord[4] << ", " << dof_coord[5]  << std::endl;
  std::cout << "\t(x,y,z) of component 2 (dof=3) = " << dof_coord[6] << ", " << dof_coord[7] << ", " << dof_coord[8]  << std::endl;
  printf("\n"); fflush(stdout);

  // Maping of the dofs for Vector Space (u1,u2,u3)
  printf("Mapping dofs<-->Vertices (Vector Space)\n");
  int nn = V.dofmap()->num_entity_dofs(0);
  std::vector<dolfin::la_index> v_d = vertex_to_dof_map(V);
  std::vector<long unsigned int> d_v = dof_to_vertex_map(V);
  k=10;
  std::cout <<"\tvextex to dof:"<< std::endl;
  std::cout <<"\t\tcoord : "<< coord[3*k]<< ", " <<coord[3*k+1]<< ", " << coord[3*k+2] << std::endl;
  std::cout <<"\t\tdof_coord (u) : "<< dof_coord[3*v_d[3*k]]<< ", " << dof_coord[3*v_d[3*k]+1]<< ", " << dof_coord[3*v_d[3*k]+2] << std::endl;
  std::cout <<"\t\tdof_coord (v) : "<< dof_coord[3*v_d[3*k]+3+3]<< ", " << dof_coord[3*v_d[3*k]+3+1]<< ", " << dof_coord[3*v_d[3*k]+3+2] << std::endl;
  std::cout <<"\t\tdof_coord (w) : "<< dof_coord[3*v_d[3*k]+6+3]<< ", " << dof_coord[3*v_d[3*k]+6+1]<< ", " << dof_coord[3*v_d[3*k]+6+2] << std::endl;
  std::cout <<"\tdof to vertex:"<< std::endl;
  // Conclusion coord=3k+a=> dof=3*v_d[3*k]+3l+a (a=0,1,2; l=0,1,2)
  k=2;
  std::cout <<"\t\tdof_coord (u) : "<< dof_coord[9*k]<< ", " << dof_coord[9*k+1]<< ", " << dof_coord[9*k+2] << std::endl;
  std::cout <<"\t\tdof_coord (v) : "<< dof_coord[9*k+3]<< ", " << dof_coord[9*k+3+1]<< ", " << dof_coord[9*k+3+2] << std::endl;
  std::cout <<"\t\tdof_coord (w) : "<< dof_coord[9*k+6]<< ", " << dof_coord[9*k+6+1]<< ", " << dof_coord[9*k+6+2] << std::endl;
  std::cout <<"\t\tcoord "<< coord[d_v[3*k]] << ", " <<coord[d_v[3*k]+1]<< ", " << coord[d_v[3*k]+2] << std::endl;
  // Conclusion dof=9k+3k+a => vertex=d_v[(9k+3k+a)/3] (a=0,1,2; l=0,1,2; k< number of vertex)
  printf("\n"); fflush(stdout);

  // Coordinates of dof for Function Space
  Space::FunctionSpace V0(mesh);
  printf("Coordinates the dofs (Function Space)\n");
  std::vector<double> dof0_coord = V0.dofmap()->tabulate_all_coordinates(mesh);
  //  dof0_coord[3k+0]= x
  //  dof0_coord[3k+1]= y
  //  dof0_coord[3k+2]= z
  // for 0<=k<number of vertices or (number of dof)
  k=20;
  std::cout << "\t(x,y,z) = " <<  dof0_coord[3*k] << ", " <<  dof0_coord[3*k+1] << ", " <<  dof0_coord[3*k+2]  << std::endl;
  printf("\n"); fflush(stdout);

  // Maping of the dofs for Function Space u
  printf("Mapping dofs<-->Vertices (Function Space)\n");
  std::vector<dolfin::la_index> v_d0 = vertex_to_dof_map(V0);
  std::vector<long unsigned int> d_v0 = dof_to_vertex_map(V0);
  std::cout <<"\tvextex to dof:"<< std::endl;
  k=5;
  std::cout <<"\t\tcoord : "<< coord[3*k+0]<< ", " <<coord[3*k+1]<< ", " << coord[3*k+2] << std::endl;
  std::cout <<"\t\tdof_coord : "<< dof0_coord[3*v_d0[k]]<< ", " << dof0_coord[3*v_d0[k]+1]<< ", " << dof0_coord[3*v_d0[k]+2] << std::endl;
  std::cout <<"\tdof to vertex:"<< std::endl;
  k=5;
  std::cout <<"\t\tcoord : "<< coord[3*d_v0[k]+0]<< ", " <<coord[3*d_v0[k]+1]<< ", " << coord[3*d_v0[k]+2] << std::endl;
  std::cout <<"\t\tdof_coord : "<<dof0_coord[3*k]<< ", " << dof0_coord[3*k+1]<< ", " << dof0_coord[3*k+2] << std::endl;

  printf("\n"); fflush(stdout);


  printf("\n-----------------------------------------------------------    "); fflush(stdout);
  printf("\n End                                                           "); fflush(stdout);
  printf("\n-----------------------------------------------------------\n\n"); fflush(stdout);

  return 0;
}
