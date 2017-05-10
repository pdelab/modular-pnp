#ifndef __DIRICHLET_SUBDOMAIN_H
#define __DIRICHLET_SUBDOMAIN_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <dolfin.h>

class Dirichlet_Subdomain : public dolfin::SubDomain {
  public:

    /// Constructor
    Dirichlet_Subdomain (
      std::vector<std::size_t> coordinates,
      std::vector<double> mesh_min,
      std::vector<double> mesh_max,
      double epsilon
    );

    /// Destructor
    ~Dirichlet_Subdomain ();

    /// Determine whether a point is inside the region
    bool inside (
      const dolfin::Array<double>& x,
      bool on_boundary
    ) const;

  private:
    std::vector<std::size_t> _coordinates;
    std::vector<double> _mesh_min, _mesh_max;
    double _epsilon;
};

class Linear_Function : public dolfin::Expression {
  public:

    /// Constructor
    Linear_Function (
      std::size_t coordinate,
      double mesh_min,
      double mesh_max,
      double lower_value,
      double upper_value
    );

    Linear_Function (
      std::size_t coordinate,
      double mesh_min,
      double mesh_max,
      std::vector<double> lower_values,
      std::vector<double> upper_values
    );



    /// Destructor
    ~Linear_Function ();

    /// Determine whether a point is inside the region
    void eval (
      dolfin::Array<double>& values,
      const dolfin::Array<double>& x
    ) const;

  private:
    std::size_t _coordinate;
    double _mesh_min, _mesh_max;
    double _lower_value, _upper_value;
    std::vector<double> _lower_values, _upper_values;
    double _distance;
    int _dimension;
};


#endif
