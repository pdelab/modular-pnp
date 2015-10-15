/*! \file boundary_conditions.cpp
 *  \brief contains the functions for each class of boundary conditions
 */

#include "boundary_conditions.h"

using namespace dolfin;

/////////////////////////////////////////////////////////////////////////////
///  Sub Domains
/////////////////////////////////////////////////////////////////////////////


XBoundaries::XBoundaries(double _Lx)
{
  Lx=_Lx;
}
// Return 1 if on the boundaries x=-Lx or x=Lx, 0 otherwise
bool XBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return on_boundary && (x[0] < -Lx+5*DOLFIN_EPS or x[0] > Lx-5*DOLFIN_EPS); //(is that ok ?)
}

YBoundaries::YBoundaries(double _Ly)
{
  Ly=_Ly;
}
// Return 1 if on the boundaries x=-Lx or x=Lx, 0 otherwise
bool YBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return on_boundary && (x[2] < -Ly+5*DOLFIN_EPS or x[1] > Ly-5*DOLFIN_EPS); //(is that ok ?)
}

ZBoundaries::ZBoundaries(double _Lz)
{
  Lz=_Lz;
}
// Return 1 if on the boundaries z=-Lz or z=Lz, 0 otherwise
bool ZBoundaries::inside(const Array<double>& x, bool on_boundary) const
{
  return ( on_boundary && (x[2] < -Lz + 5*DOLFIN_EPS or x[2] > Lz - 5*DOLFIN_EPS) );
}

dielectricChannel::dielectricChannel(double _Lz)
{
  Lz=_Lz;
}
// Return 1 if on ball inside, 0 otherwise
bool dielectricChannel::inside(const Array<double>& x, bool on_boundary) const
{
  bool toppatches = ((   (x[0] < -10./3.+DOLFIN_EPS) or (std::fabs(x[0]+5./6.) < 5./6.+DOLFIN_EPS)
                            or (std::fabs(x[0]-15./6.) < 5./6.+DOLFIN_EPS))
                           and x[2] > Lz - DOLFIN_EPS  );

  bool bottompatches = ((   (x[0] > 10./3.-DOLFIN_EPS) or (std::fabs(x[0]-5./6.) < 5./6.+DOLFIN_EPS)
                               or (std::fabs(x[0]+15./6.) < 5./6.+DOLFIN_EPS))
                              and x[2] < -Lz + DOLFIN_EPS  );

  return ( on_boundary && (toppatches or bottompatches) );
}

/////////////////////////////////////////////////////////////////////////////
///  Boundary Conditions
/////////////////////////////////////////////////////////////////////////////

//  Initial Sodium Number Density Profile
LogCharge::LogCharge(double ext_bulk, double int_bulk, double bc_dist, int bc_dir) : Expression()
{
  ext_contact=ext_bulk;
  int_contact=int_bulk;
  bc_distance=bc_dist;
  bc_direction=bc_dir;
}
void LogCharge::eval(Array<double>& values, const Array<double>& x) const
{
    values[0]  = log(ext_contact)*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
    values[0] -= log(int_contact)*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
}

//  Voltage
 Voltage::Voltage(double ext_volt, double int_volt, double bc_dist, int bc_dir): Expression()
 {
   ext_voltage=ext_volt;
   int_voltage=int_volt;
   bc_distance=bc_dist;
   bc_direction=bc_dir;
 }
void Voltage::eval(Array<double>& values, const Array<double>& x) const
{
  values[0]  = ext_voltage*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
  values[0] -= int_voltage*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
}
