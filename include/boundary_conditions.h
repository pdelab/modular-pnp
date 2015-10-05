/*! \file boundary_conditions.h
 *  \brief Main header file for Boundary_Conditions.cpp
 */

#ifndef __BOUNDARY_CONDITIONS_H
#define __BOUNDARY_CONDITIONS_H

#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>

double Lx;
double Ly;
double Lz;

/////////////////////////////////////////////////////////////////////////////
///  Sub Domains
/////////////////////////////////////////////////////////////////////////////

// Sub domain for x=-Lx and x=Lx boundary conditions
// Before it was: class DirichletBoundary : public SubDomain
class EastWestDomain: public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        // Before it was: return on_boundary && (x[0] < -Lx+2.*DOLFIN_EPS or x[0] > Lx-2.*DOLFIN_EPS);
        // Arthur changed it to:
        return on_boundary && (x[0] < -Lx+DOLFIN_EPS || x[0] > Lx-DOLFIN_EPS); //(is that ok ?)
    }
};

// Sub domain for homogeneous channel wall
class channelGate : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return ( on_boundary && (x[2] < -Lz + DOLFIN_EPS or x[2] > Lz - DOLFIN_EPS) );
    }
};

// Sub domain for homogeneous channel wall
class dielectricChannel : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        bool toppatches = ((   (x[0] < -10./3.+DOLFIN_EPS) or (std::fabs(x[0]+5./6.) < 5./6.+DOLFIN_EPS)
                            or (std::fabs(x[0]-15./6.) < 5./6.+DOLFIN_EPS))
                           and x[2] > Lz - DOLFIN_EPS  );

        bool bottompatches = ((   (x[0] > 10./3.-DOLFIN_EPS) or (std::fabs(x[0]-5./6.) < 5./6.+DOLFIN_EPS)
                               or (std::fabs(x[0]+15./6.) < 5./6.+DOLFIN_EPS))
                              and x[2] < -Lz + DOLFIN_EPS  );

        return ( on_boundary && (toppatches or bottompatches) );
    }
};

/////////////////////////////////////////////////////////////////////////////
///  Boundary Conditions
/////////////////////////////////////////////////////////////////////////////

//  Initial Sodium Number Density Profile
class LogCharge : public Expression
{
public:
    LogCharge(double ext_bulk, double int_bulk, double bc_dist, int bc_dir): Expression(),ext_contact(ext_bulk),int_contact(int_bulk),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0]  = log(ext_contact)*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
        values[0] -= log(int_contact)*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
    }
private:
    double ext_contact, int_contact, bc_distance;
    int bc_direction;
};


//  Voltage
class Voltage : public Expression
{
public:
    Voltage(double ext_volt, double int_volt, double bc_dist, int bc_dir): Expression(),ext_voltage(ext_volt),int_voltage(int_volt),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0]  = ext_voltage*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
        values[0] -= int_voltage*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
    }
private:
    double ext_voltage, int_voltage, bc_distance;
    int bc_direction;
};



//  Dirichlet boundary condition
class DirichletBC : public Expression
{
public:

    DirichetBC() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {

        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
    }
};


#endif
