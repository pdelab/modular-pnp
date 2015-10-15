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

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <ufc.h>


/////////////////////////////////////////////////////////////////////////////
///  Sub Domains
/////////////////////////////////////////////////////////////////////////////

class XBoundaries: public dolfin::SubDomain {
  double Lx;
public:
  XBoundaries(double _Lx);
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
};

class YBoundaries: public dolfin::SubDomain {
  double Ly;
public:
  YBoundaries(double _Ly);
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
};

class ZBoundaries: public dolfin::SubDomain {
double Lz;
public:
  ZBoundaries(double _Lz);
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
};

class dielectricChannel : public dolfin::SubDomain
{
  double Lz;
public:
    dielectricChannel(double _Lz);
    bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
};
/////////////////////////////////////////////////////////////////////////////
///  Boundary Conditions
/////////////////////////////////////////////////////////////////////////////

class LogCharge : public dolfin::Expression
{
public:
    LogCharge(double ext_bulk, double int_bulk, double bc_dist, int bc_dir);
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const;
private:
    double ext_contact, int_contact, bc_distance;
    int bc_direction;
};

//  Voltage
class Voltage : public dolfin::Expression
{
    double ext_volt;
    double int_volt;
    double bc_dist;
    int bc_dir;
public:
    Voltage(double ext_volt, double int_volt, double bc_dist, int bc_dir);
    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const;
private:
    double ext_voltage, int_voltage, bc_distance;
    int bc_direction;
};


#endif
