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


/*------- In file: boundary_conditions.cpp -------*/

class XBoundaries : public dolfin::SubDomain {
public:
  // constructor
  XBoundaries(double lower, double upper);
  // check if a point is in Xboundaries
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
private:
  double _lower, _upper;
};

class YBoundaries : public dolfin::SubDomain {
public:
  // constructor
  YBoundaries(double lower, double upper);
  // check if a point is in Yboundaries
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
private:
  double _lower, _upper;
};

class ZBoundaries : public dolfin::SubDomain {
public:
  // constructor
  ZBoundaries(double lower, double upper);
  // check if a point is in Zboundaries
  bool inside(const dolfin::Array<double>& x, bool on_boundary) const;
private:
  double _lower, _upper;
};

/// Initialize expressions
class LogCharge : public dolfin::Expression
{
public:
  // constructor
  LogCharge(double lower_val, double upper_val,
    double lower, double upper, int bc_coord);
  // evaluate LogCarge
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const;
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};

class Voltage : public dolfin::Expression
{
public:
  // constructor
  Voltage(double lower_val, double upper_val,
    double lower, double upper, int bc_coord);
  // evaluate Voltage
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const;
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};

#endif
