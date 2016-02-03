#ifndef __SPHERES_H
#define __SPHERES_H

#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <ufc.h>

double xc[86] = { -17.0,18.0,25.0,-1.0,20.0,14.0,22.0,2.0,23.0,-2.0,3.0,
	-13.0,24.0,-4.0,22.0,21.0,-11.0,-24.0,-9.0,-18.0,-14.0,
	-2.0,1.0,17.0,16.0,-9.0,21.0,-24.0,-1.0,-7.0,-14.0,
	12.0,8.0,-8.0,8.0,3.0,7.0,-20.0,25.0,-22.0,-13.0,
	-14.0,-21.0,21.0,-3.0,22.0,21.0,-3.0,6.0,23.0,12.0,
	0.0,-6.0,-6.0,-12.0,-7.0,-4.0,-24.0,17.0,1.0,-22.0,
	-15.0,-21.0,-22.0,22.0,-24.0,-19.0,-15.0,15.0,-2.0,9.0,
	-5.0,19.0,-23.0,15.0,18.0,-16.0,17.0,9.0,19.0,-7.0,
	-20.0,-13.0,-3.0,-16.0,-11.0};

double yc[86] = { -16.0,11.0,-15.0,-3.0,-20.0,-2.0,20.0,20.0,15.0,-16.0,0.0,
	18.0,-24.0,5.0,-16.0,-15.0,-15.0,6.0,9.0,23.0,22.0,
	19.0,-12.0,-3.0,-23.0,4.0,24.0,22.0,-22.0,4.0,-9.0,
	-9.0,-15.0,-23.0,-15.0,14.0,-2.0,6.0,2.0,-23.0,-14.0,
	-25.0,22.0,-19.0,9.0,-15.0,20.0,4.0,15.0,-12.0,17.0,
	17.0,2.0,9.0,20.0,18.0,7.0,22.0,16.0,16.0,10.0,
	5.0,-23.0,-17.0,10.0,-15.0,20.0,24.0,-8.0,-7.0,2.0,
	-5.0,-22.0,12.0,-16.0,-13.0,-11.0,-22.0,-3.0,-25.0,2.0,
	3.0,10.0,-3.0,-24.0,-22.0};

double zc[86] = { 7.0,-19.0,-15.0,24.0,-19.0,-20.0,1.0,24.0,14.0,11.0,-21.0,
	-5.0,9.0,-7.0,21.0,15.0,20.0,17.0,-23.0,13.0,8.0,
	14.0,19.0,-4.0,-12.0,-5.0,-24.0,-6.0,2.0,10.0,-1.0,
	-20.0,-24.0,-18.0,-18.0,16.0,24.0,-22.0,13.0,7.0,-17.0,
	-17.0,-7.0,0.0,0.0,14.0,7.0,8.0,-22.0,-23.0,8.0,
	2.0,6.0,22.0,6.0,18.0,2.0,20.0,6.0,11.0,5.0,
	22.0,-6.0,10.0,1.0,-22.0,12.0,-15.0,-24.0,-7.0,-19.0,
	-15.0,-8.0,22.0,-3.0,-16.0,18.0,-17.0,-19.0,9.0,-9.0,
	-18.0,-8.0,16.0,13.0,24.0};

double rc[86] = { 9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,
	9.0,9.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,
	7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,
	7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,
	7.0,7.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,
	6.0,6.0,6.0,6.0,5.0};

int Numb_spheres = 20;

class SpheresSubDomain : public dolfin::SubDomain
{
    bool inside(const dolfin::Array<double>& x, bool on_boundary) const
    {
      bool flag=false;
      for (int i=0;i<Numb_spheres;i++){
          if (on_boundary && (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0) )
                  flag=true;
                }
          return flag;
    }

};

/// Initialize expressions
class Cation_SPH : public dolfin::Expression
{
public:
  // constructor
  Cation_SPH(double lower_val, double upper_val,
    double lower, double upper, int bc_coord)
		{
			_lower_val = lower_val;
			_upper_val = upper_val;
			_lower = lower;
			_upper = upper;
			_bc_coord = bc_coord;
		}
  // evaluate LogCarge
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
	{
		values[0]  = std::log(_lower_val) * (_upper - x[_bc_coord]) / (_upper - _lower);
	  values[0] += std::log(_upper_val) * (x[_bc_coord] - _lower) / (_upper - _lower);
		// for (int i=0;i<Numb_spheres;i++){
		// 		if (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0)
		// 						values[0]=1.0;
		// }
	}
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};

/// Initialize expressions
class Anion_SPH : public dolfin::Expression
{
public:
  // constructor
  Anion_SPH(double lower_val, double upper_val,
    double lower, double upper, int bc_coord)
		{
			_lower_val = lower_val;
			_upper_val = upper_val;
			_lower = lower;
			_upper = upper;
			_bc_coord = bc_coord;
		}
  // evaluate LogCarge
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
	{
		values[0]  = std::log(_lower_val) * (_upper - x[_bc_coord]) / (_upper - _lower);
	  values[0] += std::log(_upper_val) * (x[_bc_coord] - _lower) / (_upper - _lower);
		// for (int i=0;i<Numb_spheres;i++){
		// 		if (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0)
		// 						values[0]=0.1;
		// }
	}
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};

class Potential_SPH: public dolfin::Expression
{
public:
  // constructor
  Potential_SPH(double lower_val, double upper_val,
    double lower, double upper, int bc_coord)
		{
		  _lower_val = lower_val;
		  _upper_val = upper_val;
		  _lower = lower;
		  _upper = upper;
		  _bc_coord = bc_coord;
		}
  // evaluate Voltage
  void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
	{
	  values[0]  = _lower_val * (_upper - x[_bc_coord]) / (_upper - _lower);
	  values[0] += _upper_val * (x[_bc_coord] - _lower) / (_upper - _lower);
		// for (int i=0;i<Numb_spheres;i++){
		// 		if (std::pow(x[0]-xc[i],2) + std::pow(x[1]-yc[i],2) + std::pow(x[2]-zc[i],2) < std::pow(rc[i],2)+2.0)
		// 						values[0]=1.0;
		// }
	}
private:
  double _lower_val, _upper_val, _upper, _lower;
  int _bc_coord;
};

#endif
