/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(tersoff,PairTersoff);
// clang-format on
#else

#ifndef LMP_PAIR_TERSOFF_H
#define LMP_PAIR_TERSOFF_H

#include <cmath>

#include "math_const.h"
#include "pair.h"

namespace LAMMPS_NS {

class PairTersoff : public Pair {
 friend class StencilMD;
 public:
  PairTersoff(class LAMMPS *);
  ~PairTersoff() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_ATOM> void eval();

  static constexpr int NPARAMS_PER_LINE = 17;

  struct Param {
    double lam1, lam2, lam3;
    double c, d, h;
    double gamma, powerm;
    double powern, beta;
    double biga, bigb, bigd, bigr;
    double cut, cutsq;
    double c1, c2, c3, c4;
    int ielement, jelement, kelement;
    int powermint;
    double Z_i, Z_j;    // added for TersoffZBL
    double ZBLcut, ZBLexpscale;
    double c5, ca1, ca4;    // added for TersoffMOD
    double powern_del;
    double c0;    // added for TersoffMODC
  };

 protected:
  Param *params;      // parameter set for an I-J-K interaction
  double cutmax;      // max cutoff for all elements
  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array

  int shift_flag;    // flag to turn on/off shift
  double shift;      // negative change in equilibrium bond length

  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();
  virtual void repulsive(Param *, double, double &, int, double &);
  virtual double zeta(Param *, double, double, double *, double *);
  virtual void force_zeta(Param *, double, double, double &, double &, int, double &);
  void attractive(Param *, double, double, double, double *, double *, double *, double *,
                  double *);

  virtual double ters_fc(double, Param *);
  virtual double ters_fc_d(double, Param *);
  virtual double ters_fa(double, Param *);
  virtual double ters_fa_d(double, Param *);
  virtual double ters_bij(double, Param *);
  virtual double ters_bij_d(double, Param *);

  virtual void ters_zetaterm_d(double, double *, double, double, double *, double, double, double *,
                               double *, double *, Param *);
  void costheta_d(double *, double, double *, double, double *, double *, double *);

  // inlined functions for efficiency

  inline double ters_gijk(const double costheta, const Param *const param) const
  {
    const double ters_c = param->c * param->c;
    const double ters_d = param->d * param->d;
    const double hcth = param->h - costheta;

    return param->gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));
  }

  inline double ters_gijk_d(const double costheta, const Param *const param) const
  {
    const double ters_c = param->c * param->c;
    const double ters_d = param->d * param->d;
    const double hcth = param->h - costheta;
    const double numerator = -2.0 * ters_c * hcth;
    const double denominator = 1.0 / (ters_d + hcth * hcth);
    return param->gamma * numerator * denominator * denominator;
  }
};

}    // namespace LAMMPS_NS

inline double LAMMPS_NS::PairTersoff::ters_fc(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R - ters_D) return 1.0;
  if (r > ters_R + ters_D) return 0.0;
  return 0.5 * (1.0 - sin(MathConst::MY_PI2 * (r - ters_R) / ters_D));
}

inline double LAMMPS_NS::PairTersoff::ters_fc_d(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R - ters_D) return 0.0;
  if (r > ters_R + ters_D) return 0.0;
  return -(MathConst::MY_PI4 / ters_D) * cos(MathConst::MY_PI2 * (r - ters_R) / ters_D);
}

inline double LAMMPS_NS::PairTersoff::ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r) * ters_fc(r, param);
}

inline double LAMMPS_NS::PairTersoff::ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) *
         (param->lam2 * ters_fc(r, param) - ters_fc_d(r, param));
}

inline double LAMMPS_NS::PairTersoff::ters_bij(double zeta, Param *param)
{
  const double tmp = param->beta * zeta;
  if (tmp > param->c1) return 1.0 / sqrt(tmp);
  if (tmp > param->c2)
    return (1.0 - pow(tmp, -param->powern) / (2.0 * param->powern)) / sqrt(tmp);
  if (tmp < param->c4) return 1.0;
  if (tmp < param->c3) return 1.0 - pow(tmp, param->powern) / (2.0 * param->powern);
  return pow(1.0 + pow(tmp, param->powern), -1.0 / (2.0 * param->powern));
}

inline double LAMMPS_NS::PairTersoff::ters_bij_d(double zeta, Param *param)
{
  const double tmp = param->beta * zeta;
  if (tmp > param->c1) return param->beta * -0.5 * pow(tmp, -1.5);
  if (tmp > param->c2)
    return param->beta * (-0.5 * pow(tmp, -1.5) *
                          // error in negligible 2nd term fixed 9/30/2015
                          // (1.0 - 0.5*(1.0 +  1.0/(2.0*param->powern)) *
                          (1.0 - (1.0 + 1.0 / (2.0 * param->powern)) * pow(tmp, -param->powern)));
  if (tmp < param->c4) return 0.0;
  if (tmp < param->c3) return -0.5 * param->beta * pow(tmp, param->powern - 1.0);

  const double tmp_n = pow(tmp, param->powern);
  return -0.5 * pow(1.0 + tmp_n, -1.0 - (1.0 / (2.0 * param->powern))) * tmp_n / zeta;
}

inline void LAMMPS_NS::PairTersoff::repulsive(Param *param, double rsq, double &fforce,
                                              int eflag, double &eng)
{
  const double r = sqrt(rsq);
  const double tmp_fc = ters_fc(r, param);
  const double tmp_fc_d = ters_fc_d(r, param);
  const double tmp_exp = exp(-param->lam1 * r);
  fforce = -param->biga * tmp_exp * (tmp_fc_d - tmp_fc * param->lam1) / r;
  if (eflag) eng = tmp_fc * param->biga * tmp_exp;
}

inline void LAMMPS_NS::PairTersoff::force_zeta(Param *param, double rsq, double zeta_ij,
                                               double &fforce, double &prefactor, int eflag,
                                               double &eng)
{
  const double r = sqrt(rsq);
  const double fa = ters_fa(r, param);
  const double fa_d = ters_fa_d(r, param);
  const double bij = ters_bij(zeta_ij, param);
  fforce = 0.5 * bij * fa_d;
  prefactor = -0.5 * fa * ters_bij_d(zeta_ij, param);
  if (eflag) eng = 0.5 * bij * fa;
}

// attractive term
// use param_ij cutoff for rij test
// use param_ijk cutoff for rik test
inline void LAMMPS_NS::PairTersoff::attractive(Param *param, double prefactor, double rsqij,
                                               double rsqik, double *rij_hat, double *rik_hat,
                                               double *fi, double *fj, double *fk)
{
  const double rij = sqrt(rsqij);
  const double rik = sqrt(rsqik);

  // correct 1/r for shift in rsq
  double rijinv, rikinv;
  if (shift_flag == 1) {
    rijinv = 1.0 / (rij - shift);
    rikinv = 1.0 / (rik - shift);
  } else {
    rijinv = 1.0 / rij;
    rikinv = 1.0 / rik;
  }

  ters_zetaterm_d(prefactor, rij_hat, rij, rijinv, rik_hat, rik, rikinv, fi, fj, fk, param);
}

#endif
#endif
