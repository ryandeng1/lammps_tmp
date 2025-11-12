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

  inline __attribute
  void PairTersoff::costheta_d(double *rij_hat, double rijinv,
                             double *rik_hat, double rikinv,
                             double *dri, double *drj, double *drk)
  {
    // first element is devative wrt Ri, second wrt Rj, third wrt Rk

    double cos_theta = dot3(rij_hat,rik_hat);

    scaleadd3(-cos_theta,rij_hat,rik_hat,drj);
    scale3(rijinv,drj);
    scaleadd3(-cos_theta,rik_hat,rij_hat,drk);
    scale3(rikinv,drk,drk);
    add3(drj,drk,dri);
    scale3(-1.0,dri);
  }


  // inlined functions for efficiency

  inline __attribute__((always_inline))
  double ters_gijk(const double costheta, const Param *const param) const
  {
    const double ters_c = param->c * param->c;
    const double ters_d = param->d * param->d;
    const double hcth = param->h - costheta;

    return param->gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));
  }

  inline __attribute__((always_inline))
  double ters_gijk_d(const double costheta, const Param *const param) const
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

inline __attribute__((always_inline))
double LAMMPS_NS::PairTersoff::ters_fc(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R - ters_D) return 1.0;
  if (r > ters_R + ters_D) return 0.0;
  return 0.5 * (1.0 - sin(MathConst::MY_PI2 * (r - ters_R) / ters_D));
}

inline __attribute__((always_inline))
double LAMMPS_NS::PairTersoff::ters_fc_d(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R - ters_D) return 0.0;
  if (r > ters_R + ters_D) return 0.0;
  return -(MathConst::MY_PI4 / ters_D) * cos(MathConst::MY_PI2 * (r - ters_R) / ters_D);
}

inline __attribute__((always_inline))
double LAMMPS_NS::PairTersoff::ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r) * ters_fc(r, param);
}

inline __attribute__((always_inline))
double LAMMPS_NS::PairTersoff::ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) *
         (param->lam2 * ters_fc(r, param) - ters_fc_d(r, param));
}

inline __attribute__((always_inline))
double LAMMPS_NS::PairTersoff::ters_bij(double zeta, Param *param)
{
  const double tmp = param->beta * zeta;
  if (tmp > param->c1) return 1.0 / sqrt(tmp);
  if (tmp > param->c2)
    return (1.0 - pow(tmp, -param->powern) / (2.0 * param->powern)) / sqrt(tmp);
  if (tmp < param->c4) return 1.0;
  if (tmp < param->c3) return 1.0 - pow(tmp, param->powern) / (2.0 * param->powern);
  return pow(1.0 + pow(tmp, param->powern), -1.0 / (2.0 * param->powern));
}

inline __attribute__((always_inline))
double LAMMPS_NS::PairTersoff::ters_bij_d(double zeta, Param *param)
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

inline __attribute__((always_inline))
void LAMMPS_NS::PairTersoff::repulsive(Param *param, double rsq, double &fforce,
                                              int eflag, double &eng)
{
  const double r = sqrt(rsq);
  const double tmp_fc = ters_fc(r, param);
  const double tmp_fc_d = ters_fc_d(r, param);
  const double tmp_exp = exp(-param->lam1 * r);
  fforce = -param->biga * tmp_exp * (tmp_fc_d - tmp_fc * param->lam1) / r;
  if (eflag) eng = tmp_fc * param->biga * tmp_exp;
}

inline __attribute__((always_inline))
void LAMMPS_NS::PairTersoff::force_zeta(Param *param, double rsq, double zeta_ij,
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
inline __attribute__((always_inline))
void LAMMPS_NS::PairTersoff::attractive(Param *param, double prefactor, double rsqij,
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

inline __attribute((always_inline))
void PairTersoff::ters_zetaterm_d(double prefactor,
                                  double *rij_hat, double rij, double rijinv,
                                  double *rik_hat, double rik, double rikinv,
                                  double *dri, double *drj, double *drk,
                                  Param *param)
{
  double gijk,gijk_d,ex_delr,ex_delr_d,fc,dfc,cos_theta,tmp;
  double dcosdri[3],dcosdrj[3],dcosdrk[3];

  fc = ters_fc(rik,param);
  dfc = ters_fc_d(rik,param);
  if (param->powermint == 3) tmp = cube(param->lam3 * (rij-rik));
  else tmp = param->lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (param->powermint == 3)
    ex_delr_d = 3.0*cube(param->lam3) * square(rij-rik)*ex_delr;
  else ex_delr_d = param->lam3 * ex_delr;

  cos_theta = dot3(rij_hat,rik_hat);
  gijk = ters_gijk(cos_theta,param);
  gijk_d = ters_gijk_d(cos_theta,param);
  costheta_d(rij_hat,rijinv,rik_hat,rikinv,dcosdri,dcosdrj,dcosdrk);

  // compute the derivative wrt Ri
  // dri = -dfc*gijk*ex_delr*rik_hat;
  // dri += fc*gijk_d*ex_delr*dcosdri;
  // dri += fc*gijk*ex_delr_d*(rik_hat - rij_hat);

  scale3(-dfc*gijk*ex_delr,rik_hat,dri);
  scaleadd3(fc*gijk_d*ex_delr,dcosdri,dri,dri);
  scaleadd3(fc*gijk*ex_delr_d,rik_hat,dri,dri);
  scaleadd3(-fc*gijk*ex_delr_d,rij_hat,dri,dri);
  scale3(prefactor,dri);

  // compute the derivative wrt Rj
  // drj = fc*gijk_d*ex_delr*dcosdrj;
  // drj += fc*gijk*ex_delr_d*rij_hat;

  scale3(fc*gijk_d*ex_delr,dcosdrj,drj);
  scaleadd3(fc*gijk*ex_delr_d,rij_hat,drj,drj);
  scale3(prefactor,drj);

  // compute the derivative wrt Rk
  // drk = dfc*gijk*ex_delr*rik_hat;
  // drk += fc*gijk_d*ex_delr*dcosdrk;
  // drk += -fc*gijk*ex_delr_d*rik_hat;

  scale3(dfc*gijk*ex_delr,rik_hat,drk);
  scaleadd3(fc*gijk_d*ex_delr,dcosdrk,drk,drk);
  scaleadd3(-fc*gijk*ex_delr_d,rik_hat,drk,drk);
  scale3(prefactor,drk);
}

inline __attribute((always_inline))
double PairTersoff::zeta(Param *param, double rsqij, double rsqik,
                         double *rij_hat, double *rik_hat)
{
  double rij,rik,costheta,arg,ex_delr;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);
  costheta = dot3(rij_hat,rik_hat);

  if (param->powermint == 3) arg = cube(param->lam3 * (rij-rik));
  else arg = param->lam3 * (rij-rik);

  if (arg > 69.0776) ex_delr = 1.e30;
  else if (arg < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(arg);

  return ters_fc(rik,param) * ters_gijk(costheta,param) * ex_delr;
}



#endif
#endif
