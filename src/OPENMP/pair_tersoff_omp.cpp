// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_tersoff_omp.h"

#include "atom.h"
#include "comm.h"
#include "math_extra.h"
#include "memory.h"
#include "neigh_list.h"
#include "suffix.h"

#include <cmath>

#include "omp_compat.h"
#include "thr_omp.h"
using namespace LAMMPS_NS;
using namespace MathExtra;

/* ---------------------------------------------------------------------- */

PairTersoffOMP::PairTersoffOMP(LAMMPS *lmp) :
  PairTersoff(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

void PairTersoffOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (shift_flag) {
      if (evflag) {
        if (eflag) {
          if (vflag_either) eval<1,1,1,1>(ifrom, ito, thr);
          else eval<1,1,1,0>(ifrom, ito, thr);
        } else {
          if (vflag_either) eval<1,1,0,1>(ifrom, ito, thr);
          else eval<1,1,0,0>(ifrom, ito, thr);
        }
      } else eval<1,0,0,0>(ifrom, ito, thr);

    } else {

      if (evflag) {
        if (eflag) {
          if (vflag_either) eval<0,1,1,1>(ifrom, ito, thr);
          else eval<0,1,1,0>(ifrom, ito, thr);
        } else {
          if (vflag_either) eval<0,1,0,1>(ifrom, ito, thr);
          else eval<0,1,0,0>(ifrom, ito, thr);
        }
      } else eval<0,0,0,0>(ifrom, ito, thr);
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

static int neighshort_thr[10000];

template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_EITHER>
void PairTersoffOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  auto ilist = list->ilist;
  auto numneigh = list->numneigh;
  auto firstneigh = list->firstneigh;
  auto maxshort_thr = maxshort;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const tagint * _noalias const tag = atom->tag;
  const int * _noalias const type = atom->type;
  const int nlocal = atom->nlocal;
  const double cutshortsq = cutmax*cutmax;

  double ters_R = params[0].bigr;
  double ters_D = params[0].bigd;
  double lam1 = params[0].lam1;
  double lam2 = params[0].lam2;
  double lam3 = params[0].lam3;
  double biga = params[0].biga;
  int powermint = params[0].powermint;

  double ters_c = params[0].c * params[0].c;
  double ters_d = params[0].d * params[0].d;
  double h = params[0].h;
  double beta = params[0].beta;
  double gamma = params[0].gamma;

  double bigr = params[0].bigr;
  double bigd = params[0].bigd;
  double bigb = params[0].bigb;

  double c1 = params[0].c1;
  double c2 = params[0].c2;
  double c3 = params[0].c3;
  double c4 = params[0].c4;

  double powern = params[0].powern;

  for (int ii = iifrom; ii < iito; ii++) {
    int i = ilist[ii];
    int itag = tags[i];
    int itype = map[type[i]];
    double xtmp = x[i].x;
    double ytmp = x[i].y;
    double ztmp = x[i].z;
    double fxtmp = 0;
    double fytmp = 0;
    double fztmp = 0;

    auto jlist = firstneigh[i];
    auto jnum = numneigh[i];

    int numshort = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double delx = xtmp - x[j].x;
      double dely = ytmp - x[j].y;
      double delz = ztmp - x[j].z;
      double rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutshortsq) {
        neighshort_thr[numshort++] = j;
        if (numshort >= maxshort_thr) {
          maxshort_thr += maxshort_thr/2;
          // memory->grow(neighshort_thr,maxshort_thr,"pair_thr:neighshort_thr");
        }
      }

      int jtag = tags[j];
      if (itag > jtag) {
          if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
          if ((itag+jtag) % 2 == 1) continue;
      } else {
          if (x[j].z < ztmp) continue;
          if (x[j].z == ztmp && x[j].y < ytmp) continue;
          if (x[j].z == ztmp && x[j].y == ytmp && x[j].x < xtmp) continue;
      }

      int jtype = map[type[j]];
      int ijparam = elem3param[itype][jtype][jtype];
      if (rsq >= params[ijparam].cutsq) continue;

      // repulsive calculation
      double r = sqrt(rsq);
      // ters_fc 
      double ters_fc;
      double ters_fc_d;
      if (r < ters_R - ters_D) {
          ters_fc = 1;
          ters_fc_d = 0;
      } else if (r > ters_R + ters_D) {
          ters_fc = 0;
          ters_fc_d = 0;
      } else {
          ters_fc = 0.5*(1.0 - sin(MathConst::MY_PI2*(r - ters_R)/ters_D));
          ters_fc_d =  -(MathConst::MY_PI4/ters_D) * cos(MathConst::MY_PI2*(r - ters_R)/ters_D);
      }

      double repulsive_exp = exp(-lam1 * r);
      double fpair = -biga * repulsive_exp * (ters_fc_d - ters_fc * lam1) / r;

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;

      f[j].x -= delx*fpair;
      f[j].y -= dely*fpair;
      f[j].z -= delz*fpair;
      // if (evflag) ev_tally(i,j,nlocal,newton_pair, evdwl,0.0,fpair,delx,dely,delz);
    }

    for (int jj = 0; jj < numshort; jj++) {
      int j = neighshort_thr[jj];
      int jtype = map[atom_type[j]];
      int iparam_ij = elem3param[itype][jtype][jtype];
      dbl3_t delr1 = {x[j].x - xtmp, x[j].y - ytmp, x[j].z - ztmp};
      double rsq1 = delr1.x*delr1.x + delr1.y*delr1.y + delr1.z*delr1.z;

      if (rsq1 >= params[iparam_ij].cutsq) {
          continue;
      }

      double r1 = sqrt(rsq1);
      const double r1inv = 1.0/r1;
      dbl3_t r1hat = {r1inv * delr1.x, r1inv * delr1.y, r1inv * delr1.z};

      double fjxtmp = 0;
      double fjytmp = 0;
      double fjztmp = 0;
      double zeta_ij = 0;

      // accumulate bondorder zeta for each i-j interaction via loop over k
      for (int kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        int k = neighshort_thr[kk];
        int ktype = map[atom_type[k]];
        int iparam_ijk = elem3param[itype][jtype][ktype];

        dbl3_t delr2 = {x[k].x - xtmp, x[k].y - ytmp, x[k].z - ztmp};
        double rsq2 = delr2.x*delr2.x + delr2.y*delr2.y + delr2.z*delr2.z;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        double r2 = sqrt(rsq2);
        double r2inv = 1.0/r2;
        dbl3_t r2hat = {r2inv * delr2.x, r2inv * delr2.y, r2inv * delr2.z};

        // zeta calculation
        {
            double costheta = r1hat.x * r2hat.x + r1hat.y * r2hat.y + r1hat.z * r2hat.z;
            double arg;
            if (powermint == 3) {
                double tmp = lam3 * (r1 - r2);
                arg = MathSpecial::cube(tmp);
            } else {
                arg = lam3 * (r1-r2);
            }

            double ex_delr;
            if (arg > 69.0776) ex_delr = 1.e30;
            else if (arg < -69.0776) ex_delr = 0.0;
            else ex_delr = exp(arg);

            double ters_fc_ik;
            if (r2 < ters_R - ters_D) {
                ters_fc_ik = 1;
            } else if (r2 > ters_R + ters_D) {
                ters_fc_ik = 0;
            } else {
                ters_fc_ik = 0.5*(1.0 - sin(MathConst::MY_PI2*(r2 - ters_R)/ters_D));
            }

            double hcth = h - costheta;
            double ters_gijk = gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));
            zeta_ij += ters_fc_ik * ters_gijk * ex_delr;
        }
      }

      // force_zeta

      double fforce;
      double prefactor;

      {
        double ters_fc_r1;
        double ters_fc_r1_d;
        if (r1 < ters_R - ters_D) {
            ters_fc_r1 = 1;
            ters_fc_r1_d = 0;
        } else if (r1 > ters_R + ters_D) {
            ters_fc_r1 = 0;
            ters_fc_r1_d = 0;
        } else {
            ters_fc_r1 = 0.5*(1.0 - sin(MathConst::MY_PI2*(r1 - ters_R)/ters_D));
            ters_fc_r1_d = -(MathConst::MY_PI4/ters_D) * cos(MathConst::MY_PI2*(r1 - ters_R)/ters_D);
        }

        double fa;
        double fa_d;
        if (r1 > bigr + bigd) {
            fa = 0;
            fa_d = 0;
        } else {
            fa = -bigb * exp(-lam2 * r1) * ters_fc_r1;
            fa_d = bigb * exp(-lam2 * r1) * (lam2 * ters_fc_r1 - ters_fc_r1_d);
        }

        double bij;
        double bij_d;
        double ters_bij_tmp = beta * zeta_ij;
        if (ters_bij_tmp > c1) {
            bij = 1.0 / sqrt(ters_bij_tmp);
            bij_d = beta * -0.5*pow(ters_bij_tmp,-1.5);
        } else if (ters_bij_tmp > c2) {
            bij = (1.0 - pow(ters_bij_tmp, -powern) / (2.0*powern))/sqrt(ters_bij_tmp);
            bij_d = beta * (-0.5*pow(ters_bij_tmp,-1.5) *
                // error in negligible 2nd term fixed 9/30/2015
                // (1.0 - 0.5*(1.0 +  1.0/(2.0*param->powern)) *
                (1.0 - (1.0 +  1.0/(2.0*powern)) *
                pow(ters_bij_tmp,-powern)));
        } else if (ters_bij_tmp < c4) {
            bij = 1;
            bij_d = 0;
        } else if (ters_bij_tmp < c3) {
            bij = 1.0 - pow(ters_bij_tmp,powern)/(2.0*powern);
            bij_d = -0.5*beta * pow(ters_bij_tmp,powern-1.0);
        } else {
            double tmp_n = pow(ters_bij_tmp, powern);
            bij = pow(1.0 + tmp_n, -1.0/(2.0*powern));
            bij_d = -0.5 * pow(1.0+tmp_n, -1.0-(1.0/(2.0*powern)))*tmp_n / zeta_ij;
        }

        fforce = 0.5 * bij * fa_d;
        prefactor = -0.5 * fa * bij_d;
      }

      double fpair = fforce * r1inv;
      double delx = delr1.x * fpair;
      double dely = delr1.y * fpair;
      double delz = delr1.z * fpair;

      fxtmp += delx;
      fytmp += dely;
      fztmp += delz;
      fjxtmp -= delx;
      fjytmp -= dely;
      fjztmp -= delz;

      for (int kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        int k = neighshort_thr[kk];
        int ktype = map[type[k]];
        int iparam_ijk = elem3param[itype][jtype][ktype];
        dbl3_t delr2 = {x[k].x - xtmp, x[k].y - ytmp, x[k].z - ztmp};
        double rsq2 = delr2.x*delr2.x + delr2.y*delr2.y + delr2.z*delr2.z;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        double r2 = sqrt(rsq2);
        double r2inv = 1.0/r2;
        dbl3_t r2hat = {r2inv * delr2.x, r2inv * delr2.y, r2inv * delr2.z};

        // attractive calculation/ters_zetaterm_d
        {
          double ters_fc_r2;
          double ters_fc_r2_d;
          if (r2 < ters_R - ters_D) {
              ters_fc_r2 = 1;
              ters_fc_r2_d = 0;
          } else if (r2 > ters_R + ters_D) {
              ters_fc_r2 = 0;
              ters_fc_r2_d = 0;
          } else {
              ters_fc_r2 = 0.5*(1.0 - sin(MathConst::MY_PI2*(r2 - ters_R)/ters_D));
              ters_fc_r2_d = -(MathConst::MY_PI4/ters_D) * cos(MathConst::MY_PI2*(r2 - ters_R)/ters_D);
          }

          double tmp;
          if (powermint == 3) {
              tmp = MathSpecial::cube(lam3 * (r1 - r2));
          } else {
              tmp = lam3 * (r1 - r2);
          }

          double ex_delr;
          double ex_delr_d;

          if (tmp > 69.0776) ex_delr = 1.e30;
          else if (tmp < -69.0776) ex_delr = 0.0;
          else ex_delr = exp(tmp);

          if (powermint == 3) {
              ex_delr_d = 3.0 * MathSpecial::cube(lam3) * MathSpecial::square(r1 - r2) * ex_delr;
          } else {
              ex_delr_d = lam3 * ex_delr;
          }

          double costheta = r1hat.x * r2hat.x + r1hat.y * r2hat.y + r1hat.z * r2hat.z;
          // ters_gijk
          double hcth = h - costheta;
          double ters_gijk = gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));

          // ters_gijk_d
          double numerator = -2.0 * ters_c * hcth;
          double denominator = 1.0 / (ters_d + hcth * hcth);
          double ters_gijk_d = gamma * numerator * denominator * denominator;

          // costheta_d
          double costheta_d;
          dbl3_t dcosdrj = {-costheta * r1hat.x + r2hat.x, -costheta * r1hat.y + r2hat.y, -costheta * r1hat.z + r2hat.z};
          dcosdrj.x *= r1inv;
          dcosdrj.y *= r1inv;
          dcosdrj.z *= r1inv;

          dbl3_t dcosdrk = {-costheta * r2hat.x + r1hat.x, -costheta * r2hat.y + r1hat.y, -costheta * r2hat.z + r1hat.z};
          dcosdrk.x *= r2inv;
          dcosdrk.y *= r2inv;
          dcosdrk.z *= r2inv;

          dbl3_t dcosdri = {dcosdrj.x + dcosdrk.x, dcosdrj.y + dcosdrk.y, dcosdrj.z + dcosdrk.z};
          dcosdri.x *= -1.0;
          dcosdri.y *= -1.0;
          dcosdri.z *= -1.0;

          double scale1 = -ters_fc_r2_d * ters_gijk * ex_delr;
          double scale2 = ters_fc_r2 * ters_gijk_d * ex_delr;
          double scale3 = ters_fc_r2 * ters_gijk * ex_delr_d;
          double scale4 = -scale3;

          dbl3_t dri = {scale1 * r2hat.x, scale1 * r2hat.y, scale1 * r2hat.z};
          dri.x += scale2 * dcosdri.x;
          dri.y += scale2 * dcosdri.y;
          dri.z += scale2 * dcosdri.z;

          dri.x += scale3 * r2hat.x;
          dri.y += scale3 * r2hat.y;
          dri.z += scale3 * r2hat.z;

          dri.x += scale4 * r1hat.x;
          dri.y += scale4 * r1hat.y;
          dri.z += scale4 * r1hat.z;

          dri.x *= prefactor;
          dri.y *= prefactor;
          dri.z *= prefactor;

          dbl3_t drj = {scale2 * dcosdrj.x, scale2 * dcosdrj.y, scale2 * dcosdrj.z};
          drj.x += scale3 * r1hat.x;
          drj.y += scale3 * r1hat.y;
          drj.z += scale3 * r1hat.z;

          drj.x *= prefactor;
          drj.y *= prefactor;
          drj.z *= prefactor;
          
          dbl3_t drk = {-scale1 * r2hat.x, -scale1 * r2hat.y, -scale1 * r2hat.z};
          drk.x += scale2 * dcosdrk.x;
          drk.y += scale2 * dcosdrk.y;
          drk.z += scale2 * dcosdrk.z;

          drk.x += scale4 * r2hat.x;
          drk.y += scale4 * r2hat.y;
          drk.z += scale4 * r2hat.z;

          drk.x *= prefactor;
          drk.y *= prefactor;
          drk.z *= prefactor;

          fxtmp += dri.x;
          fytmp += dri.y;
          fztmp += dri.z;

          fjxtmp += drj.x;
          fjytmp += drj.y;
          fjztmp += drj.z;

          f[k].x += drk.x;
          f[k].y += drk.y;
          f[k].z += drk.z;
        }
      }

      f[j].x += fjxtmp;
      f[j].y += fjytmp;
      f[j].z += fjztmp;
    }

    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }

  return;

  /*
  int i,j,k,ii,jj,kk,jnum,maxshort_thr;
  tagint itag,jtag;
  int itype,jtype,ktype,iparam_ij,iparam_ijk;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double fforce;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fi[3],fj[3],fk[3];
  double r1_hat[3],r2_hat[3];
  double zeta_ij,prefactor;
  double forceshiftfac;
  int *ilist,*jlist,*numneigh,**firstneigh;//,*neighshort_thr;

  evdwl = 0.0;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const tagint * _noalias const tag = atom->tag;
  const int * _noalias const type = atom->type;
  const int nlocal = atom->nlocal;
  const double cutshortsq = cutmax*cutmax;

  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  maxshort_thr = maxshort;
  // memory->create(neighshort_thr,maxshort_thr,"pair_thr:neighshort_thr");

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms

  for (ii = iifrom; ii < iito; ++ii) {

    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;

      // shift rsq and store correction for force

      if (SHIFT_FLAG) {
        double rsqtmp = rsq + shift*shift + 2*sqrt(rsq)*shift;
        forceshiftfac = sqrt(rsqtmp/rsq);
        rsq = rsqtmp;
      }

      if (rsq < cutshortsq) {
        neighshort_thr[numshort++] = j;
        if (numshort >= maxshort_thr) {
          maxshort_thr += maxshort_thr/2;
          // memory->grow(neighshort_thr,maxshort_thr,"pair_thr:neighshort_thr");
        }
      }

      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j].z < ztmp) continue;
        if (x[j].z == ztmp && x[j].y < ytmp) continue;
        if (x[j].z == ztmp && x[j].y == ytmp && x[j].x < xtmp) continue;
      }

      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];
      if (rsq >= params[iparam_ij].cutsq) continue;

      repulsive(&params[iparam_ij],rsq,fpair,EFLAG,evdwl);

      // correct force for shift in rsq

      if (SHIFT_FLAG) fpair *= forceshiftfac;

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;
      f[j].x -= delx*fpair;
      f[j].y -= dely*fpair;
      f[j].z -= delz*fpair;

      if (EVFLAG) ev_tally_thr(this,i,j,nlocal,1,
                               evdwl,0.0,fpair,delx,dely,delz,thr);
    }

    // three-body interactions
    // skip immediately if I-J is not within cutoff
    double fjxtmp,fjytmp,fjztmp;

    for (jj = 0; jj < numshort; jj++) {
      j = neighshort_thr[jj];
      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];

      delr1[0] = x[j].x - xtmp;
      delr1[1] = x[j].y - ytmp;
      delr1[2] = x[j].z - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      if (SHIFT_FLAG)
        rsq1 += shift*shift + 2*sqrt(rsq1)*shift;

      if (rsq1 >= params[iparam_ij].cutsq) continue;

      const double r1inv = 1.0/sqrt(dot3(delr1, delr1));
      scale3(r1inv, delr1, r1_hat);

      // accumulate bondorder zeta for each i-j interaction via loop over k

      fjxtmp = fjytmp = fjztmp = 0.0;
      zeta_ij = 0.0;

      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        k = neighshort_thr[kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];

        delr2[0] = x[k].x - xtmp;
        delr2[1] = x[k].y - ytmp;
        delr2[2] = x[k].z - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        if (SHIFT_FLAG)
          rsq2 += shift*shift + 2*sqrt(rsq2)*shift;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        const double r2inv = 1.0/sqrt(dot3(delr2, delr2));
        scale3(r2inv, delr2, r2_hat);

        zeta_ij += zeta(&params[iparam_ijk],rsq1,rsq2,r1_hat,r2_hat);
      }

      // pairwise force due to zeta

      force_zeta(&params[iparam_ij],rsq1,zeta_ij,fforce,prefactor,EFLAG,evdwl);

      fpair = fforce*r1inv;

      fxtmp += delr1[0]*fpair;
      fytmp += delr1[1]*fpair;
      fztmp += delr1[2]*fpair;
      fjxtmp -= delr1[0]*fpair;
      fjytmp -= delr1[1]*fpair;
      fjztmp -= delr1[2]*fpair;

      if (EVFLAG) ev_tally_thr(this,i,j,nlocal,1,evdwl,0.0,
                               -fpair,-delr1[0],-delr1[1],-delr1[2],thr);

      // attractive term via loop over k

      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        k = neighshort_thr[kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];

        delr2[0] = x[k].x - xtmp;
        delr2[1] = x[k].y - ytmp;
        delr2[2] = x[k].z - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        if (SHIFT_FLAG)
          rsq2 += shift*shift + 2*sqrt(rsq2)*shift;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        const double r2inv = 1.0/sqrt(dot3(delr2, delr2));
        scale3(r2inv, delr2, r2_hat);

        attractive(&params[iparam_ijk],prefactor,
                   rsq1,rsq2,r1_hat,r2_hat,fi,fj,fk);

        fxtmp += fi[0];
        fytmp += fi[1];
        fztmp += fi[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k].x += fk[0];
        f[k].y += fk[1];
        f[k].z += fk[2];

        if (VFLAG_EITHER) v_tally3_thr(this,i,j,k,fj,fk,delr1,delr2,thr);
      }
      f[j].x += fjxtmp;
      f[j].y += fjytmp;
      f[j].z += fjztmp;
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
  memory->destroy(neighshort_thr);
  */
}

/* ---------------------------------------------------------------------- */

double PairTersoffOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairTersoff::memory_usage();

  return bytes;
}
