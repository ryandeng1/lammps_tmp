/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "tokenizer.h"
#include <cilk/cilk.h>

#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;

// peratom variables that are auto-included in corresponding child style field lists
// these fields cannot be specified in the fields strings

const std::vector<std::string> AtomVec::default_grow = {"id", "type", "mask", "image",
                                                        "x",  "v",    "f"};
const std::vector<std::string> AtomVec::default_copy = {"id", "type", "mask", "image", "x", "v"};
const std::vector<std::string> AtomVec::default_comm = {"x"};
const std::vector<std::string> AtomVec::default_comm_vel = {"x", "v"};
const std::vector<std::string> AtomVec::default_reverse = {"f"};
const std::vector<std::string> AtomVec::default_border = {"id", "type", "mask", "x"};
const std::vector<std::string> AtomVec::default_border_vel = {"id", "type", "mask", "x", "v"};
const std::vector<std::string> AtomVec::default_exchange = {"id",    "type", "mask",
                                                            "image", "x",    "v"};
const std::vector<std::string> AtomVec::default_restart = {"id", "type", "mask", "image", "x", "v"};
const std::vector<std::string> AtomVec::default_create = {"id", "type", "mask", "image", "x", "v"};
const std::vector<std::string> AtomVec::default_data_atom = {};
const std::vector<std::string> AtomVec::default_data_vel = {};

/* ---------------------------------------------------------------------- */

AtomVec::AtomVec(LAMMPS *lmp) : Pointers(lmp)
{
  nmax = 0;
  ngrow = 0;

  molecular = Atom::ATOMIC;
  bonds_allow = angles_allow = dihedrals_allow = impropers_allow = 0;
  mass_type = dipole_type = PER_ATOM;
  forceclearflag = 0;
  maxexchange = 0;
  bonus_flag = 0;
  size_forward_bonus = size_border_bonus = 0;

  kokkosable = 0;

  nargcopy = 0;
  argcopy = nullptr;

  tag = nullptr;
  type = mask = nullptr;
  image = nullptr;
  x = v = f = nullptr;

  eval_f_stencil_md = nullptr;

  threads = nullptr;
}

/* ---------------------------------------------------------------------- */

AtomVec::~AtomVec()
{
  int datatype, cols;
  void *pdata;

  for (int i = 0; i < nargcopy; i++) delete[] argcopy[i];
  delete[] argcopy;

  for (int i = 0; i < ngrow; i++) {
    pdata = mgrow.pdata[i];
    datatype = mgrow.datatype[i];
    cols = mgrow.cols[i];
    if (datatype == Atom::DOUBLE) {
      if (cols == 0)
        memory->destroy(*((double **) pdata));
      else if (cols > 0)
        memory->destroy(*((double ***) pdata));
      else {
        memory->destroy(*((double ***) pdata));
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0)
        memory->destroy(*((int **) pdata));
      else if (cols > 0)
        memory->destroy(*((int ***) pdata));
      else {
        memory->destroy(*((int ***) pdata));
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0)
        memory->destroy(*((bigint **) pdata));
      else if (cols > 0)
        memory->destroy(*((bigint ***) pdata));
      else {
        memory->destroy(*((bigint ***) pdata));
      }
    }
  }

  delete[] threads;
}

/* ----------------------------------------------------------------------
   make copy of args for use by restart & replicate
------------------------------------------------------------------------- */

void AtomVec::store_args(int narg, char **arg)
{
  nargcopy = narg;
  if (nargcopy)
    argcopy = new char *[nargcopy];
  else
    argcopy = nullptr;
  for (int i = 0; i < nargcopy; i++) argcopy[i] = utils::strdup(arg[i]);
}

/* ----------------------------------------------------------------------
   no additional args by default
------------------------------------------------------------------------- */

void AtomVec::process_args(int narg, char ** /*arg*/)
{
  if (narg) error->all(FLERR, "Invalid atom_style command");
}

/* ----------------------------------------------------------------------
   pull settings from Domain needed for pack_comm_vel and pack_border_vel
   child classes may override this method, but should also invoke it
------------------------------------------------------------------------- */

void AtomVec::init()
{
  deform_vremap = domain->deform_vremap;
  deform_groupbit = domain->deform_groupbit;
  h_rate = domain->h_rate;

  if (lmp->kokkos != nullptr && !kokkosable)
    error->all(FLERR, "KOKKOS package requires a kokkos enabled atom_style");
}

static constexpr bigint DELTA = 16384;

/* ----------------------------------------------------------------------
   roundup N so it is a multiple of DELTA
   error if N exceeds 32-bit int, since will be used as arg to grow()
------------------------------------------------------------------------- */

bigint AtomVec::roundup(bigint n)
{
  if (n % DELTA) n = n / DELTA * DELTA + DELTA;
  if (n > MAXSMALLINT) error->one(FLERR, "Too many atoms created on one or more procs");
  return n;
}

/* ----------------------------------------------------------------------
   grow nmax so it is a multiple of DELTA
------------------------------------------------------------------------- */

void AtomVec::grow_nmax()
{
  nmax = nmax / DELTA * DELTA;
  nmax += DELTA;
}

static constexpr bigint DELTA_BONUS = 8192;

/* ----------------------------------------------------------------------
   grow nmax_bonus so it is a multiple of DELTA_BONUS
------------------------------------------------------------------------- */

int AtomVec::grow_nmax_bonus(int nmax_bonus)
{
  nmax_bonus = nmax_bonus / DELTA_BONUS * DELTA_BONUS;
  nmax_bonus += DELTA_BONUS;
  return nmax_bonus;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVec::grow(int n)
{
  int datatype, cols, maxcols;
  void *pdata;

  if (n == 0)
    grow_nmax();
  else
    nmax = MAX(n,nmax);
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT) error->one(FLERR, "Per-processor system is too big");

  tag = memory->grow(atom->tag, nmax, "atom:tag");
  type = memory->grow(atom->type, nmax, "atom:type");
  mask = memory->grow(atom->mask, nmax, "atom:mask");
  image = memory->grow(atom->image, nmax, "atom:image");
  x = memory->grow(atom->x, nmax, 3, "atom:x");
  v = memory->grow(atom->v, nmax, 3, "atom:v");
  f = memory->grow(atom->f, nmax * comm->nthreads, 3, "atom:f");

  for (int i = 0; i < ngrow; i++) {
    pdata = mgrow.pdata[i];
    datatype = mgrow.datatype[i];
    cols = mgrow.cols[i];
    const int nthreads = threads[i] ? comm->nthreads : 1;
    if (datatype == Atom::DOUBLE) {
      if (cols == 0)
        memory->grow(*((double **) pdata), nmax * nthreads, "atom:dvec");
      else if (cols > 0)
        memory->grow(*((double ***) pdata), nmax * nthreads, cols, "atom:darray");
      else {
        maxcols = *(mgrow.maxcols[i]);
        memory->grow(*((double ***) pdata), nmax * nthreads, maxcols, "atom:darray");
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0)
        memory->grow(*((int **) pdata), nmax * nthreads, "atom:ivec");
      else if (cols > 0)
        memory->grow(*((int ***) pdata), nmax * nthreads, cols, "atom:iarray");
      else {
        maxcols = *(mgrow.maxcols[i]);
        memory->grow(*((int ***) pdata), nmax * nthreads, maxcols, "atom:iarray");
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0)
        memory->grow(*((bigint **) pdata), nmax * nthreads, "atom:bvec");
      else if (cols > 0)
        memory->grow(*((bigint ***) pdata), nmax * nthreads, cols, "atom:barray");
      else {
        maxcols = *(mgrow.maxcols[i]);
        memory->grow(*((bigint ***) pdata), nmax * nthreads, maxcols, "atom:barray");
      }
    }
  }

  for (int iextra = 0; iextra < atom->nextra_grow; iextra++) {
      assert(false);
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
  }

  grow_pointers();
}

void AtomVec::grow_stencil_md(int n, Atom* atom_)
{
    int datatype, cols, maxcols;
    void *pdata;

    if (n == 0)
        grow_nmax();
    else
        nmax = MAX(n,nmax);
    atom_->nmax = nmax;
    if (nmax < 0 || nmax > MAXSMALLINT) error->one(FLERR, "Per-processor system is too big");

    tag = memory->grow(atom_->tag, nmax, "atom:tag");
    type = memory->grow(atom_->type, nmax, "atom:type");
    mask = memory->grow(atom_->mask, nmax, "atom:mask");
    image = memory->grow(atom_->image, nmax, "atom:image");
    x = memory->grow(atom_->x, nmax, 3, "atom:x");
    v = memory->grow(atom_->v, nmax, 3, "atom:v");
    f = memory->grow(atom_->f, nmax * comm->nthreads, 3, "atom:f");

    eval_f_stencil_md = memory->grow(atom_->eval_f_stencil_md, nmax * comm->nthreads, 3, "atom:eval_f_stencil_md");

    for (int i = 0; i < ngrow; i++) {
        pdata = mgrow.pdata[i];
        datatype = mgrow.datatype[i];
        cols = mgrow.cols[i];
        const int nthreads = threads[i] ? comm->nthreads : 1;
        if (datatype == Atom::DOUBLE) {
            if (cols == 0)
                memory->grow(*((double **) pdata), nmax * nthreads, "atom:dvec");
            else if (cols > 0)
                memory->grow(*((double ***) pdata), nmax * nthreads, cols, "atom:darray");
            else {
                maxcols = *(mgrow.maxcols[i]);
                memory->grow(*((double ***) pdata), nmax * nthreads, maxcols, "atom:darray");
            }
        } else if (datatype == Atom::INT) {
            if (cols == 0)
                memory->grow(*((int **) pdata), nmax * nthreads, "atom:ivec");
            else if (cols > 0)
                memory->grow(*((int ***) pdata), nmax * nthreads, cols, "atom:iarray");
            else {
                maxcols = *(mgrow.maxcols[i]);
                memory->grow(*((int ***) pdata), nmax * nthreads, maxcols, "atom:iarray");
            }
        } else if (datatype == Atom::BIGINT) {
            if (cols == 0)
                memory->grow(*((bigint **) pdata), nmax * nthreads, "atom:bvec");
            else if (cols > 0)
                memory->grow(*((bigint ***) pdata), nmax * nthreads, cols, "atom:barray");
            else {
                maxcols = *(mgrow.maxcols[i]);
                memory->grow(*((bigint ***) pdata), nmax * nthreads, maxcols, "atom:barray");
            }
        }
    }

    for (int iextra = 0; iextra < atom_->nextra_grow; iextra++) {
        assert(false);
        modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
    }

    grow_pointers_stencil_md(atom_);
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVec::copy_with_force(int i, int j, int delflag)
{
    int m, n, datatype, cols, collength, ncols;
    void *pdata, *plength;

    tag[j] = tag[i];
    type[j] = type[i];
    mask[j] = mask[i];
    image[j] = image[i];
    x[j][0] = x[i][0];
    x[j][1] = x[i][1];
    x[j][2] = x[i][2];
    v[j][0] = v[i][0];
    v[j][1] = v[i][1];
    v[j][2] = v[i][2];
    f[j][0] = f[i][0];
    f[j][1] = f[i][1];
    f[j][2] = f[i][2];

    if (ncopy) {
        for (n = 0; n < ncopy; n++) {
            pdata = mcopy.pdata[n];
            datatype = mcopy.datatype[n];
            cols = mcopy.cols[n];
            if (datatype == Atom::DOUBLE) {
                if (cols == 0) {
                    double *vec = *((double **) pdata);
                    vec[j] = vec[i];
                } else if (cols > 0) {
                    double **array = *((double ***) pdata);
                    for (m = 0; m < cols; m++) array[j][m] = array[i][m];
                } else {
                    double **array = *((double ***) pdata);
                    collength = mcopy.collength[n];
                    plength = mcopy.plength[n];
                    if (collength)
                        ncols = (*((int ***) plength))[i][collength - 1];
                    else
                        ncols = (*((int **) plength))[i];
                    for (m = 0; m < ncols; m++) array[j][m] = array[i][m];
                }
            } else if (datatype == Atom::INT) {
                if (cols == 0) {
                    int *vec = *((int **) pdata);
                    vec[j] = vec[i];
                } else if (cols > 0) {
                    int **array = *((int ***) pdata);
                    for (m = 0; m < cols; m++) array[j][m] = array[i][m];
                } else {
                    int **array = *((int ***) pdata);
                    collength = mcopy.collength[n];
                    plength = mcopy.plength[n];
                    if (collength)
                        ncols = (*((int ***) plength))[i][collength - 1];
                    else
                        ncols = (*((int **) plength))[i];
                    for (m = 0; m < ncols; m++) array[j][m] = array[i][m];
                }
            } else if (datatype == Atom::BIGINT) {
                if (cols == 0) {
                    bigint *vec = *((bigint **) pdata);
                    vec[j] = vec[i];
                } else if (cols > 0) {
                    bigint **array = *((bigint ***) pdata);
                    for (m = 0; m < cols; m++) array[j][m] = array[i][m];
                } else {
                    bigint **array = *((bigint ***) pdata);
                    collength = mcopy.collength[n];
                    plength = mcopy.plength[n];
                    if (collength)
                        ncols = (*((int ***) plength))[i][collength - 1];
                    else
                        ncols = (*((int **) plength))[i];
                    for (m = 0; m < ncols; m++) array[j][m] = array[i][m];
                }
            }
        }
    }

    if (bonus_flag) copy_bonus(i, j, delflag);

    if (atom->nextra_grow)
        for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
            modify->fix[atom->extra_grow[iextra]]->copy_arrays(i, j, delflag);
}

void AtomVec::copy(int i, int j, int delflag)
{
  int m, n, datatype, cols, collength, ncols;
  void *pdata, *plength;

  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0];
  x[j][1] = x[i][1];
  x[j][2] = x[i][2];
  v[j][0] = v[i][0];
  v[j][1] = v[i][1];
  v[j][2] = v[i][2];

  if (ncopy) {
    for (n = 0; n < ncopy; n++) {
      pdata = mcopy.pdata[n];
      datatype = mcopy.datatype[n];
      cols = mcopy.cols[n];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          vec[j] = vec[i];
        } else if (cols > 0) {
          double **array = *((double ***) pdata);
          for (m = 0; m < cols; m++) array[j][m] = array[i][m];
        } else {
          double **array = *((double ***) pdata);
          collength = mcopy.collength[n];
          plength = mcopy.plength[n];
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          for (m = 0; m < ncols; m++) array[j][m] = array[i][m];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          vec[j] = vec[i];
        } else if (cols > 0) {
          int **array = *((int ***) pdata);
          for (m = 0; m < cols; m++) array[j][m] = array[i][m];
        } else {
          int **array = *((int ***) pdata);
          collength = mcopy.collength[n];
          plength = mcopy.plength[n];
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          for (m = 0; m < ncols; m++) array[j][m] = array[i][m];
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          vec[j] = vec[i];
        } else if (cols > 0) {
          bigint **array = *((bigint ***) pdata);
          for (m = 0; m < cols; m++) array[j][m] = array[i][m];
        } else {
          bigint **array = *((bigint ***) pdata);
          collength = mcopy.collength[n];
          plength = mcopy.plength[n];
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          for (m = 0; m < ncols; m++) array[j][m] = array[i][m];
        }
      }
    }
  }

  if (bonus_flag) copy_bonus(i, j, delflag);

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i, j, delflag);
}

/* ---------------------------------------------------------------------- */

int AtomVec::pack_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m, mm, nn, datatype, cols;
  double dx, dy, dz;
  void *pdata;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
    }
  }

  if (ncomm) {
    for (nn = 0; nn < ncomm; nn++) {
      pdata = mcomm.pdata[nn];
      datatype = mcomm.datatype[nn];
      cols = mcomm.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = vec[j];
          }
        } else {
          double **array = *((double ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = array[j][mm];
          }
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          int **array = *((int ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      }
    }
  }

  if (bonus_flag) m += pack_comm_bonus(n, list, &buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVec::pack_comm_vel(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m, mm, nn, datatype, cols;
  double dx, dy, dz, dvx, dvy, dvz;
  void *pdata;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
      }
    } else {
      dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
      dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
      dvz = pbc[2] * h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
      }
    }
  }

  if (ncomm_vel) {
    for (nn = 0; nn < ncomm_vel; nn++) {
      pdata = mcomm_vel.pdata[nn];
      datatype = mcomm_vel.datatype[nn];
      cols = mcomm_vel.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = vec[j];
          }
        } else {
          double **array = *((double ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = array[j][mm];
          }
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          int **array = *((int ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      }
    }
  }

  if (bonus_flag) m += pack_comm_bonus(n, list, &buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVec::unpack_comm(int n, int first, double *buf)
{
  int i, m, last, mm, nn, datatype, cols;
  void *pdata;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
  }

  if (ncomm) {
    for (nn = 0; nn < ncomm; nn++) {
      pdata = mcomm.pdata[nn];
      datatype = mcomm.datatype[nn];
      cols = mcomm.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = first; i < last; i++) vec[i] = buf[m++];
        } else {
          double **array = *((double ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = buf[m++];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = first; i < last; i++) vec[i] = (int) ubuf(buf[m++]).i;
        } else {
          int **array = *((int ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (int) ubuf(buf[m++]).i;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = first; i < last; i++) vec[i] = (bigint) ubuf(buf[m++]).i;
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (bigint) ubuf(buf[m++]).i;
        }
      }
    }
  }

  if (bonus_flag) unpack_comm_bonus(n, first, &buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVec::unpack_comm_vel(int n, int first, double *buf)
{
  int i, m, last, mm, nn, datatype, cols;
  void *pdata;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
  }

  if (ncomm_vel) {
    for (nn = 0; nn < ncomm_vel; nn++) {
      pdata = mcomm_vel.pdata[nn];
      datatype = mcomm_vel.datatype[nn];
      cols = mcomm_vel.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = first; i < last; i++) vec[i] = buf[m++];
        } else {
          double **array = *((double ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = buf[m++];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = first; i < last; i++) vec[i] = (int) ubuf(buf[m++]).i;
        } else {
          int **array = *((int ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (int) ubuf(buf[m++]).i;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = first; i < last; i++) vec[i] = (bigint) ubuf(buf[m++]).i;
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (bigint) ubuf(buf[m++]).i;
        }
      }
    }
  }

  if (bonus_flag) unpack_comm_bonus(n, first, &buf[m]);
}

/* ---------------------------------------------------------------------- */

int AtomVec::pack_reverse(int n, int first, double *buf)
{
  int i, m, last, mm, nn, datatype, cols;
  void *pdata;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }

  if (nreverse) {
    for (nn = 0; nn < nreverse; nn++) {
      pdata = mreverse.pdata[nn];
      datatype = mreverse.datatype[nn];
      cols = mreverse.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = first; i < last; i++) { buf[m++] = vec[i]; }
        } else {
          double **array = *((double ***) pdata);
          for (i = first; i < last; i++) {
            for (mm = 0; mm < cols; mm++) buf[m++] = array[i][mm];
          }
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = first; i < last; i++) { buf[m++] = ubuf(vec[i]).d; }
        } else {
          int **array = *((int ***) pdata);
          for (i = first; i < last; i++) {
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
          }
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = first; i < last; i++) { buf[m++] = ubuf(vec[i]).d; }
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = first; i < last; i++) {
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
          }
        }
      }
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVec::unpack_reverse(int n, int *list, double *buf)
{
  int i, j, m, mm, nn, datatype, cols;
  void *pdata;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }

  if (nreverse) {
    for (nn = 0; nn < nreverse; nn++) {
      pdata = mreverse.pdata[nn];
      datatype = mreverse.datatype[nn];
      cols = mreverse.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            vec[j] += buf[m++];
          }
        } else {
          double **array = *((double ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) array[j][mm] += buf[m++];
          }
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            vec[j] += (int) ubuf(buf[m++]).i;
          }
        } else {
          int **array = *((int ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) array[j][mm] += (int) ubuf(buf[m++]).i;
          }
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            vec[j] += (bigint) ubuf(buf[m++]).i;
          }
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) array[j][mm] += (bigint) ubuf(buf[m++]).i;
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int AtomVec::pack_border(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m, mm, nn, datatype, cols;
  double dx, dy, dz;
  void *pdata;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
    }
  }

  if (nborder) {
    for (nn = 0; nn < nborder; nn++) {
      pdata = mborder.pdata[nn];
      datatype = mborder.datatype[nn];
      cols = mborder.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = vec[j];
          }
        } else {
          double **array = *((double ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = array[j][mm];
          }
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          int **array = *((int ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      }
    }
  }

  if (bonus_flag) m += pack_border_bonus(n, list, &buf[m]);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n, list, &buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVec::pack_border_vel(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m, mm, nn, datatype, cols;
  double dx, dy, dz, dvx, dvy, dvz;
  void *pdata;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
      }
    } else {
      dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
      dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
      dvz = pbc[2] * h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
      }
    }
  }

  if (nborder_vel) {
    for (nn = 0; nn < nborder_vel; nn++) {
      pdata = mborder_vel.pdata[nn];
      datatype = mborder_vel.datatype[nn];
      cols = mborder_vel.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = vec[j];
          }
        } else {
          double **array = *((double ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = array[j][mm];
          }
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          int **array = *((int ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            buf[m++] = ubuf(vec[j]).d;
          }
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = 0; i < n; i++) {
            j = list[i];
            for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[j][mm]).d;
          }
        }
      }
    }
  }

  if (bonus_flag) m += pack_border_bonus(n, list, &buf[m]);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n, list, &buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVec::unpack_border(int n, int first, double *buf)
{
  int i, m, last, mm, nn, datatype, cols;
  void *pdata;

  m = 0;
  last = first + n;
  while (last > nmax) grow(0);

  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
  }

  if (nborder) {
    for (nn = 0; nn < nborder; nn++) {
      pdata = mborder.pdata[nn];
      datatype = mborder.datatype[nn];
      cols = mborder.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = first; i < last; i++) vec[i] = buf[m++];
        } else {
          double **array = *((double ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = buf[m++];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = first; i < last; i++) vec[i] = (int) ubuf(buf[m++]).i;
        } else {
          int **array = *((int ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (int) ubuf(buf[m++]).i;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = first; i < last; i++) vec[i] = (bigint) ubuf(buf[m++]).i;
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (bigint) ubuf(buf[m++]).i;
        }
      }
    }
  }

  if (bonus_flag) m += unpack_border_bonus(n, first, &buf[m]);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
}

int AtomVec::unpack_border_stencil_md(double* buf, Atom* atom_) {
    int last_idx;
    last_idx = atom_->nlocal + atom_->nghost;

    if (last_idx == nmax || atom_->nlocal == 0) {
        grow_stencil_md(0, atom_);
    }

    int m = 1;
    double x0 = buf[m++];
    double x1 = buf[m++];
    double x2 = buf[m++];

    tagint tag_ = (tagint) ubuf(buf[m++]).i;
    int type_ = (int) ubuf(buf[m++]).i;
    int mask_ = (int) ubuf(buf[m++]).i;
    int image_ = (imageint) ubuf(buf[m++]).i;

    bool found_tag = false;
    /*
    for (int i = 0; i < last_idx; i++) {
        if (atom_->tag[i] == tag_) {
            found_tag = true;
            break;
        }
    }
    */

    assert(!found_tag);

    if (!found_tag) {
        x[last_idx][0] = x0;
        x[last_idx][1] = x1;
        x[last_idx][2] = x2;

        tag[last_idx] = tag_;
        type[last_idx] = type_;
        mask[last_idx] = mask_;
        image[last_idx] = image_;

        if (nborder) {
            for (int nn = 0; nn < nborder; nn++) {
                void *pdata = mborder.pdata[nn];
                int datatype = mborder.datatype[nn];
                int cols = mborder.cols[nn];
                if (datatype == Atom::DOUBLE) {
                    if (cols == 0) {
                        double *vec = *((double **) pdata);
                        vec[last_idx] = buf[m++];
                    } else {
                        double **array = *((double ***) pdata);
                        for (int mm = 0; mm < cols; mm++) array[last_idx][mm] = buf[m++];
                    }
                } else if (datatype == Atom::INT) {
                    if (cols == 0) {
                        int *vec = *((int **) pdata);
                        vec[last_idx] = (int) ubuf(buf[m++]).i;
                    } else {
                        int **array = *((int ***) pdata);
                        for (int mm = 0; mm < cols; mm++) {
                            array[last_idx][mm] = (int) ubuf(buf[m++]).i;
                        }
                    }
                } else if (datatype == Atom::BIGINT) {
                    if (cols == 0) {
                        bigint *vec = *((bigint **) pdata);
                        vec[last_idx] = (bigint) ubuf(buf[m++]).i;
                    } else {
                        bigint **array = *((bigint ***) pdata);
                        for (int mm = 0; mm < cols; mm++) array[last_idx][mm] = (bigint) ubuf(buf[m++]).i;
                    }
                }
            }
        }

        atom_->nghost++;
    } else {
        assert(false);
        if (nborder) {
            for (int nn = 0; nn < nborder; nn++) {
                void *pdata = mborder.pdata[nn];
                int datatype = mborder.datatype[nn];
                int cols = mborder.cols[nn];
                if (datatype == Atom::DOUBLE) {
                    if (cols == 0) {
                        m++;
                    } else {
                        for (int mm = 0; mm < cols; mm++) {
                            m++;
                        }
                    }
                } else if (datatype == Atom::INT) {
                    if (cols == 0) {
                        m++;
                    } else {
                        for (int mm = 0; mm < cols; mm++) {
                            m++;
                        }
                    }
                } else if (datatype == Atom::BIGINT) {
                    if (cols == 0) {
                        m++;
                    } else {
                        for (int mm = 0; mm < cols; mm++) {
                            m++;
                        }
                    }
                }
            }
        }
    }

    if (bonus_flag) {
        assert(false);
        // m += unpack_border_bonus(n, first, &buf[m]);
    }

    if (atom->nextra_border) {
        assert(false);
        /*
        for (int iextra = 0; iextra < atom->nextra_border; iextra++)
            m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
        */
    }

    assert(m == static_cast<int>(buf[0]));
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVec::unpack_border_vel(int n, int first, double *buf)
{
  int i, m, last, mm, nn, datatype, cols;
  void *pdata;

  m = 0;
  last = first + n;
  while (last > nmax) grow(0);

  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
  }

  if (nborder_vel) {
    for (nn = 0; nn < nborder_vel; nn++) {
      pdata = mborder_vel.pdata[nn];
      datatype = mborder_vel.datatype[nn];
      cols = mborder_vel.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          for (i = first; i < last; i++) vec[i] = buf[m++];
        } else {
          double **array = *((double ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = buf[m++];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          for (i = first; i < last; i++) vec[i] = (int) ubuf(buf[m++]).i;
        } else {
          int **array = *((int ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (int) ubuf(buf[m++]).i;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          for (i = first; i < last; i++) vec[i] = (bigint) ubuf(buf[m++]).i;
        } else {
          bigint **array = *((bigint ***) pdata);
          for (i = first; i < last; i++)
            for (mm = 0; mm < cols; mm++) array[i][mm] = (bigint) ubuf(buf[m++]).i;
        }
      }
    }
  }

  if (bonus_flag) m += unpack_border_bonus(n, first, &buf[m]);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
------------------------------------------------------------------------- */

int AtomVec::pack_exchange(int i, double *buf)
{
  int mm, nn, datatype, cols, collength, ncols;
  void *pdata, *plength;

  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;

  if (nexchange) {
    for (nn = 0; nn < nexchange; nn++) {
      pdata = mexchange.pdata[nn];
      datatype = mexchange.datatype[nn];
      cols = mexchange.cols[nn];
      if (datatype == Atom::DOUBLE) {
        assert(false);
        if (cols == 0) {
          double *vec = *((double **) pdata);
          buf[m++] = vec[i];
        } else if (cols > 0) {
          double **array = *((double ***) pdata);
          for (mm = 0; mm < cols; mm++) buf[m++] = array[i][mm];
        } else {
          double **array = *((double ***) pdata);
          collength = mexchange.collength[nn];
          plength = mexchange.plength[nn];
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          for (mm = 0; mm < ncols; mm++) buf[m++] = array[i][mm];
        }
      }
      if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          buf[m++] = ubuf(vec[i]).d;
        } else if (cols > 0) {
          int **array = *((int ***) pdata);
          for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
        } else {
          int **array = *((int ***) pdata);
          collength = mexchange.collength[nn];
          plength = mexchange.plength[nn];
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          for (mm = 0; mm < ncols; mm++) buf[m++] = ubuf(array[i][mm]).d;
        }
      }
      if (datatype == Atom::BIGINT) {
        assert(false);
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          buf[m++] = ubuf(vec[i]).d;
        } else if (cols > 0) {
          bigint **array = *((bigint ***) pdata);
          for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
        } else {
          bigint **array = *((bigint ***) pdata);
          collength = mexchange.collength[nn];
          plength = mexchange.plength[nn];
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          for (mm = 0; mm < ncols; mm++) buf[m++] = ubuf(array[i][mm]).d;
        }
      }
    }
  }

  if (bonus_flag) {
      assert(false);
      m += pack_exchange_bonus(i, &buf[m]);
  }

  if (atom->nextra_grow) {
      assert(false);
      for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
          m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i, &buf[m]);
  }

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVec::unpack_exchange(double *buf)
{
  std::cout << "LAMMPS unpack exchange" << std::endl;
  int mm, nn, datatype, cols, collength, ncols;
  void *pdata, *plength;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;

  if (nexchange) {
    for (nn = 0; nn < nexchange; nn++) {
      pdata = mexchange.pdata[nn];
      datatype = mexchange.datatype[nn];
      cols = mexchange.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          vec[nlocal] = buf[m++];
        } else if (cols > 0) {
          double **array = *((double ***) pdata);
          for (mm = 0; mm < cols; mm++) array[nlocal][mm] = buf[m++];
        } else {
          double **array = *((double ***) pdata);
          collength = mexchange.collength[nn];
          plength = mexchange.plength[nn];
          if (collength)
            ncols = (*((int ***) plength))[nlocal][collength - 1];
          else
            ncols = (*((int **) plength))[nlocal];
          for (mm = 0; mm < ncols; mm++) array[nlocal][mm] = buf[m++];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          vec[nlocal] = (int) ubuf(buf[m++]).i;
        } else if (cols > 0) {
          int **array = *((int ***) pdata);
          for (mm = 0; mm < cols; mm++) array[nlocal][mm] = (int) ubuf(buf[m++]).i;
        } else {
          int **array = *((int ***) pdata);
          collength = mexchange.collength[nn];
          plength = mexchange.plength[nn];
          if (collength)
            ncols = (*((int ***) plength))[nlocal][collength - 1];
          else
            ncols = (*((int **) plength))[nlocal];
          for (mm = 0; mm < ncols; mm++) array[nlocal][mm] = (int) ubuf(buf[m++]).i;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          vec[nlocal] = (bigint) ubuf(buf[m++]).i;
        } else if (cols > 0) {
          bigint **array = *((bigint ***) pdata);
          for (mm = 0; mm < cols; mm++) array[nlocal][mm] = (bigint) ubuf(buf[m++]).i;
        } else {
          bigint **array = *((bigint ***) pdata);
          collength = mexchange.collength[nn];
          plength = mexchange.plength[nn];
          if (collength)
            ncols = (*((int ***) plength))[nlocal][collength - 1];
          else
            ncols = (*((int **) plength))[nlocal];
          for (mm = 0; mm < ncols; mm++) array[nlocal][mm] = (bigint) ubuf(buf[m++]).i;
        }
      }
    }
  }

  if (bonus_flag) m += unpack_exchange_bonus(nlocal, &buf[m]);

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->unpack_exchange(nlocal, &buf[m]);

  atom->nlocal++;
  return m;
}

int AtomVec::pack_exchange_stencil_md(int i, double *buf, int* pbc) {
    assert(false);
    int m = 1;
    double dx = pbc[0] * domain->prd[0];
    double dy = pbc[1] * domain->prd[1];
    double dz = pbc[2] * domain->prd[2];
    buf[m++] = x[i][0] + dx;
    buf[m++] = x[i][1] + dy;
    buf[m++] = x[i][2] + dz;

    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];

    buf[m++] = ubuf(tag[i]).d;
    buf[m++] = ubuf(type[i]).d;
    buf[m++] = ubuf(mask[i]).d;
    buf[m++] = ubuf(image[i]).d;

    if (USE_BOND) {
        if (nexchange) {
            for (int nn = 0; nn < nexchange; nn++) {
                void* pdata = mexchange.pdata[nn];
                int datatype = mexchange.datatype[nn];
                int cols = mexchange.cols[nn];

                if (datatype == Atom::DOUBLE) {
                    if (cols == 0) {
                        double *vec = *((double **) pdata);
                        buf[m++] = vec[i];
                    } else if (cols > 0) {
                        double **array = *((double ***) pdata);
                        for (int mm = 0; mm < cols; mm++) buf[m++] = array[i][mm];
                    } else {
                        int ncols;
                        double **array = *((double ***) pdata);
                        int collength = mexchange.collength[nn];
                        void* plength = mexchange.plength[nn];
                        if (collength)
                            ncols = (*((int ***) plength))[i][collength - 1];
                        else
                            ncols = (*((int **) plength))[i];
                        for (int mm = 0; mm < ncols; mm++) buf[m++] = array[i][mm];
                    }
                }
                if (datatype == Atom::INT) {
                    if (cols == 0) {
                        int *vec = *((int **) pdata);
                        buf[m++] = ubuf(vec[i]).d;
                    } else if (cols > 0) {
                        int **array = *((int ***) pdata);
                        for (int mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
                    } else {
                        int ncols;
                        int **array = *((int ***) pdata);
                        int collength = mexchange.collength[nn];
                        void* plength = mexchange.plength[nn];
                        if (collength)
                            ncols = (*((int ***) plength))[i][collength - 1];
                        else
                            ncols = (*((int **) plength))[i];
                        for (int mm = 0; mm < ncols; mm++) buf[m++] = ubuf(array[i][mm]).d;
                    }
                }
                if (datatype == Atom::BIGINT) {
                    if (cols == 0) {
                        bigint *vec = *((bigint **) pdata);
                        buf[m++] = ubuf(vec[i]).d;
                    } else if (cols > 0) {
                        bigint **array = *((bigint ***) pdata);
                        for (int mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
                    } else {
                        int ncols;
                        bigint **array = *((bigint ***) pdata);
                        int collength = mexchange.collength[nn];
                        void* plength = mexchange.plength[nn];
                        if (collength)
                            ncols = (*((int ***) plength))[i][collength - 1];
                        else
                            ncols = (*((int **) plength))[i];
                        for (int mm = 0; mm < ncols; mm++) buf[m++] = ubuf(array[i][mm]).d;
                    }
                }
            }
        }

        if (bonus_flag) {
            assert(false);
            m += pack_exchange_bonus(i, &buf[m]);
        }

        if (atom->nextra_grow) {
            assert(false);
            for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
                m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i, &buf[m]);
        }

    }

    buf[0] = m;
    return m;
}

int AtomVec::unpack_exchange_stencil_md(double *buf, Atom *atom_, Domain *domain_,
                                                    int flag) {
    int last_idx;
    if (flag == LAMMPS_SEND_LOCAL) {
        last_idx = atom_->nlocal;
    } else if (flag == LAMMPS_SEND_GHOST) {
        last_idx = atom_->nlocal + atom_->nghost;
    } else {
        assert(false);
    }

    if (last_idx == nmax || atom_->nlocal == 0) {
        grow_stencil_md(0, atom_);
    }

    int m = 1;
    double x0 = buf[m++];
    double x1 = buf[m++];
    double x2 = buf[m++];

    double v0 = buf[m++];
    double v1 = buf[m++];
    double v2 = buf[m++];

    tagint tag_ = (tagint) ubuf(buf[m++]).i;
    int type_ = (int) ubuf(buf[m++]).i;
    int mask_ = (int) ubuf(buf[m++]).i;
    int image_ = (imageint) ubuf(buf[m++]).i;

    bool found_tag = false;
    /*
    for (int i = 0; i < last_idx; i++) {
        if (atom_->tag[i] == tag_) {
            found_tag = true;
            break;
        }
    }
    */
    assert(!found_tag);

    if (!found_tag) {
        x[last_idx][0] = x0;
        x[last_idx][1] = x1;
        x[last_idx][2] = x2;

        v[last_idx][0] = v0;
        v[last_idx][1] = v1;
        v[last_idx][2] = v2;

        tag[last_idx] = tag_;
        type[last_idx] = type_;
        mask[last_idx] = mask_;
        image[last_idx] = image_;

        if (USE_BOND) {
            if (nexchange) {
                for (int nn = 0; nn < nexchange; nn++) {
                    void *pdata = mexchange.pdata[nn];
                    int datatype = mexchange.datatype[nn];
                    int cols = mexchange.cols[nn];
                    if (datatype == Atom::DOUBLE) {
                        if (cols == 0) {
                            double *vec = *((double **) pdata);
                            vec[last_idx] = buf[m++];
                        } else if (cols > 0) {
                            double **array = *((double ***) pdata);
                            for (int mm = 0; mm < cols; mm++) array[last_idx][mm] = buf[m++];
                        } else {
                            int ncols;
                            double **array = *((double ***) pdata);
                            int collength = mexchange.collength[nn];
                            void *plength = mexchange.plength[nn];
                            if (collength)
                                ncols = (*((int ***) plength))[last_idx][collength - 1];
                            else
                                ncols = (*((int **) plength))[last_idx];
                            for (int mm = 0; mm < ncols; mm++) array[last_idx][mm] = buf[m++];
                        }
                    } else if (datatype == Atom::INT) {
                        if (cols == 0) {
                            int *vec = *((int **) pdata);
                            vec[last_idx] = (int) ubuf(buf[m++]).i;
                        } else if (cols > 0) {
                            int **array = *((int ***) pdata);
                            for (int mm = 0; mm < cols; mm++) array[last_idx][mm] = (int) ubuf(buf[m++]).i;
                        } else {
                            int ncols;
                            int **array = *((int ***) pdata);
                            int collength = mexchange.collength[nn];
                            void *plength = mexchange.plength[nn];
                            if (collength)
                                ncols = (*((int ***) plength))[last_idx][collength - 1];
                            else
                                ncols = (*((int **) plength))[last_idx];
                            for (int mm = 0; mm < ncols; mm++) {
                                array[last_idx][mm] = (int) ubuf(buf[m++]).i;
                            }
                        }
                    } else if (datatype == Atom::BIGINT) {
                        if (cols == 0) {
                            bigint *vec = *((bigint **) pdata);
                            vec[last_idx] = (bigint) ubuf(buf[m++]).i;
                        } else if (cols > 0) {
                            bigint **array = *((bigint ***) pdata);
                            for (int mm = 0; mm < cols; mm++) array[last_idx][mm] = (bigint) ubuf(buf[m++]).i;
                        } else {
                            int ncols;
                            bigint **array = *((bigint ***) pdata);
                            int collength = mexchange.collength[nn];
                            void *plength = mexchange.plength[nn];
                            if (collength)
                                ncols = (*((int ***) plength))[last_idx][collength - 1];
                            else
                                ncols = (*((int **) plength))[last_idx];
                            for (int mm = 0; mm < ncols; mm++) array[last_idx][mm] = (bigint) ubuf(buf[m++]).i;
                        }
                    }
                }
            }
        }

        if (bonus_flag) {
            assert(false);
            m += unpack_exchange_bonus(last_idx, &buf[m]);
        }

        if (atom->nextra_grow) {
            assert(false);
            for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
                m += modify->fix[atom->extra_grow[iextra]]->unpack_exchange(last_idx, &buf[m]);
        }

        if (flag == LAMMPS_SEND_LOCAL) {
            atom_->nlocal++;
        } else if (flag == LAMMPS_SEND_GHOST) {
            atom_->nghost++;
        }
    } else {
        // found_tag occurs when sending/receiving ghost atoms
        if (USE_BOND) {
            if (nexchange) {
                for (int nn = 0; nn < nexchange; nn++) {
                    void *pdata = mexchange.pdata[nn];
                    int datatype = mexchange.datatype[nn];
                    int cols = mexchange.cols[nn];
                    if (datatype == Atom::DOUBLE) {
                        if (cols == 0) {
                            double *vec = *((double **) pdata);
                            m++;
                        } else if (cols > 0) {
                            for (int mm = 0; mm < cols; mm++) {
                                m++;
                            }
                        } else {
                            int ncols;
                            int collength = mexchange.collength[nn];
                            void *plength = mexchange.plength[nn];
                            if (collength)
                                ncols = (*((int ***) plength))[last_idx][collength - 1];
                            else
                                ncols = (*((int **) plength))[last_idx];
                            for (int mm = 0; mm < ncols; mm++) {
                                m++;
                            }
                        }
                    } else if (datatype == Atom::INT) {
                        if (cols == 0) {
                            int *vec = *((int **) pdata);
                            m++;
                        } else if (cols > 0) {
                            for (int mm = 0; mm < cols; mm++) {
                                m++;
                            }
                        } else {
                            int ncols;
                            int collength = mexchange.collength[nn];
                            void *plength = mexchange.plength[nn];
                            if (collength)
                                ncols = (*((int ***) plength))[last_idx][collength - 1];
                            else
                                ncols = (*((int **) plength))[last_idx];
                            for (int mm = 0; mm < ncols; mm++) {
                                m++;
                            }
                        }
                    } else if (datatype == Atom::BIGINT) {
                        if (cols == 0) {
                            bigint *vec = *((bigint **) pdata);
                            m++;
                        } else if (cols > 0) {
                            bigint **array = *((bigint ***) pdata);
                            for (int mm = 0; mm < cols; mm++) {
                                m++;
                            }
                        } else {
                            int ncols;
                            bigint **array = *((bigint ***) pdata);
                            int collength = mexchange.collength[nn];
                            void *plength = mexchange.plength[nn];
                            if (collength)
                                ncols = (*((int ***) plength))[last_idx][collength - 1];
                            else
                                ncols = (*((int **) plength))[last_idx];
                            for (int mm = 0; mm < ncols; mm++) {
                                m++;
                            }
                        }
                    }
                }
            }
        }
    }

    return m;
}

int AtomVec::unpack_border_stencil_md(int n, int first, double *buf, Atom *atom_,
                                      int zoid_num) {
    int i, m, last;

    m = 0;
    last = first + n;

    while (last > nmax) { grow_stencil_md(0, atom_); }

    std::set<int> tags;
    for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
        if (tags.find(atom_->tag[idx]) != tags.end()) {
            std::cout << RED << "tag: " << atom_->tag[idx] << " at idx: " << idx << " is a duplicate. out of: " << atom_->nlocal + atom_->nghost << RESET_COLOR << std::endl;
        }
        tags.insert(atom_->tag[idx]);
    }

    if (tags.size() != atom_->nlocal + atom_->nghost) {
        std::cout << RED << "tags size: " << tags.size() << " num atoms: " << atom_->nlocal + atom_->nghost << RESET_COLOR << std::endl;
    }
    assert(tags.size() == atom_->nlocal + atom_->nghost);

    int insert_idx = first;
    std::set<int> inserted_tags;
    for (int j = 0; j < n; j++) {
        double x0 = buf[m++];
        double x1 = buf[m++];
        double x2 = buf[m++];
        tagint tag_ = (tagint) ubuf(buf[m++]).i;
        int type_ = (int) ubuf(buf[m++]).i;
        int mask_ = ubuf(buf[m++]).i;
        double v0 = buf[m++];
        double v1 = buf[m++];
        double v2 = buf[m++];

        if (tag_ < 0 || tag_ > atom->natoms) {
            std::cout << RED << "zoid num recv: " << zoid_num << " tag: " << tag_ << RESET_COLOR << std::endl;
        }
        assert(tag_ >= 0 && tag_ <= atom->natoms);

        if (tags.find(tag_) == tags.end()) {
            if (inserted_tags.find(tag_) != inserted_tags.end()) {
                std::cout << "Tag: " << tag_ << " is repeated. What index? " << insert_idx
                          << " nlocal: " << atom_->nlocal << " nghost? " << atom_->nlocal + atom_->nghost << std::endl;
                assert(false);
            }
            assert(inserted_tags.find(tag_) == inserted_tags.end());

            x[insert_idx][0] = x0;
            x[insert_idx][1] = x1;
            x[insert_idx][2] = x2;

            tag[insert_idx] = tag_;
            type[insert_idx] = type_;
            mask[insert_idx] = mask_;

            v[insert_idx][0] = v0;
            v[insert_idx][1] = v1;
            v[insert_idx][2] = v2;

            inserted_tags.insert(tag_);
            insert_idx++;
        }
    }

    if (atom->nextra_border) {
        assert(false);
        for (int iextra = 0; iextra < atom->nextra_border; iextra++)
            m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
    }

    return insert_idx - first;
}

int AtomVec::pack_data_to_process_stencil_md(int num_zoid_recv, int* zoid_idxs,
                                             int* total_num_elems_send_force, int* num_send_force, int** force_idx_list, int** force_size_list,
                                             int num_segments, bool* segment_types, int* segment_idxs, int* segment_lengths,
                                             int* total_num_elems_send_vel, int* num_send_vel, int** vel_idx_list, int** vel_size_list,
                                             int* local_list, double* buf, int* pbc_flags) {
    if (DEBUG_SEND_RECV_DATA) {
        int m = 0;

        // pack all the force data at the beginning?
        std::vector<int> force_offset_idxs;
        std::vector<int> force_num_segments;

        for (int i = 0; i < num_zoid_recv; i++) {
            int zoid_idx = zoid_idxs[i];
            assert(zoid_idx >= 0 && zoid_idx <= 26);

            int num_send_force_segments = num_send_force[zoid_idx];

            int* force_segment_idxs = force_idx_list[zoid_idx];
            int* force_segment_sizes = force_size_list[zoid_idx];

            for (int j = 0; j < num_send_force_segments; j++) {
                int force_idx = force_segment_idxs[j];
                int force_size = force_segment_sizes[j];

                for (int k = 0; k < force_size; k++) {
                    int idx = force_idx + k;
                    tagint tag_ = tag[idx];
                    buf[m++] = ubuf(tag_).d;
                    buf[m++] = eval_f_stencil_md[idx][0];
                    buf[m++] = eval_f_stencil_md[idx][1];
                    buf[m++] = eval_f_stencil_md[idx][2];

                    eval_f_stencil_md[idx][0] = 0;
                    eval_f_stencil_md[idx][1] = 0;
                    eval_f_stencil_md[idx][2] = 0;
                }
            }

            force_offset_idxs.push_back(m);
            force_num_segments.push_back(num_send_force_segments);
        }

        int pos_start_idx = m;

        int local_list_idx = 0;

        // Note: potentially sending data to middle and pbc, so just send raw positions, have the receiver reinterpret it?
        for (int i = 0; i < num_segments; i++) {
            bool segment_type = segment_types[i];
            int segment_size = segment_lengths[i];
            if (segment_type == SEND_DATA_PROCESS_LOCAL) {
                for (int j = 0; j < segment_size; j++) {
                    int idx = local_list[local_list_idx++];
                    tagint tag_ = tag[idx];
                    // TODO: Kokkos-ify
                    buf[m++] = ubuf(tag_).d;
                    buf[m++] = x[idx][0];
                    buf[m++] = x[idx][1];
                    buf[m++] = x[idx][2];
                }
            } else {
                if (segment_type != SEND_DATA_PROCESS_GHOST) {
                    std::cout << RED << "error segment type: " << segment_type << RESET_COLOR << std::endl;
                }
                assert(segment_type == SEND_DATA_PROCESS_GHOST);
                int segment_idx = segment_idxs[i];
                for (int j = 0; j < segment_size; j++) {
                    int idx = segment_idx + j;
                    // std::cout << "me: " << comm->me << " idx: " << idx << " segment idx: " << segment_idx << std::endl;
                    tagint tag_ = tag[idx];
                    // TODO: Kokkos-ify
                    buf[m++] = ubuf(tag_).d;
                    buf[m++] = x[idx][0];
                    buf[m++] = x[idx][1];
                    buf[m++] = x[idx][2];
                }
            }
        }

        // TODO: only send velocity for atoms that are local to the zoids

        int vel_start_idx = m;

        for (int i = 0; i < num_zoid_recv; i++) {
            int zoid_idx = zoid_idxs[i];
            assert(zoid_idx >= 0 && zoid_idx <= 26);

            int num_send_vel_segments = num_send_vel[zoid_idx];

            int* vel_segment_idxs = vel_idx_list[zoid_idx];
            int* vel_segment_sizes = vel_size_list[zoid_idx];

            for (int j = 0; j < num_send_vel_segments; j++) {
                int vel_idx = vel_segment_idxs[j];
                int vel_size = vel_segment_sizes[j];

                for (int k = 0; k < vel_size; k++) {
                    int idx = vel_idx + k;
                    tagint tag_ = tag[idx];
                    buf[m++] = ubuf(tag_).d;
                    buf[m++] = v[idx][0];
                    buf[m++] = v[idx][1];
                    buf[m++] = v[idx][2];
                }
            }
        }

        /*
        if (pbc_flags != NULL) {
            std::cout << "proc: " << comm->me << " pack pos start: " << pos_start_idx << " vel start: " << vel_start_idx << " end: " << m
                << " force offset idxs: " << force_offset_idxs << " num segments: " << force_num_segments << std::endl;
        }
        */

        // TODO: maybe add this method in Kokkos
        // modified_stencil_md(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK, this);

        return m;
    } else {
        auto * _noalias const x_ = (dbl3_t_stencil_md *) x[0];

        int m = 0;

        int arr_sizes_f[num_zoid_recv];
        arr_sizes_f[0] = 0;
        for (int i = 1; i < num_zoid_recv; i++) {
            arr_sizes_f[i] = arr_sizes_f[i - 1] + total_num_elems_send_force[zoid_idxs[i - 1]] * 3;
        }

        cilk_for (int i = 0; i < num_zoid_recv; i++) {
            int zoid_idx = zoid_idxs[i];
            assert(zoid_idx >= 0 && zoid_idx <= 26);

            int num_send_force_segments = num_send_force[zoid_idx];

            int* force_segment_idxs = force_idx_list[zoid_idx];
            int* force_segment_sizes = force_size_list[zoid_idx];

            int idx = 0;
            for (int j = 0; j < num_send_force_segments; j++) {
                int force_idx = force_segment_idxs[j];
                int force_size = force_segment_sizes[j];

                /*
                for (int k = 0; k < force_size; k++) {
                    int idx = force_idx + k;
                    buf[m++] = eval_f_stencil_md[idx][0];
                    buf[m++] = eval_f_stencil_md[idx][1];
                    buf[m++] = eval_f_stencil_md[idx][2];
                }
                */

                // memcpy(&buf[m], &eval_f_stencil_md[force_idx][0], force_size * sizeof(double) * 3);
                // m += force_size * 3;
                memcpy(&buf[m + arr_sizes_f[i] + idx], &eval_f_stencil_md[force_idx][0], force_size * sizeof(double) * 3);
                memset(&eval_f_stencil_md[force_idx][0], 0, force_size * sizeof(double) * 3);
                idx += force_size * 3;
            }
        }

        for (int i = 0; i < num_zoid_recv; i++) {
            m += total_num_elems_send_force[zoid_idxs[i]] * 3;
        }

        // m += force_size * 3;

        int pos_start_idx = m;

        int local_list_idx = 0;

        // Note: potentially sending data to middle and pbc, so just send raw positions, have the receiver reinterpret it?
        for (int i = 0; i < num_segments; i++) {
            bool segment_type = segment_types[i];
            int segment_size = segment_lengths[i];
            if (segment_type == SEND_DATA_PROCESS_LOCAL) {
                #pragma cilk grainsize 2048
                cilk_for (int j = 0; j < segment_size; j++) {
                    int idx = local_list[local_list_idx + j];
                    // buf[m + j * 3 + 0] = x[idx][0];
                    // buf[m + j * 3 + 1] = x[idx][1];
                    // buf[m + j * 3 + 2] = x[idx][2];
                    buf[m + j * 3 + 0] = x_[idx].x;
                    buf[m + j * 3 + 1] = x_[idx].y;
                    buf[m + j * 3 + 2] = x_[idx].z;
                }
                local_list_idx += segment_size;
                m += 3 * segment_size;
            } else {
                assert(segment_type == SEND_DATA_PROCESS_GHOST);
                int segment_idx = segment_idxs[i];
                /*
                for (int j = 0; j < segment_size; j++) {
                    int idx = segment_idx + j;
                    buf[m++] = x[idx][0];
                    buf[m++] = x[idx][1];
                    buf[m++] = x[idx][2];
                }
                */
                memcpy(&buf[m], &x[segment_idx][0], segment_size * sizeof(double) * 3);
                m += segment_size * 3;
            }
        }

        // TODO: only send velocity for atoms that are local to the zoids

        int vel_start_idx = m;

        int arr_sizes_v[num_zoid_recv];
        arr_sizes_v[0] = 0;
        for (int i = 1; i < num_zoid_recv; i++) {
            arr_sizes_v[i] = arr_sizes_v[i - 1] + total_num_elems_send_vel[zoid_idxs[i - 1]] * 3;
        }

        cilk_for (int i = 0; i < num_zoid_recv; i++) {
            int zoid_idx = zoid_idxs[i];
            assert(zoid_idx >= 0 && zoid_idx <= 26);

            int num_send_vel_segments = num_send_vel[zoid_idx];

            int* vel_segment_idxs = vel_idx_list[zoid_idx];
            int* vel_segment_sizes = vel_size_list[zoid_idx];

            int idx = 0;
            for (int j = 0; j < num_send_vel_segments; j++) {
                int vel_idx = vel_segment_idxs[j];
                int vel_size = vel_segment_sizes[j];

                /*
                for (int k = 0; k < vel_size; k++) {
                    int idx = vel_idx + k;
                    buf[m++] = v[idx][0];
                    buf[m++] = v[idx][1];
                    buf[m++] = v[idx][2];
                }
                */

                // memcpy(&buf[m], &v[vel_idx][0], vel_size * sizeof(double) * 3);
                memcpy(&buf[m + arr_sizes_v[i] + idx], &v[vel_idx][0], vel_size * sizeof(double) * 3);
                idx += vel_size * 3;
                // m += vel_size * 3;
            }
        }

        for (int i = 0; i < num_zoid_recv; i++) {
            m += total_num_elems_send_vel[zoid_idxs[i]] * 3;
        }

        return m;
    }
}

void AtomVec::send_data_bins_stencil_md(int* send_bin_to_idx, int* send_bin_to_size, int* recv_bin_to_idx, int* recv_bin_to_size,
                                        int send_force_num_bins, IDX_3D* send_force_bins,
                                        int send_pos_num_bins, IDX_3D* send_pos_bins,
                                        int send_vel_num_bins, IDX_3D* send_vel_bins,
                                        int recv_force_num_bins, IDX_3D* recv_force_bins,
                                        int recv_pos_num_bins, IDX_3D* recv_pos_bins,
                                        int recv_vel_num_bins, IDX_3D* recv_vel_bins,
                                        tagint* recv_tag, double** recv_f, double** recv_x, double** recv_v, int* pbc_flags) {

    if (DEBUG_SEND_RECV_DATA) {
        for (int i = 0; i < send_force_num_bins; i++) {
            auto send_bin = send_force_bins[i];
            int bin_idx = get_bin_idx(send_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];
            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            if (send_size != recv_bin_to_size[bin_idx]) {
                std::cout << "sizes do not match up send size: " << send_size << " recv bin size: " << recv_bin_to_size[bin_idx] << std::endl;
            }
            assert(send_size == recv_bin_to_size[bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;

                tagint src_tag = tag[send_idx];
                tagint dst_tag = recv_tag[recv_idx];

                assert(src_tag == dst_tag);
                recv_f[recv_idx][0] += eval_f_stencil_md[send_idx][0];
                recv_f[recv_idx][1] += eval_f_stencil_md[send_idx][1];
                recv_f[recv_idx][2] += eval_f_stencil_md[send_idx][2];
            }
        }

        for (int i = 0; i < send_pos_num_bins; i++) {
            auto send_bin = send_pos_bins[i];
            int bin_idx = get_bin_idx(send_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];
            std::cout << "send pos bin: " << std::get<0>(send_bin) << " " << std::get<1>(send_bin) << " " << std::get<2>(send_bin) << " bin idx: " << bin_idx << " send size: " << send_size << " recv size: " << recv_bin_to_size[bin_idx] << " send arr idx: " << send_arr_idx << " recv array idx: " << recv_arr_idx << std::endl;
            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = tag[send_idx];
                tagint dst_tag = recv_tag[recv_idx];
                assert(src_tag == dst_tag);
                recv_x[recv_idx][0] = x[send_idx][0] + pbc_flags[0] * domain->prd[0];
                recv_x[recv_idx][1] = x[send_idx][1] + pbc_flags[1] * domain->prd[1];
                recv_x[recv_idx][2] = x[send_idx][2] + pbc_flags[2] * domain->prd[2];
            }
        }

        for (int i = 0; i < send_vel_num_bins; i++) {
            auto send_bin = send_vel_bins[i];
            int send_bin_idx = get_bin_idx(send_bin);
            int send_size = send_bin_to_size[send_bin_idx];
            int send_arr_idx = send_bin_to_idx[send_bin_idx];
            int recv_arr_idx = recv_bin_to_idx[send_bin_idx];
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[send_bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = tag[send_idx];
                tagint dst_tag = recv_tag[recv_idx];
                assert(src_tag == dst_tag);
                recv_v[recv_idx][0] = v[send_idx][0];
                recv_v[recv_idx][1] = v[send_idx][1];
                recv_v[recv_idx][2] = v[send_idx][2];
            }
        }
    } else {

    }
}

void AtomVec::recv_force_bins_stencil_md(int* send_bin_to_idx, int* send_bin_to_size,
                                         int* recv_bin_to_idx, int* recv_bin_to_size,
                                         int send_force_num_bins, IDX_3D* send_force_bins,
                                         int recv_force_num_bins, IDX_3D* recv_force_bins,
                                         tagint* send_tag, const dbl3_t_stencil_md* _noalias const send_f) {

    auto * _noalias const recv_f_ = (dbl3_t_stencil_md *) f[0];

    if (DEBUG_SEND_RECV_DATA) {
        for (int i = 0; i < recv_force_num_bins; i++) {
            auto recv_bin = recv_force_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = send_tag[send_idx];
                tagint dst_tag = tag[recv_idx];
                if (src_tag != dst_tag) {
                    std::cout << "src tag: " << src_tag << " dst_tag: " << dst_tag << std::endl;
                    std::cout << "bin: " << std::get<0>(recv_bin) << " " << std::get<1>(recv_bin) << " " << std::get<2>(recv_bin) << " bin_idx: " << bin_idx << std::endl;
                }
                assert(src_tag == dst_tag);
                // f[recv_idx][0] += send_f[send_idx][0];
                // f[recv_idx][1] += send_f[send_idx][1];
                // f[recv_idx][2] += send_f[send_idx][2];
                recv_f_[recv_idx].x += send_f[send_idx].x;
                recv_f_[recv_idx].y += send_f[send_idx].y;
                recv_f_[recv_idx].z += send_f[send_idx].z;
            }
        }
    } else {
        // cilk_for (int i = 0; i < recv_force_num_bins; i++) {
        for (int i = 0; i < recv_force_num_bins; i++) {
            auto recv_bin = recv_force_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                recv_f_[recv_idx].x += send_f[send_idx].x;
                recv_f_[recv_idx].y += send_f[send_idx].y;
                recv_f_[recv_idx].z += send_f[send_idx].z;
            }
        }
    }
}

void AtomVec::recv_pos_bins_stencil_md(int* send_bin_to_idx, int* send_bin_to_size,
                                       int* recv_bin_to_idx, int* recv_bin_to_size,
                                       int send_pos_num_bins, IDX_3D* send_pos_bins,
                                       int recv_pos_num_bins, IDX_3D* recv_pos_bins,
                                       tagint* send_tag, const dbl3_t_stencil_md* _noalias const send_x, int* pbc_flags) {

    auto * _noalias const recv_x_ = (dbl3_t_stencil_md *) x[0];

    if (DEBUG_SEND_RECV_DATA) {
        for (int i = 0; i < recv_pos_num_bins; i++) {
            auto recv_bin = recv_pos_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];
            // std::cout << "recv pos bin: " << std::get<0>(send_bin) << " " << std::get<1>(send_bin) << " " << std::get<2>(send_bin) << " bin idx: " << bin_idx << " send size: " << send_size << " recv size: " << recv_bin_to_size[bin_idx] << " send arr idx: " << send_arr_idx << " recv array idx: " << recv_arr_idx << std::endl;
            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            if (send_size != recv_bin_to_size[bin_idx]) {
                std::cout << "bin: " << std::get<0>(recv_bin) << " " << std::get<1>(recv_bin) << " " << std::get<2>(recv_bin)
                          << " bin idx: " << bin_idx << " send size: " << send_size << " recv size: " << recv_bin_to_size[bin_idx] << std::endl;
                for (int j = 0; j < std::max<int>(send_size, recv_bin_to_size[bin_idx]); j++) {
                    int send_idx = send_arr_idx + j;
                    int recv_idx = recv_arr_idx + j;
                    std::cout << "idx: " << j << " send tag: " << send_tag[send_idx] << " send pos: " << send_x[send_idx].x << " " << send_x[send_idx].y << " " << send_x[send_idx].z
                              << " recv tag: " << tag[recv_idx] << " pos: " << x[recv_idx][0] << " " << x[recv_idx][1] << " " << x[recv_idx][2] << std::endl;
                }
            }
            assert(send_size == recv_bin_to_size[bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = send_tag[send_idx];
                tagint dst_tag = tag[recv_idx];
                assert(src_tag == dst_tag);
                // x[recv_idx][0] = send_x[send_idx][0] + pbc_flags[0] * domain->prd[0];
                // x[recv_idx][1] = send_x[send_idx][1] + pbc_flags[1] * domain->prd[1];
                // x[recv_idx][2] = send_x[send_idx][2] + pbc_flags[2] * domain->prd[2];
                recv_x_[recv_idx].x = send_x[send_idx].x + pbc_flags[0] * domain->prd[0];
                recv_x_[recv_idx].y = send_x[send_idx].y + pbc_flags[1] * domain->prd[1];
                recv_x_[recv_idx].z = send_x[send_idx].z + pbc_flags[2] * domain->prd[2];
            }
        }
    } else {
        // cilk_for (int i = 0; i < recv_pos_num_bins; i++) {
        for (int i = 0; i < recv_pos_num_bins; i++) {
            auto recv_bin = recv_pos_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                // x[recv_idx][0] = send_x[send_idx][0] + pbc_flags[0] * domain->prd[0];
                // x[recv_idx][1] = send_x[send_idx][1] + pbc_flags[1] * domain->prd[1];
                // x[recv_idx][2] = send_x[send_idx][2] + pbc_flags[2] * domain->prd[2];
                recv_x_[recv_idx].x = send_x[send_idx].x + pbc_flags[0] * domain->prd[0];
                recv_x_[recv_idx].y = send_x[send_idx].y + pbc_flags[1] * domain->prd[1];
                recv_x_[recv_idx].z = send_x[send_idx].z + pbc_flags[2] * domain->prd[2];
            }
        }
    }
}

void AtomVec::recv_vel_bins_stencil_md(int* send_bin_to_idx, int* send_bin_to_size,
                                       int* recv_bin_to_idx, int* recv_bin_to_size,
                                       int send_vel_num_bins, IDX_3D* send_vel_bins,
                                       int recv_vel_num_bins, IDX_3D* recv_vel_bins,
                                       tagint* send_tag, const dbl3_t_stencil_md* _noalias const send_v) {

    auto * _noalias const recv_v_ = (dbl3_t_stencil_md *) v[0];

    if (DEBUG_SEND_RECV_DATA) {
        for (int i = 0; i < recv_vel_num_bins; i++) {
            auto recv_bin = recv_vel_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = send_tag[send_idx];
                tagint dst_tag = tag[recv_idx];
                assert(src_tag == dst_tag);
                // v[recv_idx][0] = send_v[send_idx][0];
                // v[recv_idx][1] = send_v[send_idx][1];
                // v[recv_idx][2] = send_v[send_idx][2];
                recv_v_[recv_idx].x = send_v[send_idx].x;
                recv_v_[recv_idx].y = send_v[send_idx].y;
                recv_v_[recv_idx].z = send_v[send_idx].z;
            }
        }
    } else {
        // cilk_for (int i = 0; i < recv_vel_num_bins; i++) {
        for (int i = 0; i < recv_vel_num_bins; i++) {
            auto recv_bin = recv_vel_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                // v[recv_idx][0] = send_v[send_idx][0];
                // v[recv_idx][1] = send_v[send_idx][1];
                // v[recv_idx][2] = send_v[send_idx][2];
                recv_v_[recv_idx].x = send_v[send_idx].x;
                recv_v_[recv_idx].y = send_v[send_idx].y;
                recv_v_[recv_idx].z = send_v[send_idx].z;
            }
        }
    }
}

void AtomVec::recv_data_bins_stencil_md(int* send_bin_to_idx, int* send_bin_to_size,
                                        int* recv_bin_to_idx, int* recv_bin_to_size,
                                        int send_force_num_bins, IDX_3D* send_force_bins,
                                        int send_pos_num_bins, IDX_3D* send_pos_bins,
                                        int send_vel_num_bins, IDX_3D* send_vel_bins,
                                        int recv_force_num_bins, IDX_3D* recv_force_bins,
                                        int recv_pos_num_bins, IDX_3D* recv_pos_bins,
                                        int recv_vel_num_bins, IDX_3D* recv_vel_bins,
                                        tagint* send_tag, const dbl3_t_stencil_md* _noalias const send_f,
                                        const dbl3_t_stencil_md* _noalias const send_x,
                                        const dbl3_t_stencil_md* _noalias const send_v, int* pbc_flags) {

    cilk_scope {
        cilk_spawn recv_force_bins_stencil_md(send_bin_to_idx, send_bin_to_size, recv_bin_to_idx, recv_bin_to_size,
                                              send_force_num_bins, send_force_bins, recv_force_num_bins, recv_force_bins,
                                              send_tag, send_f);

        cilk_spawn recv_pos_bins_stencil_md(send_bin_to_idx, send_bin_to_size, recv_bin_to_idx, recv_bin_to_size,
                                            send_pos_num_bins, send_pos_bins, recv_pos_num_bins, recv_pos_bins,
                                            send_tag, send_x, pbc_flags);

        recv_vel_bins_stencil_md(send_bin_to_idx, send_bin_to_size, recv_bin_to_idx, recv_bin_to_size,
                                 send_vel_num_bins, send_vel_bins, recv_vel_num_bins, recv_vel_bins,
                                 send_tag, send_v);
    }

    /*
    if (DEBUG_SEND_RECV_DATA) {
        for (int i = 0; i < recv_force_num_bins; i++) {
            auto recv_bin = recv_force_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = send_tag[send_idx];
                tagint dst_tag = tag[recv_idx];
                if (src_tag != dst_tag) {
                    std::cout << "src tag: " << src_tag << " dst_tag: " << dst_tag << std::endl;
                    std::cout << "bin: " << std::get<0>(recv_bin) << " " << std::get<1>(recv_bin) << " " << std::get<2>(recv_bin) << " bin_idx: " << bin_idx << std::endl;
                }
                assert(src_tag == dst_tag);
                f[recv_idx][0] += send_f[send_idx][0];
                f[recv_idx][1] += send_f[send_idx][1];
                f[recv_idx][2] += send_f[send_idx][2];
            }
        }

        for (int i = 0; i < recv_pos_num_bins; i++) {
            auto recv_bin = recv_pos_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];
            // std::cout << "recv pos bin: " << std::get<0>(send_bin) << " " << std::get<1>(send_bin) << " " << std::get<2>(send_bin) << " bin idx: " << bin_idx << " send size: " << send_size << " recv size: " << recv_bin_to_size[bin_idx] << " send arr idx: " << send_arr_idx << " recv array idx: " << recv_arr_idx << std::endl;
            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            if (send_size != recv_bin_to_size[bin_idx]) {
                std::cout << "bin: " << std::get<0>(recv_bin) << " " << std::get<1>(recv_bin) << " " << std::get<2>(recv_bin)
                    << " bin idx: " << bin_idx << " send size: " << send_size << " recv size: " << recv_bin_to_size[bin_idx] << std::endl;
                for (int j = 0; j < std::max<int>(send_size, recv_bin_to_size[bin_idx]); j++) {
                    int send_idx = send_arr_idx + j;
                    int recv_idx = recv_arr_idx + j;
                    std::cout << "idx: " << j << " send tag: " << send_tag[send_idx] << " send pos: " << send_x[send_idx].x << " " << send_x[send_idx].y << " " << send_x[send_idx].z
                        << " recv tag: " << tag[recv_idx] << " pos: " << x[recv_idx][0] << " " << x[recv_idx][1] << " " << x[recv_idx][2] << std::endl;
                }
            }
            assert(send_size == recv_bin_to_size[bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = send_tag[send_idx];
                tagint dst_tag = tag[recv_idx];
                assert(src_tag == dst_tag);
                x[recv_idx][0] = send_x[send_idx].x + pbc_flags[0] * domain->prd[0];
                x[recv_idx][1] = send_x[send_idx].y + pbc_flags[1] * domain->prd[1];
                x[recv_idx][2] = send_x[send_idx].z + pbc_flags[2] * domain->prd[2];
            }
        }

        for (int i = 0; i < recv_vel_num_bins; i++) {
            auto recv_bin = recv_vel_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);
            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                tagint src_tag = send_tag[send_idx];
                tagint dst_tag = tag[recv_idx];
                assert(src_tag == dst_tag);
                v[recv_idx][0] = send_v[send_idx].x;
                v[recv_idx][1] = send_v[send_idx].y;
                v[recv_idx][2] = send_v[send_idx].z;
            }
        }
    } else {
        // cilk_for (int i = 0; i < recv_force_num_bins; i++) {
        for (int i = 0; i < recv_force_num_bins; i++) {
            auto recv_bin = recv_force_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                f[recv_idx][0] += send_f[send_idx].x;
                f[recv_idx][1] += send_f[send_idx].y;
                f[recv_idx][2] += send_f[send_idx].z;
            }
        }

        // cilk_for (int i = 0; i < recv_pos_num_bins; i++) {
        for (int i = 0; i < recv_pos_num_bins; i++) {
            auto recv_bin = recv_pos_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_arr_idx != -1);
            assert(recv_arr_idx != -1);
            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                x[recv_idx][0] = send_x[send_idx].x + pbc_flags[0] * domain->prd[0];
                x[recv_idx][1] = send_x[send_idx].y + pbc_flags[1] * domain->prd[1];
                x[recv_idx][2] = send_x[send_idx].z + pbc_flags[2] * domain->prd[2];
            }
        }

        // cilk_for (int i = 0; i < recv_vel_num_bins; i++) {
        for (int i = 0; i < recv_vel_num_bins; i++) {
            auto recv_bin = recv_vel_bins[i];
            int bin_idx = get_bin_idx(recv_bin);
            int send_size = send_bin_to_size[bin_idx];
            int send_arr_idx = send_bin_to_idx[bin_idx];
            int recv_arr_idx = recv_bin_to_idx[bin_idx];

            assert(send_size != -1);
            assert(send_size == recv_bin_to_size[bin_idx]);

            for (int j = 0; j < send_size; j++) {
                int send_idx = send_arr_idx + j;
                int recv_idx = recv_arr_idx + j;
                v[recv_idx][0] = send_v[send_idx].x;
                v[recv_idx][1] = send_v[send_idx].y;
                v[recv_idx][2] = send_v[send_idx].z;
            }
        }
    }
    */
}

void AtomVec::unpack_data_from_process_stencil_md(int nrecv_force, int nrecv_pos,
                                         int force_offset_buf, int num_recv_force, int* recv_force_list,
                                         int num_pos_segments_buf, bool* segment_types_buf, int* segment_idxs_buf, int* segment_sizes_buf,
                                         int vel_offset_buf, int num_recv_vel, int* recv_pos_local_list,
                                         int num_recv_ghost, int* recv_ghost_idx_list, int* recv_ghost_size_list,
                                         double* buf, int* pbc_flags) {
    if (DEBUG_SEND_RECV_DATA) {
        // 0 is the starting idx of the buffeer
        int m = 0 + force_offset_buf * (3 + 1);

        for (int i = 0; i < num_recv_force; i++) {
            tagint target_tag = (tagint) ubuf(buf[m++]).i;
            double f_x = buf[m++];
            double f_y = buf[m++];
            double f_z = buf[m++];

            int idx = recv_force_list[i];

            /*
            if (target_tag != tag[idx]) {
                std::cout << RED << "process: " << comm->me << " error recv local force. Received tag: " << target_tag << " but I want tag: " << tag[idx]
                    << " at idx: " << idx << " i: " << i << " out of: " << num_recv_force
                    << " force offset in buf: " << force_offset_buf << RESET_COLOR << std::endl;
            }
            */
            assert(tag[idx] == target_tag);

            f[idx][0] += f_x;
            f[idx][1] += f_y;
            f[idx][2] += f_z;
        }

        int pos_start_idx = nrecv_force * (3 + 1);

        int local_list_idx = 0;

        int ghost_idx = 0;
        int curr_pos_segment = 0;

        for (int i = 0; i < num_pos_segments_buf; i++) {
            bool segment_type = segment_types_buf[i];
            int segment_size = segment_sizes_buf[i];
            int segment_idx = segment_idxs_buf[i];
            int counter = segment_idx * (3 + 1) + pos_start_idx;
            if (segment_type == RECV_DATA_PROCESS_LOCAL) {
                for (int j = 0; j < segment_size; j++) {
                    tagint target_tag = (tagint) ubuf(buf[counter++]).i;
                    double x_x = buf[counter++];
                    double x_y = buf[counter++];
                    double x_z = buf[counter++];

                    int idx = recv_pos_local_list[local_list_idx++];
                    /*
                    if (target_tag != tag[idx]) {
                        std::cout << RED << " error at process: " << comm->me << " recv local pos. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx
                            << " force start buf idx: " << force_offset_buf << " pos start buf idx: " << pos_start_idx << " counter? " << counter
                            << " segment number: " << i << " segment idx: " << segment_idxs_buf[i] << " segment size: " << segment_sizes_buf[i] << RESET_COLOR << std::endl;
                    }
                    */
                    assert(tag[idx] == target_tag);

                    x[idx][0] = x_x + domain->prd[0] * pbc_flags[0];
                    x[idx][1] = x_y + domain->prd[1] * pbc_flags[1];
                    x[idx][2] = x_z + domain->prd[2] * pbc_flags[2];
                }
            } else {
                assert(segment_type == RECV_DATA_PROCESS_GHOST);
                for (int j = 0; j < segment_size; j++) {
                    int buf_idx = counter / 4;
                    tagint target_tag = (tagint) ubuf(buf[counter++]).i;
                    double x_x = buf[counter++];
                    double x_y = buf[counter++];
                    double x_z = buf[counter++];

                    if (ghost_idx >= recv_ghost_size_list[curr_pos_segment]) {
                        ghost_idx = 0;
                        curr_pos_segment++;
                    }

                    int idx = recv_ghost_idx_list[curr_pos_segment] + ghost_idx;
                    /*
                    if (target_tag != tag[idx]) {
                        std::cout << RED << " error at process: " << comm->me << " recv ghost pos. Received tag: " << target_tag << " but I want tag: " << tag[idx]
                                  << " at atom local idx: " << idx << " buf idx: " << buf_idx << " segment idx: " << segment_idx << " segment buf number: " << i
                                  << " force start buf idx: " << force_offset_buf << " pos start buf idx: " << pos_start_idx << " counter? " << counter
                                  << " segment number: " << i << " segment idx: " << segment_idxs_buf[i] << " segment size: " << segment_sizes_buf[i]
                                  << " curr pos segment? " << curr_pos_segment << " ghost idx: " << ghost_idx
                                  << " recv ghost size: " << recv_ghost_size_list[curr_pos_segment] << RESET_COLOR << std::endl;
                    }
                    */
                    assert(tag[idx] == target_tag);

                    // TODO: test if ghost pos actually needed
                    x[idx][0] = x_x + domain->prd[0] * pbc_flags[0];
                    x[idx][1] = x_y + domain->prd[1] * pbc_flags[1];
                    x[idx][2] = x_z + domain->prd[2] * pbc_flags[2];

                    ghost_idx++;
                }
            }
        }

        // TODO: need to send the pos/vel offset?
        int vel_start_idx = nrecv_force * (3 + 1) + nrecv_pos * (3 + 1) + vel_offset_buf * (3 + 1);
        m = vel_start_idx;

        for (int i = 0; i < num_recv_vel; i++) {
            tagint target_tag = (tagint) ubuf(buf[m++]).i;
            double v_x = buf[m++];
            double v_y = buf[m++];
            double v_z = buf[m++];

            int idx = recv_pos_local_list[i];
            /*
            if (target_tag != tag[idx]) {
                std::cout << RED << "process: " << comm->me << " error recv local force. Received tag: " << target_tag << " but I want tag: " << tag[idx]
                          << " at idx: " << idx << " i: " << i << " out of: " << num_recv_force
                          << " force start buf idx: " << force_offset_buf << RESET_COLOR << std::endl;
            }
            */
            assert(tag[idx] == target_tag);

            v[idx][0] = v_x;
            v[idx][1] = v_y;
            v[idx][2] = v_z;
        }
    } else {
        auto * _noalias f_ = (dbl3_t_stencil_md *) f[0];
        auto * _noalias x_ = (dbl3_t_stencil_md *) x[0];
        auto * _noalias v_ = (dbl3_t_stencil_md *) v[0];

        bool is_all_zero = (pbc_flags[0] == 0 && pbc_flags[1] == 0 && pbc_flags[2] == 0);

        // 0 is the starting idx of the buffeer
        int m = 0 + force_offset_buf * (3);

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < num_recv_force; i++) {
            double f_x = buf[m + i * 3];
            double f_y = buf[m + i * 3 + 1];
            double f_z = buf[m + i * 3 + 2];

            int idx = recv_force_list[i];

            // f[idx][0] += f_x;
            // f[idx][1] += f_y;
            // f[idx][2] += f_z;
            f_[idx].x += f_x;
            f_[idx].y += f_y;
            f_[idx].z += f_z;
        }

        m += 3 * num_recv_force;

        int pos_start_idx = nrecv_force * (3);

        int local_list_idx = 0;

        int ghost_idx = 0;
        int curr_pos_segment = 0;

        for (int i = 0; i < num_pos_segments_buf; i++) {
            bool segment_type = segment_types_buf[i];
            int segment_size = segment_sizes_buf[i];
            int segment_idx = segment_idxs_buf[i];
            int counter = segment_idx * (3) + pos_start_idx;
            if (segment_type == RECV_DATA_PROCESS_LOCAL) {
                if (is_all_zero) {
                    for (int j = 0; j < segment_size; j++) {
                        double x_x = buf[counter + j * 3];
                        double x_y = buf[counter + j * 3 + 1];
                        double x_z = buf[counter + j * 3 + 2];

                        int idx = recv_pos_local_list[local_list_idx + j];

                        // x[idx][0] = x_x + domain->prd[0] * pbc_flags[0];
                        // x[idx][1] = x_y + domain->prd[1] * pbc_flags[1];
                        // x[idx][2] = x_z + domain->prd[2] * pbc_flags[2];
                        x_[idx].x = x_x;
                        x_[idx].y = x_y;
                        x_[idx].z = x_z;
                    }
                } else {
                    for (int j = 0; j < segment_size; j++) {

                        double x_x = buf[counter + j * 3];
                        double x_y = buf[counter + j * 3 + 1];
                        double x_z = buf[counter + j * 3 + 2];

                        int idx = recv_pos_local_list[local_list_idx + j];

                        // x[idx][0] = x_x + domain->prd[0] * pbc_flags[0];
                        // x[idx][1] = x_y + domain->prd[1] * pbc_flags[1];
                        // x[idx][2] = x_z + domain->prd[2] * pbc_flags[2];
                        x_[idx].x = x_x + domain->prd[0] * pbc_flags[0];
                        x_[idx].y = x_y + domain->prd[1] * pbc_flags[1];
                        x_[idx].z = x_z + domain->prd[2] * pbc_flags[2];
                    }
                }
                local_list_idx += segment_size;
            } else {
                assert(segment_type == RECV_DATA_PROCESS_GHOST);
                if (is_all_zero) {
                    for (int j = 0; j < segment_size; j++) {
                        int buf_idx = counter / 3;
                        double x_x = buf[counter++];
                        double x_y = buf[counter++];
                        double x_z = buf[counter++];

                        if (ghost_idx >= recv_ghost_size_list[curr_pos_segment]) {
                            ghost_idx = 0;
                            curr_pos_segment++;
                        }

                        int idx = recv_ghost_idx_list[curr_pos_segment] + ghost_idx;

                        // x[idx][0] = x_x + domain->prd[0] * pbc_flags[0];
                        // x[idx][1] = x_y + domain->prd[1] * pbc_flags[1];
                        // x[idx][2] = x_z + domain->prd[2] * pbc_flags[2];

                        x_[idx].x = x_x;
                        x_[idx].y = x_y;
                        x_[idx].z = x_z;

                        ghost_idx++;
                    }
                } else {
                    for (int j = 0; j < segment_size; j++) {
                        int buf_idx = counter / 3;
                        double x_x = buf[counter++];
                        double x_y = buf[counter++];
                        double x_z = buf[counter++];

                        if (ghost_idx >= recv_ghost_size_list[curr_pos_segment]) {
                            ghost_idx = 0;
                            curr_pos_segment++;
                        }

                        int idx = recv_ghost_idx_list[curr_pos_segment] + ghost_idx;

                        // x[idx][0] = x_x + domain->prd[0] * pbc_flags[0];
                        // x[idx][1] = x_y + domain->prd[1] * pbc_flags[1];
                        // x[idx][2] = x_z + domain->prd[2] * pbc_flags[2];

                        x_[idx].x = x_x + domain->prd[0] * pbc_flags[0];
                        x_[idx].y = x_y + domain->prd[1] * pbc_flags[1];
                        x_[idx].z = x_z + domain->prd[2] * pbc_flags[2];

                        ghost_idx++;
                    }
                }
            }
        }

        // TODO: need to send the pos/vel offset?
        int vel_start_idx = nrecv_force * (3) + nrecv_pos * (3) + vel_offset_buf * (3);
        m = vel_start_idx;

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < num_recv_vel; i++) {
            double v_x = buf[m + i * 3];
            double v_y = buf[m + i * 3 + 1];
            double v_z = buf[m + i * 3 + 2];

            int idx = recv_pos_local_list[i];

            // v[idx][0] = v_x;
            // v[idx][1] = v_y;
            // v[idx][2] = v_z;

            v_[idx].x = v_x;
            v_[idx].y = v_y;
            v_[idx].z = v_z;
        }
    }
}

int AtomVec::pack_data_stencil_md(int num_send_force, int num_send_pos,
                             int* force_idx_list, int* force_size_list,
                             int* pos_idx_list, int* pos_size_list,
                             int* local_to_ghost_list,
                             int num_segments, bool* segment_types, int* segment_idxs, int* segment_sizes,
                             double* buf, int* pbc_flags, bool debug) {
  /*
   * Send data.
   * 1. Send send_list data, composed of 2 parts
   *    - forces, stored in send_force
   *    - positions, stored in send_pos
   * 2. Send second_send_list data
   *    - local atoms, stored in second_send_list
   *    - ghost atoms, stored in send_ghost_idxs
   *
   *
   */

  if (DEBUG_SEND_RECV_DATA) {
      int m = 0;

      assert(num_send_force >= 0 && num_send_force <= 100000);
      assert(num_send_pos >= 0 && num_send_pos <= 100000);

      for (int i = 0; i < num_send_force; i++) {
          int force_idx = force_idx_list[i];
          int force_size = force_size_list[i];

          for (int j = 0; j < force_size; j++) {
              int idx = force_idx + j;
              tagint tag_ = tag[idx];
              // TODO: Kokkos-ify
              buf[m++] = ubuf(tag_).d;
              buf[m++] = eval_f_stencil_md[idx][0];
              buf[m++] = eval_f_stencil_md[idx][1];
              buf[m++] = eval_f_stencil_md[idx][2];
          }
      }

      int send_force = m;

      /*
      for (int i = 0; i < num_send_pos; i++) {
          int pos_idx = pos_idx_list[i];
          int pos_size = pos_size_list[i];
          // std::cout << "num send pos? " << num_send_pos << " pos size: " << pos_size << std::endl;
          for (int j = 0; j < pos_size; j++) {
              int idx = pos_idx + j;
              tagint tag_ = tag[idx];
              // TODO: Kokkos-ify
              // TODO: should this even be here?
              buf[m++] = ubuf(h_tag(idx)).d;
              buf[m++] = eval_f_stencil_md[idx][0];
              buf[m++] = eval_f_stencil_md[idx][1];
              buf[m++] = eval_f_stencil_md[idx][2];
          }
      }
      */

      for (int i = 0; i < num_send_pos; i++) {
          int pos_idx = pos_idx_list[i];
          int pos_size = pos_size_list[i];
          // std::cout << "num send pos? " << num_send_pos << " pos size: " << pos_size << std::endl;
          for (int j = 0; j < pos_size; j++) {
              int idx = pos_idx + j;
              tagint tag_ = tag[idx];
              // TODO: Kokkos-ify
              buf[m++] = ubuf(tag[idx]).d;
              buf[m++] = x[idx][0] + pbc_flags[0] * domain->prd[0];
              buf[m++] = x[idx][1] + pbc_flags[1] * domain->prd[1];
              buf[m++] = x[idx][2] + pbc_flags[2] * domain->prd[2];
          }
      }

      for (int i = 0; i < num_send_pos; i++) {
          int pos_idx = pos_idx_list[i];
          int pos_size = pos_size_list[i];
          for (int j = 0; j < pos_size; j++) {
              int idx = pos_idx + j;
              tagint tag_ = tag[idx];
              // TODO: Kokkos-ify
              buf[m++] = ubuf(tag[idx]).d;
              buf[m++] = v[idx][0];
              buf[m++] = v[idx][1];
              buf[m++] = v[idx][2];
          }
      }

      int send_pos = m - send_force;

      int local_list_idx = 0;

      for (int i = 0; i < num_segments; i++) {
          bool segment_type = segment_types[i];
          int segment_size = segment_sizes[i];
          if (segment_type == LOCAL_SEGMENT_TYPE) {
              for (int j = 0; j < segment_size; j++) {
                  int idx = local_to_ghost_list[local_list_idx++];
                  tagint tag_ = tag[idx];
                  // TODO: Kokkos-ify
                  buf[m++] = ubuf(tag_).d;
                  buf[m++] = x[idx][0] + pbc_flags[0] * domain->prd[0];
                  buf[m++] = x[idx][1] + pbc_flags[1] * domain->prd[1];
                  buf[m++] = x[idx][2] + pbc_flags[2] * domain->prd[2];
              }
          } else {
              assert(segment_type == GHOST_SEGMENT_TYPE);
              int segment_idx = segment_idxs[i];
              for (int j = 0; j < segment_size; j++) {
                  int idx = segment_idx + j;
                  tagint tag_ = tag[idx];
                  // TODO: Kokkos-ify
                  buf[m++] = ubuf(tag_).d;
                  buf[m++] = x[idx][0] + pbc_flags[0] * domain->prd[0];
                  buf[m++] = x[idx][1] + pbc_flags[1] * domain->prd[1];
                  buf[m++] = x[idx][2] + pbc_flags[2] * domain->prd[2];
              }
          }
      }

      return m;
  } else {
      int m = 0;

      assert(num_send_force >= 0 && num_send_force <= 100000);
      assert(num_send_pos >= 0 && num_send_pos <= 100000);

      for (int i = 0; i < num_send_force; i++) {
          int force_idx = force_idx_list[i];
          int force_size = force_size_list[i];

          for (int j = 0; j < force_size; j++) {
              int idx = force_idx + j;
              buf[m++] = eval_f_stencil_md[idx][0];
              buf[m++] = eval_f_stencil_md[idx][1];
              buf[m++] = eval_f_stencil_md[idx][2];
          }
      }

      int send_force = m;

      for (int i = 0; i < num_send_pos; i++) {
          int pos_idx = pos_idx_list[i];
          int pos_size = pos_size_list[i];
          // std::cout << "num send pos? " << num_send_pos << " pos size: " << pos_size << std::endl;
          for (int j = 0; j < pos_size; j++) {
              int idx = pos_idx + j;
              buf[m++] = x[idx][0] + pbc_flags[0] * domain->prd[0];
              buf[m++] = x[idx][1] + pbc_flags[1] * domain->prd[1];
              buf[m++] = x[idx][2] + pbc_flags[2] * domain->prd[2];
          }
      }

      for (int i = 0; i < num_send_pos; i++) {
          int pos_idx = pos_idx_list[i];
          int pos_size = pos_size_list[i];
          for (int j = 0; j < pos_size; j++) {
              int idx = pos_idx + j;
              buf[m++] = v[idx][0];
              buf[m++] = v[idx][1];
              buf[m++] = v[idx][2];
          }
      }

      int send_pos = m - send_force;

      int local_list_idx = 0;

      for (int i = 0; i < num_segments; i++) {
          bool segment_type = segment_types[i];
          int segment_size = segment_sizes[i];
          if (segment_type == LOCAL_SEGMENT_TYPE) {
              for (int j = 0; j < segment_size; j++) {
                  int idx = local_to_ghost_list[local_list_idx++];
                  buf[m++] = x[idx][0] + pbc_flags[0] * domain->prd[0];
                  buf[m++] = x[idx][1] + pbc_flags[1] * domain->prd[1];
                  buf[m++] = x[idx][2] + pbc_flags[2] * domain->prd[2];
              }
          } else {
              assert(segment_type == GHOST_SEGMENT_TYPE);
              int segment_idx = segment_idxs[i];
              for (int j = 0; j < segment_size; j++) {
                  int idx = segment_idx + j;
                  buf[m++] = x[idx][0] + pbc_flags[0] * domain->prd[0];
                  buf[m++] = x[idx][1] + pbc_flags[1] * domain->prd[1];
                  buf[m++] = x[idx][2] + pbc_flags[2] * domain->prd[2];
              }
          }
      }

      return m;
  }
}

void AtomVec::unpack_data_stencil_md(int num_recv_force, int num_recv_pos,
                                                 int* recv_force_list, int* recv_pos_list,
                                                 int num_recv_ghost, int* recv_ghost_idx_list, int* recv_ghost_size_list,
                                                 double* buf) {
  if (DEBUG_SEND_RECV_DATA) {
      int m = 0;
      for (int i = 0; i < num_recv_force; i++) {
          double d = buf[m];
          tagint target_tag = (tagint) ubuf(buf[m++]).i;
          double f_x = buf[m++];
          double f_y = buf[m++];
          double f_z = buf[m++];

          int idx = recv_force_list[i];
          if (target_tag != tag[idx]) {
              std::cout << RED << " recv local force. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx << " i: " << i << " received double: " << d << RESET_COLOR << std::endl;
          }
          assert(tag[idx] == target_tag);

          f[idx][0] += f_x;
          f[idx][1] += f_y;
          f[idx][2] += f_z;
      }

      /*
      for (int i = 0; i < num_recv_pos; i++) {
          tagint target_tag = (tagint) ubuf(buf[m++]).i;
          double f_x = buf[m++];
          double f_y = buf[m++];
          double f_z = buf[m++];

          int idx = recv_pos_list[i];
          if (target_tag != tag[idx]) {
              std::cout << RED << " recv local pos. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx << RESET_COLOR << std::endl;
          }

          if (target_tag == 31165) {
              std::cout << YELLOW << "GOT FORCE FOR TAG IN POS: " << f_x << " " << f_y << " " << f_z << std::endl;
          }

          assert(tag[idx] == target_tag);

          f[idx][0] += f_x;
          f[idx][1] += f_y;
          f[idx][2] += f_z;
      }
      */

      for (int i = 0; i < num_recv_pos; i++) {
          tagint target_tag = (tagint) ubuf(buf[m++]).i;
          double x_x = buf[m++];
          double x_y = buf[m++];
          double x_z = buf[m++];

          int idx = recv_pos_list[i];
          if (target_tag != tag[idx]) {
              std::cout << RED << " recv local pos. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx << RESET_COLOR << std::endl;
          }
          assert(tag[idx] == target_tag);

          x[idx][0] = x_x;
          x[idx][1] = x_y;
          x[idx][2] = x_z;
      }

      for (int i = 0; i < num_recv_pos; i++) {
          tagint target_tag = (tagint) ubuf(buf[m++]).i;
          double v_x = buf[m++];
          double v_y = buf[m++];
          double v_z = buf[m++];

          int idx = recv_pos_list[i];
          if (target_tag != tag[idx]) {
              std::cout << RED << " recv local vel. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx << RESET_COLOR << std::endl;
          }
          assert(tag[idx] == target_tag);

          v[idx][0] = v_x;
          v[idx][1] = v_y;
          v[idx][2] = v_z;
      }

      for (int i = 0; i < num_recv_ghost; i++) {
          int ghost_idx = recv_ghost_idx_list[i];
          int ghost_size = recv_ghost_size_list[i];
          for (int j = 0; j < ghost_size; j++) {
              int idx = ghost_idx + j;
              int buf_idx = m;
              tagint target_tag = (tagint) ubuf(buf[m++]).i;
              double x_x = buf[m++];
              double x_y = buf[m++];
              double x_z = buf[m++];

              if (target_tag != tag[idx]) {
                  int total = 0;
                  for (int k = 0; k < num_recv_ghost; k++) {
                      total += recv_ghost_size_list[k];
                  }
                  std::cout << RED << " receive ghost pos. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx << " buf idx: " << buf_idx
                    << " total ghost: " << total << " segment num: " << i << " within segment idx: " << j << " pos: " << x_x << " " << x_y << " " << x_z << RESET_COLOR << std::endl;
              }
              assert(tag[idx] == target_tag);

              x[idx][0] = x_x;
              x[idx][1] = x_y;
              x[idx][2] = x_z;
          }
      }

      /*
      for (int i = 0; i < num_recv_ghost; i++) {
          int ghost_idx = recv_ghost_idx_list[i];
          int ghost_size = recv_ghost_size_list[i];
          for (int j = 0; j < ghost_size; j++) {
              int idx = ghost_idx + j;
              tagint target_tag = (tagint) ubuf(buf[m++]).i;
              double v_x = buf[m++];
              double v_y = buf[m++];
              double v_z = buf[m++];

              if (target_tag != tag[idx]) {
                  std::cout << RED << " receive ghost vel. Received tag: " << target_tag << " but I want tag: " << tag[idx] << " at idx: " << idx << RESET_COLOR << std::endl;
              }
              assert(tag[idx] == target_tag);

              v[idx][0] = v_x;
              v[idx][1] = v_y;
              v[idx][2] = v_z;
          }
      }
      */

      /*
      atomKK_->modified_stencil_md(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK, atom_);

      if (atom->nextra_border) {
          assert(false);
          for (int iextra = 0; iextra < atom->nextra_border; iextra++)
              m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
      }

      return;
      */
  } else {
      int m = 0;
      for (int i = 0; i < num_recv_force; i++) {
          double f_x = buf[m++];
          double f_y = buf[m++];
          double f_z = buf[m++];

          int idx = recv_force_list[i];

          f[idx][0] += f_x;
          f[idx][1] += f_y;
          f[idx][2] += f_z;
      }

      for (int i = 0; i < num_recv_pos; i++) {
          double x_x = buf[m++];
          double x_y = buf[m++];
          double x_z = buf[m++];

          int idx = recv_pos_list[i];

          x[idx][0] = x_x;
          x[idx][1] = x_y;
          x[idx][2] = x_z;
      }

      for (int i = 0; i < num_recv_pos; i++) {
          double v_x = buf[m++];
          double v_y = buf[m++];
          double v_z = buf[m++];

          int idx = recv_pos_list[i];

          v[idx][0] = v_x;
          v[idx][1] = v_y;
          v[idx][2] = v_z;
      }

      for (int i = 0; i < num_recv_ghost; i++) {
          int ghost_idx = recv_ghost_idx_list[i];
          int ghost_size = recv_ghost_size_list[i];
          for (int j = 0; j < ghost_size; j++) {
              int idx = ghost_idx + j;
              int buf_idx = m;
              double x_x = buf[m++];
              double x_y = buf[m++];
              double x_z = buf[m++];

              x[idx][0] = x_x;
              x[idx][1] = x_y;
              x[idx][2] = x_z;
          }
      }
  }
}

int AtomVec::pack_border_stencil_md(int i, double* buf) {
    // std::cout << "pack exchange: " << &mexchange << std::endl;
    int mm, nn, datatype, cols, collength, ncols;
    void *pdata, *plength;

    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];

    buf[m++] = ubuf(tag[i]).d;
    buf[m++] = ubuf(type[i]).d;
    buf[m++] = ubuf(mask[i]).d;
    buf[m++] = ubuf(image[i]).d;

    if (nborder) {
        for (nn = 0; nn < nborder; nn++) {
            pdata = mborder.pdata[nn];
            datatype = mborder.datatype[nn];
            cols = mborder.cols[nn];
            if (datatype == Atom::DOUBLE) {
                if (cols == 0) {
                    double *vec = *((double **) pdata);
                    buf[m++] = vec[i];
                } else {
                    double **array = *((double ***) pdata);
                    for (mm = 0; mm < cols; mm++) buf[m++] = array[i][mm];
                }
            } else if (datatype == Atom::INT) {
                if (cols == 0) {
                    int *vec = *((int **) pdata);
                    buf[m++] = ubuf(vec[i]).d;
                } else {
                    int **array = *((int ***) pdata);
                    for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
                }
            } else if (datatype == Atom::BIGINT) {
                if (cols == 0) {
                    bigint *vec = *((bigint **) pdata);
                    buf[m++] = ubuf(vec[i]).d;
                } else {
                    bigint **array = *((bigint ***) pdata);
                    for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
                }
            }
        }
    }

    if (bonus_flag) {
        assert(false);
    }

    if (atom->nextra_border) {
        assert(false);
    }

    buf[0] = m;
    return m;
}

void AtomVec::add_local_atom_stencil_md(Atom* atom_, Domain* domain_, double* coord, double* vel, tagint tag_, int type_, int mask_, imageint image_) {
    assert(false);
    int idx;
    int nlocal = atom_->nlocal;
    if (nlocal == nmax) {
        grow_stencil_md(0, atom_);
    }

    x[nlocal][0] = coord[0];
    x[nlocal][1] = coord[1];
    x[nlocal][2] = coord[2];
    v[nlocal][0] = vel[0];
    v[nlocal][1] = vel[1];
    v[nlocal][2] = vel[2];
    tag[nlocal] = (tagint) tag_;
    type[nlocal] = type_;
    mask[nlocal] = mask_;
    image[nlocal] = (imageint) image_;

    domain_->remap(x[nlocal], image[nlocal]);
    atom_->nlocal++;
}

/* START DOUBLE BUFFERING */
void AtomVec::unpack_data_from_process_stencil_md_double_buffering(std::vector<int>& tags,
                                                                   int nrecv_force, int nrecv_pos,
                                                                   int force_offset_buf,
                                                                   int vel_offset_buf,
                                                                   std::vector<dbl3_t_stencil_md>& forces,
                                                                   std::vector<int>& recv_force_idxs,
                                                                   std::vector<dbl3_t_stencil_md>& pos,
                                                                   std::vector<int>& recv_pos_local_idxs,
                                                                   std::vector<int>& recv_pos_ghost_idxs,
                                                                   std::vector<dbl3_t_stencil_md>& vel,
                                                                   std::vector<int>& recv_vel_idxs,
                                                                   int num_pos_segments_buf, bool* segment_types_buf,
                                                                   int* segment_idxs_buf, int* segment_sizes_buf,
                                                                   int* pbc_flags_,
                                                                   double* buf) {
    if (DEBUG_SEND_RECV_DATA) {
        // 0 is the starting idx of the buffeer
        int m = 0 + force_offset_buf * (3 + 1);

        for (int i = 0; i < recv_force_idxs.size(); i++) {
            tagint target_tag = (tagint) ubuf(buf[m++]).i;
            double f_x = buf[m++];
            double f_y = buf[m++];
            double f_z = buf[m++];

            int idx = recv_force_idxs[i];

            if (target_tag != tags[idx]) {
                std::cout << RED << "process: " << comm->me << " error recv local force. Received tag: " << target_tag << " but I want tag: " << tags[idx]
                    << " at idx: " << idx << " i: " << i << " out of: " << recv_force_idxs.size()
                    << " force offset in buf: " << force_offset_buf << RESET_COLOR << std::endl;
            }
            assert(tags[idx] == target_tag);

            forces[idx].x += f_x;
            forces[idx].y += f_y;
            forces[idx].z += f_z;
        }

        int pos_start_idx = nrecv_force * (3 + 1);

        int local_list_idx = 0;
        int ghost_list_idx = 0;

        for (int i = 0; i < num_pos_segments_buf; i++) {
            bool segment_type = segment_types_buf[i];
            int segment_size = segment_sizes_buf[i];
            int segment_idx = segment_idxs_buf[i];
            int counter = segment_idx * (3 + 1) + pos_start_idx;

            if (segment_type == RECV_DATA_PROCESS_LOCAL) {
                for (int j = 0; j < segment_size; j++) {
                    tagint target_tag = (tagint) ubuf(buf[counter++]).i;
                    double x_x = buf[counter++];
                    double x_y = buf[counter++];
                    double x_z = buf[counter++];

                    int idx = recv_pos_local_idxs[local_list_idx++];
                    if (tags[idx] != target_tag) {
                        std::cout << BOLDRED << "me: " << comm->me << " RECV DATA PROCESS LOCAL ERROR. Got tag: " << target_tag << " wanted tag: " << tags[idx]
                        << " nrecv force: " << nrecv_force << " i: " << i
                        << " segment idx: " << segment_idx
                        << " segment size: " << segment_size
                        << " local list idx: " << local_list_idx
                        << " counter: " << counter
                        << RESET_COLOR << std::endl;
                        assert(false);
                    }

                    assert(tags[idx] == target_tag);
                    pos[idx].x = x_x + domain->prd[0] * pbc_flags_[0];
                    pos[idx].y = x_y + domain->prd[1] * pbc_flags_[1];
                    pos[idx].z = x_z + domain->prd[2] * pbc_flags_[2];
                }
            } else {
                assert(segment_type == RECV_DATA_PROCESS_GHOST);
                for (int j = 0; j < segment_size; j++) {
                    tagint target_tag = (tagint) ubuf(buf[counter++]).i;
                    double x_x = buf[counter++];
                    double x_y = buf[counter++];
                    double x_z = buf[counter++];

                    int idx = recv_pos_ghost_idxs[ghost_list_idx++];
                    if (tags[idx] != target_tag) {
                        std::cout << BOLDRED << "me: " << comm->me << " RECV DATA PROCESS GHOST ERROR. Got tag: " << target_tag << " wanted tag: " << tags[idx]
                                  << " nrecv force: " << nrecv_force << " i: " << i
                                  << " segment idx: " << segment_idx
                                  << " segment size: " << segment_size
                                  << " local list idx: " << local_list_idx
                                  << " counter: " << counter
                                  << " ghost list idx: " << ghost_list_idx - 1
                                  << RESET_COLOR << std::endl;
                        assert(false);
                    }
                    assert(target_tag == tags[idx]);

                    pos[idx].x = x_x + domain->prd[0] * pbc_flags_[0];
                    pos[idx].y = x_y + domain->prd[1] * pbc_flags_[1];
                    pos[idx].z = x_z + domain->prd[2] * pbc_flags_[2];
                }
            }
        }

        int vel_start_idx = nrecv_force * (3 + 1) + nrecv_pos * (3 + 1) + vel_offset_buf * (3 + 1);
        m = vel_start_idx;

        for (int i = 0; i < recv_vel_idxs.size(); i++) {
            tagint target_tag = (tagint) ubuf(buf[m++]).i;
            double v_x = buf[m++];
            double v_y = buf[m++];
            double v_z = buf[m++];

            int idx = recv_vel_idxs[i];
            assert(tags[idx] == target_tag);

            vel[idx].x = v_x;
            vel[idx].y = v_y;
            vel[idx].z = v_z;
        }
    } else {
        // 0 is the starting idx of the buffeer
        int m = 0 + force_offset_buf * (3);

        #pragma cilk grainsize 4096
        cilk_for (int i = 0; i < recv_force_idxs.size(); i++) {
            // double f_x = buf[m++];
            // double f_y = buf[m++];
            // double f_z = buf[m++];
            double f_x = buf[m + i * 3];
            double f_y = buf[m + i * 3 + 1];
            double f_z = buf[m + i * 3 + 2];

            int idx = recv_force_idxs[i];

            forces[idx].x += f_x;
            forces[idx].y += f_y;
            forces[idx].z += f_z;
        }

        int pos_start_idx = nrecv_force * (3);

        int local_list_idx = 0;
        int ghost_list_idx = 0;

        for (int i = 0; i < num_pos_segments_buf; i++) {
            bool segment_type = segment_types_buf[i];
            int segment_size = segment_sizes_buf[i];
            int segment_idx = segment_idxs_buf[i];
            int counter = segment_idx * (3) + pos_start_idx;

            if (segment_type == RECV_DATA_PROCESS_LOCAL) {
                for (int j = 0; j < segment_size; j++) {
                    double x_x = buf[counter++];
                    double x_y = buf[counter++];
                    double x_z = buf[counter++];

                    int idx = recv_pos_local_idxs[local_list_idx++];
                    pos[idx].x = x_x + domain->prd[0] * pbc_flags_[0];
                    pos[idx].y = x_y + domain->prd[1] * pbc_flags_[1];
                    pos[idx].z = x_z + domain->prd[2] * pbc_flags_[2];
                }
            } else {
                assert(segment_type == RECV_DATA_PROCESS_GHOST);
                for (int j = 0; j < segment_size; j++) {
                    double x_x = buf[counter++];
                    double x_y = buf[counter++];
                    double x_z = buf[counter++];

                    int idx = recv_pos_ghost_idxs[ghost_list_idx++];
                    pos[idx].x = x_x + domain->prd[0] * pbc_flags_[0];
                    pos[idx].y = x_y + domain->prd[1] * pbc_flags_[1];
                    pos[idx].z = x_z + domain->prd[2] * pbc_flags_[2];
                }
            }
        }

        int vel_start_idx = nrecv_force * (3) + nrecv_pos * (3) + vel_offset_buf * (3);
        m = vel_start_idx;

        #pragma cilk grainsize 4096
        cilk_for (int i = 0; i < recv_vel_idxs.size(); i++) {
            // double v_x = buf[m++];
            // double v_y = buf[m++];
            // double v_z = buf[m++];
            double v_x = buf[m + i * 3];
            double v_y = buf[m + i * 3 + 1];
            double v_z = buf[m + i * 3 + 2];

            int idx = recv_vel_idxs[i];
            vel[idx].x = v_x;
            vel[idx].y = v_y;
            vel[idx].z = v_z;
        }
    }
}

/* END DOUBLE BUFFERING */

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVec::size_restart()
{
  int i, nn, cols, collength, ncols;
  void *plength;

  // NOTE: need to worry about overflow of returned int N

  int nlocal = atom->nlocal;

  // 11 = length storage + id,type,mask,image,x,v

  int n = 11 * nlocal;

  if (nrestart) {
    for (nn = 0; nn < nrestart; nn++) {
      cols = mrestart.cols[nn];
      if (cols == 0)
        n += nlocal;
      else if (cols > 0)
        n += cols * nlocal;
      else {
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        for (i = 0; i < nlocal; i++) {
          if (collength)
            ncols = (*((int ***) plength))[i][collength - 1];
          else
            ncols = (*((int **) plength))[i];
          n += ncols;
        }
      }
    }
  }

  if (bonus_flag) n += size_restart_bonus();

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for (i = 0; i < nlocal; i++) n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
------------------------------------------------------------------------- */

int AtomVec::pack_restart(int i, double *buf)
{
  int mm, nn, datatype, cols, collength, ncols;
  void *pdata, *plength;

  // if needed, change values before packing

  pack_restart_pre(i);

  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];

  for (nn = 0; nn < nrestart; nn++) {
    pdata = mrestart.pdata[nn];
    datatype = mrestart.datatype[nn];
    cols = mrestart.cols[nn];
    if (datatype == Atom::DOUBLE) {
      if (cols == 0) {
        double *vec = *((double **) pdata);
        buf[m++] = vec[i];
      } else if (cols > 0) {
        double **array = *((double ***) pdata);
        for (mm = 0; mm < cols; mm++) buf[m++] = array[i][mm];
      } else {
        double **array = *((double ***) pdata);
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        if (collength)
          ncols = (*((int ***) plength))[i][collength - 1];
        else
          ncols = (*((int **) plength))[i];
        for (mm = 0; mm < ncols; mm++) buf[m++] = array[i][mm];
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0) {
        int *vec = *((int **) pdata);
        buf[m++] = ubuf(vec[i]).d;
      } else if (cols > 0) {
        int **array = *((int ***) pdata);
        for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
      } else {
        int **array = *((int ***) pdata);
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        if (collength)
          ncols = (*((int ***) plength))[i][collength - 1];
        else
          ncols = (*((int **) plength))[i];
        for (mm = 0; mm < ncols; mm++) buf[m++] = ubuf(array[i][mm]).d;
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0) {
        bigint *vec = *((bigint **) pdata);
        buf[m++] = ubuf(vec[i]).d;
      } else if (cols > 0) {
        bigint **array = *((bigint ***) pdata);
        for (mm = 0; mm < cols; mm++) buf[m++] = ubuf(array[i][mm]).d;
      } else {
        bigint **array = *((bigint ***) pdata);
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        if (collength)
          ncols = (*((int ***) plength))[i][collength - 1];
        else
          ncols = (*((int **) plength))[i];
        for (mm = 0; mm < ncols; mm++) buf[m++] = ubuf(array[i][mm]).d;
      }
    }
  }

  if (bonus_flag) m += pack_restart_bonus(i, &buf[m]);

  // if needed, restore values after packing

  pack_restart_post(i);

  // invoke fixes which store peratom restart info

  for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
    m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVec::unpack_restart(double *buf)
{
  int mm, nn, datatype, cols, collength, ncols;
  void *pdata, *plength;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store) memory->grow(atom->extra, nmax, atom->nextra_store, "atom:extra");
  }

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];

  for (nn = 0; nn < nrestart; nn++) {
    pdata = mrestart.pdata[nn];
    datatype = mrestart.datatype[nn];
    cols = mrestart.cols[nn];
    if (datatype == Atom::DOUBLE) {
      if (cols == 0) {
        double *vec = *((double **) pdata);
        vec[nlocal] = buf[m++];
      } else if (cols > 0) {
        double **array = *((double ***) pdata);
        for (mm = 0; mm < cols; mm++) array[nlocal][mm] = buf[m++];
      } else {
        double **array = *((double ***) pdata);
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        if (collength)
          ncols = (*((int ***) plength))[nlocal][collength - 1];
        else
          ncols = (*((int **) plength))[nlocal];
        for (mm = 0; mm < ncols; mm++) array[nlocal][mm] = buf[m++];
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0) {
        int *vec = *((int **) pdata);
        vec[nlocal] = (int) ubuf(buf[m++]).i;
      } else if (cols > 0) {
        int **array = *((int ***) pdata);
        for (mm = 0; mm < cols; mm++) array[nlocal][mm] = (int) ubuf(buf[m++]).i;
      } else {
        int **array = *((int ***) pdata);
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        if (collength)
          ncols = (*((int ***) plength))[nlocal][collength - 1];
        else
          ncols = (*((int **) plength))[nlocal];
        for (mm = 0; mm < ncols; mm++) array[nlocal][mm] = (int) ubuf(buf[m++]).i;
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0) {
        bigint *vec = *((bigint **) pdata);
        vec[nlocal] = (bigint) ubuf(buf[m++]).i;
      } else if (cols > 0) {
        bigint **array = *((bigint ***) pdata);
        for (mm = 0; mm < cols; mm++) array[nlocal][mm] = (bigint) ubuf(buf[m++]).i;
      } else {
        bigint **array = *((bigint ***) pdata);
        collength = mrestart.collength[nn];
        plength = mrestart.plength[nn];
        if (collength)
          ncols = (*((int ***) plength))[nlocal][collength - 1];
        else
          ncols = (*((int **) plength))[nlocal];
        for (mm = 0; mm < ncols; mm++) array[nlocal][mm] = (bigint) ubuf(buf[m++]).i;
      }
    }
  }

  if (bonus_flag) m += unpack_restart_bonus(nlocal, &buf[m]);

  // if needed, initialize other peratom values

  unpack_restart_init(nlocal);

  // store extra restart info which fixes can unpack when instantiated

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = static_cast<int>(buf[0]) - m;
    for (int i = 0; i < size; i++) extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
------------------------------------------------------------------------- */

void AtomVec::create_atom(int itype, double *coord)
{
  int m, n, datatype, cols;
  void *pdata;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  // initialization additional fields

  for (n = 0; n < ncreate; n++) {
    pdata = mcreate.pdata[n];
    datatype = mcreate.datatype[n];
    cols = mcreate.cols[n];
    if (datatype == Atom::DOUBLE) {
      if (cols == 0) {
        double *vec = *((double **) pdata);
        vec[nlocal] = 0.0;
      } else {
        double **array = *((double ***) pdata);
        for (m = 0; m < cols; m++) array[nlocal][m] = 0.0;
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0) {
        int *vec = *((int **) pdata);
        vec[nlocal] = 0;
      } else {
        int **array = *((int ***) pdata);
        for (m = 0; m < cols; m++) array[nlocal][m] = 0;
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0) {
        bigint *vec = *((bigint **) pdata);
        vec[nlocal] = 0;
      } else {
        bigint **array = *((bigint ***) pdata);
        for (m = 0; m < cols; m++) array[nlocal][m] = 0;
      }
    }
  }

  // if needed, initialize non-zero peratom values

  create_atom_post(nlocal);

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other peratom quantities
------------------------------------------------------------------------- */

void AtomVec::data_atom(double *coord, imageint imagetmp, const std::vector<std::string> &values)
{
  int m, n, datatype, cols;
  void *pdata;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = imagetmp;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  int ivalue = 0;
  for (n = 0; n < ndata_atom; n++) {
    pdata = mdata_atom.pdata[n];
    datatype = mdata_atom.datatype[n];
    cols = mdata_atom.cols[n];
    if (datatype == Atom::DOUBLE) {
      if (cols == 0) {
        double *vec = *((double **) pdata);
        vec[nlocal] = utils::numeric(FLERR, values[ivalue++], true, lmp);
      } else {
        double **array = *((double ***) pdata);
        if (array == atom->x) {    // x was already set by coord arg
          ivalue += cols;
          continue;
        }
        for (m = 0; m < cols; m++)
          array[nlocal][m] = utils::numeric(FLERR, values[ivalue++], true, lmp);
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0) {
        int *vec = *((int **) pdata);
        vec[nlocal] = utils::inumeric(FLERR, values[ivalue++], true, lmp);
      } else {
        int **array = *((int ***) pdata);
        for (m = 0; m < cols; m++)
          array[nlocal][m] = utils::inumeric(FLERR, values[ivalue++], true, lmp);
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0) {
        bigint *vec = *((bigint **) pdata);
        vec[nlocal] = utils::bnumeric(FLERR, values[ivalue++], true, lmp);
      } else {
        bigint **array = *((bigint ***) pdata);
        for (m = 0; m < cols; m++)
          array[nlocal][m] = utils::bnumeric(FLERR, values[ivalue++], true, lmp);
      }
    }
  }

  // error checks applicable to all styles

  if (tag[nlocal] <= 0) error->one(FLERR, "Invalid atom ID in Atoms section of data file");
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR, "Invalid atom type in Atoms section of data file");

  // if needed, modify unpacked values or initialize other peratom values

  data_atom_post(nlocal);

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVec::pack_data(double **buf)
{
  int i, j, m, n, datatype, cols;
  void *pdata;

  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {

    // if needed, change values before packing

    pack_data_pre(i);

    j = 0;
    for (n = 0; n < ndata_atom; n++) {
      pdata = mdata_atom.pdata[n];
      datatype = mdata_atom.datatype[n];
      cols = mdata_atom.cols[n];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          buf[i][j++] = vec[i];
        } else {
          double **array = *((double ***) pdata);
          for (m = 0; m < cols; m++) buf[i][j++] = array[i][m];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          buf[i][j++] = ubuf(vec[i]).d;
        } else {
          int **array = *((int ***) pdata);
          for (m = 0; m < cols; m++) buf[i][j++] = ubuf(array[i][m]).d;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          buf[i][j++] = ubuf(vec[i]).d;
        } else {
          bigint **array = *((bigint ***) pdata);
          for (m = 0; m < cols; m++) buf[i][j++] = ubuf(array[i][m]).d;
        }
      }
    }

    buf[i][j++] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][j++] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][j++] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;

    // if needed, restore values after packing

    pack_data_post(i);
  }
}

/* ----------------------------------------------------------------------
   write atom info to data file
   id is first field, 3 image flags are final fields
------------------------------------------------------------------------- */

void AtomVec::write_data(FILE *fp, int n, double **buf)
{
  int i, j, m, nn, datatype, cols;

  for (i = 0; i < n; i++) {
    fmt::print(fp, "{}", ubuf(buf[i][0]).i);

    j = 1;
    for (nn = 1; nn < ndata_atom; nn++) {
      datatype = mdata_atom.datatype[nn];
      cols = mdata_atom.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          fmt::print(fp, " {:.16}", buf[i][j++]);
        } else {
          for (m = 0; m < cols; m++) fmt::print(fp, " {}", buf[i][j++]);
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        } else {
          for (m = 0; m < cols; m++) fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        } else {
          for (m = 0; m < cols; m++) fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        }
      }
    }

    fmt::print(fp, " {} {} {}\n", ubuf(buf[i][j]).i, ubuf(buf[i][j + 1]).i, ubuf(buf[i][j + 2]).i);
  }
}

/* ----------------------------------------------------------------------
   unpack one line from Velocities section of data file
------------------------------------------------------------------------- */

void AtomVec::data_vel(int ilocal, const std::vector<std::string> &values)
{
  int m, n, datatype, cols;
  void *pdata;

  double **v = atom->v;
  int ivalue = 1;
  v[ilocal][0] = utils::numeric(FLERR, values[ivalue++], true, lmp);
  v[ilocal][1] = utils::numeric(FLERR, values[ivalue++], true, lmp);
  v[ilocal][2] = utils::numeric(FLERR, values[ivalue++], true, lmp);

  if (ndata_vel > 2) {
    for (n = 2; n < ndata_vel; n++) {
      pdata = mdata_vel.pdata[n];
      datatype = mdata_vel.datatype[n];
      cols = mdata_vel.cols[n];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          vec[ilocal] = utils::numeric(FLERR, values[ivalue++], true, lmp);
        } else {
          double **array = *((double ***) pdata);
          for (m = 0; m < cols; m++)
            array[ilocal][m] = utils::numeric(FLERR, values[ivalue++], true, lmp);
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          vec[ilocal] = utils::inumeric(FLERR, values[ivalue++], true, lmp);
        } else {
          int **array = *((int ***) pdata);
          for (m = 0; m < cols; m++)
            array[ilocal][m] = utils::inumeric(FLERR, values[ivalue++], true, lmp);
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          vec[ilocal] = utils::bnumeric(FLERR, values[ivalue++], true, lmp);
        } else {
          bigint **array = *((bigint ***) pdata);
          for (m = 0; m < cols; m++)
            array[ilocal][m] = utils::bnumeric(FLERR, values[ivalue++], true, lmp);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pack velocity info for data file
------------------------------------------------------------------------- */

void AtomVec::pack_vel(double **buf)
{
  int i, j, m, n, datatype, cols;
  void *pdata;

  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    j = 0;
    for (n = 0; n < ndata_vel; n++) {
      pdata = mdata_vel.pdata[n];
      datatype = mdata_vel.datatype[n];
      cols = mdata_vel.cols[n];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          double *vec = *((double **) pdata);
          buf[i][j++] = vec[i];
        } else {
          double **array = *((double ***) pdata);
          for (m = 0; m < cols; m++) buf[i][j++] = array[i][m];
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          int *vec = *((int **) pdata);
          buf[i][j++] = ubuf(vec[i]).d;
        } else {
          int **array = *((int ***) pdata);
          for (m = 0; m < cols; m++) buf[i][j++] = ubuf(array[i][m]).d;
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          bigint *vec = *((bigint **) pdata);
          buf[i][j++] = ubuf(vec[i]).d;
        } else {
          bigint **array = *((bigint ***) pdata);
          for (m = 0; m < cols; m++) buf[i][j++] = ubuf(array[i][m]).d;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   write velocity info to data file
   id and velocity vector are first 4 fields
------------------------------------------------------------------------- */

void AtomVec::write_vel(FILE *fp, int n, double **buf)
{
  int i, j, m, nn, datatype, cols;

  for (i = 0; i < n; i++) {
    fmt::print(fp, "{}", ubuf(buf[i][0]).i);

    j = 1;
    for (nn = 1; nn < ndata_vel; nn++) {
      datatype = mdata_vel.datatype[nn];
      cols = mdata_vel.cols[nn];
      if (datatype == Atom::DOUBLE) {
        if (cols == 0) {
          fmt::print(fp, " {}", buf[i][j++]);
        } else {
          for (m = 0; m < cols; m++) fmt::print(fp, " {}", buf[i][j++]);
        }
      } else if (datatype == Atom::INT) {
        if (cols == 0) {
          fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        } else {
          for (m = 0; m < cols; m++) fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        }
      } else if (datatype == Atom::BIGINT) {
        if (cols == 0) {
          fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        } else {
          for (m = 0; m < cols; m++) fmt::print(fp, " {}", ubuf(buf[i][j++]).i);
        }
      }
    }
    fputs("\n", fp);
  }
}

/* ----------------------------------------------------------------------
   pack bond info for data file into buf if non-nullptr
   return count of bonds from this proc
   do not count/pack bonds with bondtype = 0
   if bondtype is negative, flip back to positive
------------------------------------------------------------------------- */

int AtomVec::pack_bond(tagint **buf)
{
  assert(false);
  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  int i, j;
  int m = 0;
  if (newton_bond) {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_bond[i]; j++) {
        if (bond_type[i][j] == 0) continue;
        if (buf) {
          buf[m][0] = MAX(bond_type[i][j], -bond_type[i][j]);
          buf[m][1] = tag[i];
          buf[m][2] = bond_atom[i][j];
        }
        m++;
      }
  } else {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_bond[i]; j++)
        if (tag[i] < bond_atom[i][j]) {
          if (bond_type[i][j] == 0) continue;
          if (buf) {
            buf[m][0] = MAX(bond_type[i][j], -bond_type[i][j]);
            buf[m][1] = tag[i];
            buf[m][2] = bond_atom[i][j];
          }
          m++;
        }
  }

  return m;
}

/* ----------------------------------------------------------------------
   write bond info to data file
------------------------------------------------------------------------- */

void AtomVec::write_bond(FILE *fp, int n, tagint **buf, int index)
{
  for (int i = 0; i < n; i++) {
    fmt::print(fp, "{} {} {} {}\n", index, buf[i][0], buf[i][1], buf[i][2]);
    index++;
  }
}

/* ----------------------------------------------------------------------
   pack angle info for data file into buf if non-nullptr
   return count of angles from this proc
   do not count/pack angles with angletype = 0
   if angletype is negative, flip back to positive
------------------------------------------------------------------------- */

int AtomVec::pack_angle(tagint **buf)
{
  tagint *tag = atom->tag;
  int *num_angle = atom->num_angle;
  int **angle_type = atom->angle_type;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  int i, j;
  int m = 0;
  if (newton_bond) {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_angle[i]; j++) {
        if (angle_type[i][j] == 0) continue;
        if (buf) {
          buf[m][0] = MAX(angle_type[i][j], -angle_type[i][j]);
          buf[m][1] = angle_atom1[i][j];
          buf[m][2] = angle_atom2[i][j];
          buf[m][3] = angle_atom3[i][j];
        }
        m++;
      }
  } else {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_angle[i]; j++)
        if (tag[i] == angle_atom2[i][j]) {
          if (angle_type[i][j] == 0) continue;
          if (buf) {
            buf[m][0] = MAX(angle_type[i][j], -angle_type[i][j]);
            buf[m][1] = angle_atom1[i][j];
            buf[m][2] = angle_atom2[i][j];
            buf[m][3] = angle_atom3[i][j];
          }
          m++;
        }
  }

  return m;
}

/* ----------------------------------------------------------------------
   write angle info to data file
------------------------------------------------------------------------- */

void AtomVec::write_angle(FILE *fp, int n, tagint **buf, int index)
{
  for (int i = 0; i < n; i++) {
    fmt::print(fp, "{} {} {} {} {}\n", index, buf[i][0], buf[i][1], buf[i][2], buf[i][3]);
    index++;
  }
}

/* ----------------------------------------------------------------------
   pack dihedral info for data file
------------------------------------------------------------------------- */

int AtomVec::pack_dihedral(tagint **buf)
{
  tagint *tag = atom->tag;
  int *num_dihedral = atom->num_dihedral;
  int **dihedral_type = atom->dihedral_type;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  int i, j;
  int m = 0;
  if (newton_bond) {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_dihedral[i]; j++) {
        if (buf) {
          buf[m][0] = MAX(dihedral_type[i][j], -dihedral_type[i][j]);
          buf[m][1] = dihedral_atom1[i][j];
          buf[m][2] = dihedral_atom2[i][j];
          buf[m][3] = dihedral_atom3[i][j];
          buf[m][4] = dihedral_atom4[i][j];
        }
        m++;
      }
  } else {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_dihedral[i]; j++)
        if (tag[i] == dihedral_atom2[i][j]) {
          if (buf) {
            buf[m][0] = MAX(dihedral_type[i][j], -dihedral_type[i][j]);
            buf[m][1] = dihedral_atom1[i][j];
            buf[m][2] = dihedral_atom2[i][j];
            buf[m][3] = dihedral_atom3[i][j];
            buf[m][4] = dihedral_atom4[i][j];
          }
          m++;
        }
  }

  return m;
}

/* ----------------------------------------------------------------------
   write dihedral info to data file
------------------------------------------------------------------------- */

void AtomVec::write_dihedral(FILE *fp, int n, tagint **buf, int index)
{
  for (int i = 0; i < n; i++) {
    fmt::print(fp, "{} {} {} {} {} {}\n", index, buf[i][0], buf[i][1], buf[i][2], buf[i][3],
               buf[i][4]);
    index++;
  }
}

/* ----------------------------------------------------------------------
   pack improper info for data file
------------------------------------------------------------------------- */

int AtomVec::pack_improper(tagint **buf)
{
  tagint *tag = atom->tag;
  int *num_improper = atom->num_improper;
  int **improper_type = atom->improper_type;
  tagint **improper_atom1 = atom->improper_atom1;
  tagint **improper_atom2 = atom->improper_atom2;
  tagint **improper_atom3 = atom->improper_atom3;
  tagint **improper_atom4 = atom->improper_atom4;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  int i, j;
  int m = 0;
  if (newton_bond) {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_improper[i]; j++) {
        if (buf) {
          buf[m][0] = MAX(improper_type[i][j], -improper_type[i][j]);
          buf[m][1] = improper_atom1[i][j];
          buf[m][2] = improper_atom2[i][j];
          buf[m][3] = improper_atom3[i][j];
          buf[m][4] = improper_atom4[i][j];
        }
        m++;
      }
  } else {
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < num_improper[i]; j++)
        if (tag[i] == improper_atom2[i][j]) {
          if (buf) {
            buf[m][0] = MAX(improper_type[i][j], -improper_type[i][j]);
            buf[m][1] = improper_atom1[i][j];
            buf[m][2] = improper_atom2[i][j];
            buf[m][3] = improper_atom3[i][j];
            buf[m][4] = improper_atom4[i][j];
          }
          m++;
        }
  }

  return m;
}

/* ----------------------------------------------------------------------
   write improper info to data file
------------------------------------------------------------------------- */

void AtomVec::write_improper(FILE *fp, int n, tagint **buf, int index)
{
  for (int i = 0; i < n; i++) {
    fmt::print(fp, "{} {} {} {} {} {}\n", index, buf[i][0], buf[i][1], buf[i][2], buf[i][3],
               buf[i][4]);
    index++;
  }
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

double AtomVec::memory_usage()
{
  int datatype, cols, maxcols;
  void *pdata;

  double bytes = 0;

  bytes += memory->usage(tag, nmax);
  bytes += memory->usage(type, nmax);
  bytes += memory->usage(mask, nmax);
  bytes += memory->usage(image, nmax);
  bytes += memory->usage(x, nmax, 3);
  bytes += memory->usage(v, nmax, 3);
  bytes += memory->usage(f, nmax * comm->nthreads, 3);

  for (int i = 0; i < ngrow; i++) {
    pdata = mgrow.pdata[i];
    datatype = mgrow.datatype[i];
    cols = mgrow.cols[i];
    const int nthreads = threads[i] ? comm->nthreads : 1;
    if (datatype == Atom::DOUBLE) {
      if (cols == 0) {
        bytes += memory->usage(*((double **) pdata), nmax * nthreads);
      } else if (cols > 0) {
        bytes += memory->usage(*((double ***) pdata), nmax * nthreads, cols);
      } else {
        maxcols = *(mgrow.maxcols[i]);
        bytes += memory->usage(*((double ***) pdata), nmax * nthreads, maxcols);
      }
    } else if (datatype == Atom::INT) {
      if (cols == 0) {
        bytes += memory->usage(*((int **) pdata), nmax * nthreads);
      } else if (cols > 0) {
        bytes += memory->usage(*((int ***) pdata), nmax * nthreads, cols);
      } else {
        maxcols = *(mgrow.maxcols[i]);
        bytes += memory->usage(*((int ***) pdata), nmax * nthreads, maxcols);
      }
    } else if (datatype == Atom::BIGINT) {
      if (cols == 0) {
        bytes += memory->usage(*((bigint **) pdata), nmax * nthreads);
      } else if (cols > 0) {
        bytes += memory->usage(*((bigint ***) pdata), nmax * nthreads, cols);
      } else {
        maxcols = *(mgrow.maxcols[i]);
        bytes += memory->usage(*((bigint ***) pdata), nmax * nthreads, maxcols);
      }
    }
  }

  if (bonus_flag) bytes += memory_usage_bonus();

  return bytes;
}

// ----------------------------------------------------------------------
// internal methods
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   process field strings to initialize data structs for all other methods
------------------------------------------------------------------------- */

void AtomVec::setup_fields()
{
  int n, cols;

  if ((fields_data_atom.size() < 1) || (fields_data_atom[0] != "id"))
    error->all(FLERR, "Atom style fields_data_atom must have 'id' as first field");
  if ((fields_data_vel.size() < 2) || (fields_data_vel[0] != "id") || (fields_data_vel[1] != "v"))
    error->all(FLERR, "Atom style fields_data_vel must have 'id' and 'v' as first two fields");

  // process field strings
  // return # of fields and matching index into atom.peratom (in Method struct)

  ngrow = process_fields(fields_grow, default_grow, &mgrow);
  ncopy = process_fields(fields_copy, default_copy, &mcopy);
  ncomm = process_fields(fields_comm, default_comm, &mcomm);
  ncomm_vel = process_fields(fields_comm_vel, default_comm_vel, &mcomm_vel);
  nreverse = process_fields(fields_reverse, default_reverse, &mreverse);
  nborder = process_fields(fields_border, default_border, &mborder);
  nborder_vel = process_fields(fields_border_vel, default_border_vel, &mborder_vel);
  nexchange = process_fields(fields_exchange, default_exchange, &mexchange);
  nrestart = process_fields(fields_restart, default_restart, &mrestart);
  ncreate = process_fields(fields_create, default_create, &mcreate);
  ndata_atom = process_fields(fields_data_atom, default_data_atom, &mdata_atom);
  ndata_vel = process_fields(fields_data_vel, default_data_vel, &mdata_vel);

  // populate field-based data struct for each method to use

  init_method(ngrow, &mgrow);
  init_method(ncopy, &mcopy);
  init_method(ncomm, &mcomm);
  init_method(ncomm_vel, &mcomm_vel);
  init_method(nreverse, &mreverse);
  init_method(nborder, &mborder);
  init_method(nborder_vel, &mborder_vel);
  init_method(nexchange, &mexchange);
  init_method(nrestart, &mrestart);
  init_method(ncreate, &mcreate);
  init_method(ndata_atom, &mdata_atom);
  init_method(ndata_vel, &mdata_vel);

  // create threads data struct for grow and memory_usage to use

  if (ngrow)
    threads = new bool[ngrow];
  else
    threads = nullptr;
  for (int i = 0; i < ngrow; i++) {
    const auto &field = atom->peratom[mgrow.index[i]];
    threads[i] = field.threadflag == 1;
  }

  // set style-specific sizes

  comm_x_only = 1;
  if (ncomm) comm_x_only = 0;
  if (bonus_flag && size_forward_bonus) comm_x_only = 0;

  if (nreverse == 0)
    comm_f_only = 1;
  else
    comm_f_only = 0;

  size_forward = 3;
  for (n = 0; n < ncomm; n++) {
    cols = mcomm.cols[n];
    if (cols == 0)
      size_forward++;
    else
      size_forward += cols;
  }
  if (bonus_flag) size_forward += size_forward_bonus;

  size_reverse = 3;
  for (n = 0; n < nreverse; n++) {
    cols = mreverse.cols[n];
    if (cols == 0)
      size_reverse++;
    else
      size_reverse += cols;
  }

  size_border = 6;
  for (n = 0; n < nborder; n++) {
    cols = mborder.cols[n];
    if (cols == 0)
      size_border++;
    else
      size_border += cols;
  }
  if (bonus_flag) size_border += size_border_bonus;

  size_velocity = 3;
  for (n = 0; n < ncomm_vel; n++) {
    cols = mcomm_vel.cols[n];
    if (cols == 0)
      size_velocity++;
    else
      size_velocity += cols;
  }

  size_data_atom = 0;
  for (n = 0; n < ndata_atom; n++) {
    cols = mdata_atom.cols[n];
    if (atom->peratom[mdata_atom.index[n]].name == "x") xcol_data = size_data_atom + 1;
    if (cols == 0)
      size_data_atom++;
    else
      size_data_atom += cols;
  }

  size_data_vel = 0;
  for (n = 0; n < ndata_vel; n++) {
    cols = mdata_vel.cols[n];
    if (cols == 0)
      size_data_vel++;
    else
      size_data_vel += cols;
  }
}

void AtomVec::setup_fields_stencil_md(Atom* atom_) {
    int n, cols;

    if ((fields_data_atom.size() < 1) || (fields_data_atom[0] != "id"))
        error->all(FLERR, "Atom style fields_data_atom must have 'id' as first field");
    if ((fields_data_vel.size() < 2) || (fields_data_vel[0] != "id") || (fields_data_vel[1] != "v"))
        error->all(FLERR, "Atom style fields_data_vel must have 'id' and 'v' as first two fields");

    // process field strings
    // return # of fields and matching index into atom.peratom (in Method struct)

    ngrow = process_fields_stencil_md(fields_grow, default_grow, &mgrow, atom_);
    ncopy = process_fields_stencil_md(fields_copy, default_copy, &mcopy, atom_);
    ncomm = process_fields_stencil_md(fields_comm, default_comm, &mcomm, atom_);
    ncomm_vel = process_fields_stencil_md(fields_comm_vel, default_comm_vel, &mcomm_vel, atom_);
    nreverse = process_fields_stencil_md(fields_reverse, default_reverse, &mreverse, atom_);
    nborder = process_fields_stencil_md(fields_border, default_border, &mborder, atom_);
    nborder_vel = process_fields_stencil_md(fields_border_vel, default_border_vel, &mborder_vel, atom_);
    nexchange = process_fields_stencil_md(fields_exchange, default_exchange, &mexchange, atom_);
    nrestart = process_fields_stencil_md(fields_restart, default_restart, &mrestart, atom_);
    ncreate = process_fields_stencil_md(fields_create, default_create, &mcreate, atom_);
    ndata_atom = process_fields_stencil_md(fields_data_atom, default_data_atom, &mdata_atom, atom_);
    ndata_vel = process_fields_stencil_md(fields_data_vel, default_data_vel, &mdata_vel, atom_);

    // populate field-based data struct for each method to use

    init_method_stencil_md(ngrow, &mgrow, atom_);
    init_method_stencil_md(ncopy, &mcopy, atom_);
    init_method_stencil_md(ncomm, &mcomm, atom_);
    init_method_stencil_md(ncomm_vel, &mcomm_vel, atom_);
    init_method_stencil_md(nreverse, &mreverse, atom_);
    init_method_stencil_md(nborder, &mborder, atom_);
    init_method_stencil_md(nborder_vel, &mborder_vel, atom_);
    init_method_stencil_md(nexchange, &mexchange, atom_);
    init_method_stencil_md(nrestart, &mrestart, atom_);
    init_method_stencil_md(ncreate, &mcreate, atom_);
    init_method_stencil_md(ndata_atom, &mdata_atom, atom_);
    init_method_stencil_md(ndata_vel, &mdata_vel, atom_);

    // create threads data struct for grow and memory_usage to use

    if (ngrow)
        threads = new bool[ngrow];
    else
        threads = nullptr;
    for (int i = 0; i < ngrow; i++) {
        const auto &field = atom_->peratom[mgrow.index[i]];
        threads[i] = field.threadflag == 1;
    }

    // set style-specific sizes

    comm_x_only = 1;
    if (ncomm) comm_x_only = 0;
    if (bonus_flag && size_forward_bonus) comm_x_only = 0;

    if (nreverse == 0)
        comm_f_only = 1;
    else
        comm_f_only = 0;

    size_forward = 3;
    for (n = 0; n < ncomm; n++) {
        cols = mcomm.cols[n];
        if (cols == 0)
            size_forward++;
        else
            size_forward += cols;
    }
    if (bonus_flag) size_forward += size_forward_bonus;

    size_reverse = 3;
    for (n = 0; n < nreverse; n++) {
        cols = mreverse.cols[n];
        if (cols == 0)
            size_reverse++;
        else
            size_reverse += cols;
    }

    size_border = 6;
    for (n = 0; n < nborder; n++) {
        cols = mborder.cols[n];
        if (cols == 0)
            size_border++;
        else
            size_border += cols;
    }
    if (bonus_flag) size_border += size_border_bonus;

    size_velocity = 3;
    for (n = 0; n < ncomm_vel; n++) {
        cols = mcomm_vel.cols[n];
        if (cols == 0)
            size_velocity++;
        else
            size_velocity += cols;
    }

    size_data_atom = 0;
    for (n = 0; n < ndata_atom; n++) {
        cols = mdata_atom.cols[n];
        if (atom_->peratom[mdata_atom.index[n]].name == "x") xcol_data = size_data_atom + 1;
        if (cols == 0)
            size_data_atom++;
        else
            size_data_atom += cols;
    }

    size_data_vel = 0;
    for (n = 0; n < ndata_vel; n++) {
        cols = mdata_vel.cols[n];
        if (cols == 0)
            size_data_vel++;
        else
            size_data_vel += cols;
    }
}

/* ----------------------------------------------------------------------
   process a single field string
------------------------------------------------------------------------- */

int AtomVec::process_fields(const std::vector<std::string> &words,
                            const std::vector<std::string> &def_words, Method *method)
{
  int nfield = words.size();
  int ndef = def_words.size();

  // process fields one by one, add to index vector

  const auto &peratom = atom->peratom;
  const int nperatom = peratom.size();

  // allocate memory in method
  method->resize(nfield);

  std::vector<int> &index = method->index;
  int match;

  for (int i = 0; i < nfield; i++) {
    const std::string &field = words[i];

    // find field in master Atom::peratom list

    for (match = 0; match < nperatom; match++)
      if (field == peratom[match].name) break;
    if (match == nperatom) error->all(FLERR, "Peratom field {} not recognized", field);
    index[i] = match;

    // error if field appears multiple times

    for (match = 0; match < i; match++)
      if (index[i] == index[match]) error->all(FLERR, "Peratom field {} is repeated", field);

    // error if field is in default str

    for (match = 0; match < ndef; match++)
      if (field == def_words[match]) error->all(FLERR, "Peratom field {} is a default", field);
  }

  return nfield;
}

int AtomVec::process_fields_stencil_md(const std::vector<std::string> &words,
                                       const std::vector<std::string> &def_words,
                                       Method *method, Atom* atom_) {
    int nfield = words.size();
    int ndef = def_words.size();

    // process fields one by one, add to index vector

    const auto &peratom = atom_->peratom;
    const int nperatom = peratom.size();

    // allocate memory in method
    method->resize(nfield);

    std::vector<int> &index = method->index;
    int match;

    for (int i = 0; i < nfield; i++) {
        const std::string &field = words[i];

        // find field in master Atom::peratom list

        for (match = 0; match < nperatom; match++)
            if (field == peratom[match].name) break;
        if (match == nperatom) error->all(FLERR, "Peratom field {} not recognized", field);
        index[i] = match;

        // error if field appears multiple times

        for (match = 0; match < i; match++)
            if (index[i] == index[match]) error->all(FLERR, "Peratom field {} is repeated", field);

        // error if field is in default str

        for (match = 0; match < ndef; match++)
            if (field == def_words[match]) error->all(FLERR, "Peratom field {} is a default", field);
    }

    return nfield;
}

/* ----------------------------------------------------------------------
   init method data structs for processing fields
------------------------------------------------------------------------- */

void AtomVec::init_method(int nfield, Method *method)
{
  for (int i = 0; i < nfield; i++) {
    const auto &field = atom->peratom[method->index[i]];
    method->pdata[i] = (void *) field.address;
    method->datatype[i] = field.datatype;
    method->cols[i] = field.cols;
    if (method->cols[i] < 0) {
      method->maxcols[i] = field.address_maxcols;
      method->collength[i] = field.collength;
      method->plength[i] = field.address_length;
    }
  }
}

void AtomVec::init_method_stencil_md(int nfield, Method *method, Atom* atom_) {
    for (int i = 0; i < nfield; i++) {
        const auto &field = atom_->peratom[method->index[i]];
        method->pdata[i] = (void *) field.address;
        method->datatype[i] = field.datatype;
        method->cols[i] = field.cols;
        if (method->cols[i] < 0) {
            method->maxcols[i] = field.address_maxcols;
            method->collength[i] = field.collength;
            method->plength[i] = field.address_length;
        }
    }
}

/* ----------------------------------------------------------------------
   Method class members
------------------------------------------------------------------------- */

void AtomVec::Method::resize(int nfield)
{
  pdata.resize(nfield);
  datatype.resize(nfield);
  cols.resize(nfield);
  maxcols.resize(nfield);
  collength.resize(nfield);
  plength.resize(nfield);
  index.resize(nfield);
}
