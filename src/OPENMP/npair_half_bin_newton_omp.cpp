// clang-format off
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

#include "omp_compat.h"
#include "npair_half_bin_newton_omp.h"
#include "npair_omp.h"
#include "neigh_list.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "domain.h"
#include "my_page.h"
#include "error.h"
#include <cilk/cilk.h>

using namespace LAMMPS_NS;

constexpr int target_tag = 21731;

/* ---------------------------------------------------------------------- */

NPairHalfBinNewtonOmp::NPairHalfBinNewtonOmp(LAMMPS *lmp) : NPair(lmp) {}

/* ----------------------------------------------------------------------
   binned neighbor list construction with full Newton's 3rd law
   each owned atom i checks its own bin and other bins in Newton stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void NPairHalfBinNewtonOmp::build(NeighList *list)
{
  std::cout << " npairhalfbinnewtonomp lammps build start" << std::endl;
  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;
  const int molecular = atom->molecular;
  const int moltemplate = (molecular == Atom::TEMPLATE) ? 1 : 0;

  if (LAMMPS_USE_CILK) {
      const int nthreads = comm->nthreads;
      const int ifix = modify->find_fix("package_omp");

      cilk_for(int tid = 0; tid < nthreads; tid++) {
          const int idelta = 1 + nlocal / nthreads;
          const int ifrom = tid * idelta;
          const int ito = ((ifrom + idelta) > nlocal) ? nlocal : (ifrom + idelta);
          FixOMP *fix = static_cast<FixOMP *>(modify->fix[ifix]);
          ThrData *thr = fix->get_thr(tid);
          thr->timer(Timer::START);

          int i,j,k,n,itype,jtype,ibin,which,imol,iatom;
          tagint tagprev;
          double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
          int *neighptr;

          // loop over each atom, storing neighbors

          double **x = atom->x;
          int *type = atom->type;
          int *mask = atom->mask;
          tagint *tag = atom->tag;
          tagint *molecule = atom->molecule;
          tagint **special = atom->special;
          int **nspecial = atom->nspecial;

          int *molindex = atom->molindex;
          int *molatom = atom->molatom;
          Molecule **onemols = atom->avec->onemols;

          int *ilist = list->ilist;
          int *numneigh = list->numneigh;
          int **firstneigh = list->firstneigh;

          // each thread has its own page allocator
          MyPage<int> &ipage = list->ipage[tid];
          ipage.reset();

          for (i = ifrom; i < ito; i++) {
              n = 0;
              neighptr = ipage.vget();

              itype = type[i];
              xtmp = x[i][0];
              ytmp = x[i][1];
              ztmp = x[i][2];
              if (moltemplate) {
                  imol = molindex[i];
                  iatom = molatom[i];
                  tagprev = tag[i] - iatom - 1;
              }

              // loop over rest of atoms in i's bin, ghosts are at end of linked list
              // if j is owned atom, store it, since j is beyond i in linked list
              // if j is ghost, only store if j coords are "above and to the right" of i

              for (j = bins[i]; j >= 0; j = bins[j]) {
                  if (j >= nlocal) {
                      if (x[j][2] < ztmp) continue;
                      if (x[j][2] == ztmp) {
                          if (x[j][1] < ytmp) continue;
                          if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
                      }
                  }

                  jtype = type[j];
                  if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

                  delx = xtmp - x[j][0];
                  dely = ytmp - x[j][1];
                  delz = ztmp - x[j][2];
                  rsq = delx*delx + dely*dely + delz*delz;

                  if (rsq <= cutneighsq[itype][jtype]) {
                      if (molecular != Atom::ATOMIC) {
                          if (!moltemplate)
                              which = find_special(special[i],nspecial[i],tag[j]);
                          else if (imol >=0)
                              which = find_special(onemols[imol]->special[iatom],
                                                   onemols[imol]->nspecial[iatom],
                                                   tag[j]-tagprev);
                          else which = 0;
                          if (which == 0) neighptr[n++] = j;
                          else if (domain->minimum_image_check(delx,dely,delz))
                              neighptr[n++] = j;
                          else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
                          // OLD: if (which >= 0) neighptr[n++] = j ^ (which << SBBITS);
                      } else neighptr[n++] = j;
                  }
              }

              // loop over all atoms in other bins in stencil, store every pair

              ibin = atom2bin[i];
              for (k = 0; k < nstencil; k++) {
                  for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
                      jtype = type[j];
                      if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

                      delx = xtmp - x[j][0];
                      dely = ytmp - x[j][1];
                      delz = ztmp - x[j][2];
                      rsq = delx*delx + dely*dely + delz*delz;

                      if (rsq <= cutneighsq[itype][jtype]) {
                          if (molecular != Atom::ATOMIC) {
                              if (!moltemplate)
                                  which = find_special(special[i],nspecial[i],tag[j]);
                              else if (imol >=0)
                                  which = find_special(onemols[imol]->special[iatom],
                                                       onemols[imol]->nspecial[iatom],
                                                       tag[j]-tagprev);
                              else which = 0;
                              if (which == 0) neighptr[n++] = j;
                              else if (domain->minimum_image_check(delx,dely,delz))
                                  neighptr[n++] = j;
                              else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
                              // OLD: if (which >= 0) neighptr[n++] = j ^ (which << SBBITS);
                          } else neighptr[n++] = j;
                      }
                  }
              }

              ilist[i] = i;
              firstneigh[i] = neighptr;
              numneigh[i] = n;
              ipage.vgot(n);
              if (ipage.status())
                  error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
          }

          thr->timer(Timer::NEIGH);
          list->inum = nlocal;
      }
      return;
  }

  NPAIR_OMP_INIT;
#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(list)
#endif
  NPAIR_OMP_SETUP(nlocal);

  int i,j,k,n,itype,jtype,ibin,which,imol,iatom;
  tagint tagprev;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  // loop over each atom, storing neighbors

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;

  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  Molecule **onemols = atom->avec->onemols;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  for (i = ifrom; i < ito; i++) {

    n = 0;
    neighptr = ipage.vget();

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    if (moltemplate) {
      imol = molindex[i];
      iatom = molatom[i];
      tagprev = tag[i] - iatom - 1;
    }

    // loop over rest of atoms in i's bin, ghosts are at end of linked list
    // if j is owned atom, store it, since j is beyond i in linked list
    // if j is ghost, only store if j coords are "above and to the right" of i

    for (j = bins[i]; j >= 0; j = bins[j]) {
      if (j >= nlocal) {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp) {
          if (x[j][1] < ytmp) continue;
          if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
        }
      }

      jtype = type[j];
      if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq <= cutneighsq[itype][jtype]) {
        if (molecular != Atom::ATOMIC) {
          if (!moltemplate)
            which = find_special(special[i],nspecial[i],tag[j]);
          else if (imol >=0)
            which = find_special(onemols[imol]->special[iatom],
                                 onemols[imol]->nspecial[iatom],
                                 tag[j]-tagprev);
          else which = 0;
          if (which == 0) neighptr[n++] = j;
          else if (domain->minimum_image_check(delx,dely,delz))
            neighptr[n++] = j;
          else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
          // OLD: if (which >= 0) neighptr[n++] = j ^ (which << SBBITS);
        } else neighptr[n++] = j;
      }
    }

    // loop over all atoms in other bins in stencil, store every pair

    ibin = atom2bin[i];
    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        jtype = type[j];
        if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq[itype][jtype]) {
          if (molecular != Atom::ATOMIC) {
            if (!moltemplate)
              which = find_special(special[i],nspecial[i],tag[j]);
            else if (imol >=0)
              which = find_special(onemols[imol]->special[iatom],
                                   onemols[imol]->nspecial[iatom],
                                   tag[j]-tagprev);
            else which = 0;
            if (which == 0) neighptr[n++] = j;
            else if (domain->minimum_image_check(delx,dely,delz))
              neighptr[n++] = j;
            else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
            // OLD: if (which >= 0) neighptr[n++] = j ^ (which << SBBITS);
          } else neighptr[n++] = j;
        }
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
  NPAIR_OMP_CLOSE;
  list->inum = nlocal;
}

void NPairHalfBinNewtonOmp::build_stencil_md(NeighList *list, Atom* atom_, Domain* domain_, queue_info& zoid) {
    if (zoid.num == 0) {
        std::cout << "zoid: " << zoid.num << " npairhalfbinnewtonomp stencilmd build start" << std::endl;
    }
    int i,j,k,n,itype,jtype,ibin;
    double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
    int *neighptr;
    int moltemplate;
    int imol, iatom;
    tagint tagprev;

    double **x = atom_->x;
    int *type = atom_->type;
    int *mask = atom_->mask;
    tagint *tag = atom_->tag;
    tagint *molecule = atom_->molecule;
    tagint **special = atom_->special;
    int **nspecial = atom_->nspecial;
    int nlocal = atom_->nlocal;
    if (includegroup) nlocal = atom_->nfirst;

    int *molindex = atom_->molindex;
    int *molatom = atom_->molatom;
    Molecule **onemols = atom_->avec->onemols;
    if (molecular == Atom::TEMPLATE) moltemplate = 1;
    else moltemplate = 0;

    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;
    MyPage<int> *ipage = list->ipage;

    int inum = 0;
    ipage->reset();

    for (i = 0; i < nlocal; i++) {
        n = 0;
        neighptr = ipage->vget();

        itype = type[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];

        if (moltemplate) {
            assert(false);
            imol = molindex[i];
            iatom = molatom[i];
            tagprev = tag[i] - iatom - 1;
        }

        std::set<int> neighbor_idxs;

        // loop over rest of atoms in i's bin, ghosts are at end of linked list
        // if j is owned atom, store it, since j is beyond i in linked list
        // if j is ghost, only store if j coords are "above and to the right" of i

        for (j = bins[i]; j >= 0; j = bins[j]) {
            if (i == j) {
                continue;
            }
            /*
            if (j >= nlocal) {
                if (x[j][2] < ztmp) continue;
                if (x[j][2] == ztmp) {
                    if (x[j][1] < ytmp) continue;
                    if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
                }
            }
            */

            if (j < nlocal) {
                if (x[j][2] < ztmp) continue;
                if (x[j][2] == ztmp) {
                    if (x[j][1] < ytmp) continue;
                    if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
                }
            }

            // add an edge if the ghost atom is ghost in a shrinking dimension
            if (j >= nlocal) {
                bool shrinking_out_of_bounds = false;
                bool expanding_out_of_bounds = false;

                bool debug = false;
                double debug_lo[3] = {0};
                double debug_hi[3] = {0};
                bool debug_ghost_edge[3] = {false, false, false};

                for (int dim = 0; dim < 3; dim++) {
                    double lo = domain_->sublo[dim];
                    double hi = domain_->subhi[dim];
                    bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                    bool out_of_bounds = (x[j][dim] < lo || x[j][dim] >= hi);
                    if (my_dim_shrinking && out_of_bounds) {
                        shrinking_out_of_bounds = true;
                    } else if (!my_dim_shrinking && out_of_bounds) {
                        expanding_out_of_bounds = true;
                    }

                    debug_lo[dim] = lo;
                    debug_hi[dim] = hi;
                }

                bool add_ghost_edge = shrinking_out_of_bounds;

                if (!add_ghost_edge) {
                    continue;
                }
            }

            jtype = type[j];
            if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;

            if (rsq <= cutneighsq[itype][jtype]) {
                int which;
                if (molecular != Atom::ATOMIC) {
                    if (!moltemplate) {
                        // RYAN: this is the path that is taken
                        which = find_special(special[i],nspecial[i],tag[j]);
                    } else if (imol >= 0) {
                        assert(false);
                        which = find_special(onemols[imol]->special[iatom],
                                             onemols[imol]->nspecial[iatom],
                                             tag[j] - tagprev);
                    } else {
                        assert(false);
                        which = 0;
                    }

                    if (which == 0) {
                        neighptr[n++] = j;
                        neighbor_idxs.insert(j);
                    } else if (domain->minimum_image_check(delx,dely,delz)) {
                        neighptr[n++] = j;
                        neighbor_idxs.insert(j);
                    } else if (which > 0) {
                        neighptr[n++] = j ^ (which << SBBITS);
                        neighbor_idxs.insert(j);
                    }
                    // OLD: if (which >= 0) neighptr[n++] = j ^ (which << SBBITS);
                } else {
                    neighptr[n++] = j;
                    neighbor_idxs.insert(j);
                }
                // assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                // neighbor_idxs.insert(j);
                // neighptr[n++] = j;
            }
        }

        // loop over all atoms in other bins in stencil, store every pair

        ibin = atom2bin[i];

        for (k = 0; k < nstencil; k++) {
            for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
                if (i == j) {
                    continue;
                }
                if (neighbor_idxs.find(j) != neighbor_idxs.end()) {
                    continue;
                }
                // add an edge if the ghost atom is ghost in a shrinking dimension
                if (j < nlocal) {
                    if (x[j][2] < ztmp) continue;
                    if (x[j][2] == ztmp) {
                        if (x[j][1] < ytmp) continue;
                        if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
                    }
                }

                if (j >= nlocal) {
                    bool shrinking_out_of_bounds = false;
                    bool expanding_out_of_bounds = false;

                    for (int dim = 0; dim < 3; dim++) {
                        double lo = domain_->sublo[dim];
                        double hi = domain_->subhi[dim];
                        bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                        bool out_of_bounds = (x[j][dim] < lo || x[j][dim] >= hi);
                        if (my_dim_shrinking && out_of_bounds) {
                            shrinking_out_of_bounds = true;
                        } else if (!my_dim_shrinking && out_of_bounds) {
                            expanding_out_of_bounds = true;
                        }
                    }

                    bool add_ghost_edge = shrinking_out_of_bounds;

                    if (!add_ghost_edge) {
                        continue;
                    }
                }

                jtype = type[j];
                if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

                delx = xtmp - x[j][0];
                dely = ytmp - x[j][1];
                delz = ztmp - x[j][2];
                rsq = delx*delx + dely*dely + delz*delz;

                if (rsq <= cutneighsq[itype][jtype]) {
                    if (molecular != Atom::ATOMIC) {
                        int which;
                        if (!moltemplate) {
                            which = find_special(special[i], nspecial[i], tag[j]);
                        } else if (imol >= 0) {
                            which = find_special(onemols[imol]->special[iatom],
                                                 onemols[imol]->nspecial[iatom],
                                                 tag[j] - tagprev);
                        } else {
                            which = 0;
                        }

                        if (which == 0) {
                            neighptr[n++] = j;
                            assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                            neighbor_idxs.insert(j);
                        } else if (domain->minimum_image_check(delx,dely,delz)) {
                            neighptr[n++] = j;
                            assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                            neighbor_idxs.insert(j);
                        } else if (which > 0) {
                            neighptr[n++] = j ^ (which << SBBITS);
                            assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                            neighbor_idxs.insert(j);
                        }
                        // OLD: if (which >= 0) neighptr[n++] = j ^ (which << SBBITS);
                    } else {
                        neighptr[n++] = j;
                        assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                        neighbor_idxs.insert(j);
                    }

                    /*
                    if (neighbor_idxs.find(j) != neighbor_idxs.end()) {
                        std::cout << RED << "zoid: " << zoid.num << " src idx: " << i << " repeated idx: " << j << RESET_COLOR << std::endl;
                    }
                    assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                    neighbor_idxs.insert(j);
                    neighptr[n++] = j;
                    */
                }
            }
        }

        ilist[inum++] = i;
        firstneigh[i] = neighptr;
        numneigh[i] = n;
        ipage->vgot(n);
        if (ipage->status())
            error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
    }

    list->inum = inum;
}
