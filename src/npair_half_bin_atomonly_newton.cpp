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

#include "npair_half_bin_atomonly_newton.h"

#include "atom.h"
#include "error.h"
#include "my_page.h"
#include "neigh_list.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NPairHalfBinAtomonlyNewton::NPairHalfBinAtomonlyNewton(LAMMPS *lmp) :
  NPair(lmp) {}

/* ----------------------------------------------------------------------
   binned neighbor list construction with full Newton's 3rd law
   each owned atom i checks its own bin and other bins in Newton stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void NPairHalfBinAtomonlyNewton::build(NeighList *list)
{
  int i,j,k,n,itype,jtype,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  if (includegroup) nlocal = atom->nfirst;

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

    std::set<int> neighbor_idxs;

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
          assert(neighbor_idxs.find(j) == neighbor_idxs.end());
          neighbor_idxs.insert(j);
          neighptr[n++] = j;
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
            assert(neighbor_idxs.find(j) == neighbor_idxs.end());
            neighbor_idxs.insert(j);
            neighptr[n++] = j;
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

// default lj/cut calls this npair
void NPairHalfBinAtomonlyNewton::build_stencil_md(NeighList *list, Atom* atom_, Domain* domain_, queue_info& zoid) {
    int i,j,k,n,itype,jtype,ibin;
    double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
    int *neighptr;

    double **x = atom_->x;
    int *type = atom_->type;
    int *mask = atom_->mask;
    tagint *molecule = atom_->molecule;
    int nlocal = atom_->nlocal;
    if (includegroup) nlocal = atom_->nfirst;

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
                    bool out_of_bounds = (x[j][dim] < lo || x[j][dim] > hi);
                    if (my_dim_shrinking && out_of_bounds) {
                        shrinking_out_of_bounds = true;
                    } else if (!my_dim_shrinking && out_of_bounds) {
                        expanding_out_of_bounds = true;
                    }

                    debug_lo[dim] = lo;
                    debug_hi[dim] = hi;
                }

                bool add_ghost_edge = shrinking_out_of_bounds;

                if (debug) {
                    std::cout << "zoid: " << zoid.num << " my bin my pos: " << x[i][0] << " " << x[i][1] << " " << x[i][2] << " other pos: " << x[j][0] << " " << x[j][1] << " " << x[j][2]
                              << " lo: " << debug_lo[0] << " " << debug_lo[1] << " " << debug_lo[2]
                              << " hi: " << debug_hi[0] << " " << debug_hi[1] << " " << debug_hi[2]
                              << " add ghost edge? " << debug_ghost_edge[0] << " " << debug_ghost_edge[1] << " " << debug_ghost_edge[2]
                              << std::endl;
                }

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
                assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                neighbor_idxs.insert(j);
                neighptr[n++] = j;
            }
        }

        // loop over all atoms in other bins in stencil, store every pair

        ibin = atom2bin[i];

        for (k = 0; k < nstencil; k++) {
            for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
                bool debug2 = false;
                if (debug2) {
                    std::cout << "FOUND TARGET ATOM in stencil? idx i: "
                        << i << " tag i: " << atom_->tag[i] << " idx j: " << j << " tag j: " << atom_->tag[j] << " nlocal? " << nlocal
                        << " cut neigh sq? " << cutneighsq[itype][jtype]
                        << std::endl;
                }
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
                if (debug2) {
                    std::cout << "PASSED THE GAUNTLET FOUND TARGET ATOM in stencil? idx i: "
                              << i << " tag i: " << atom_->tag[i] << " idx j: " << j << " tag j: " << atom_->tag[j] << " nlocal? " << nlocal
                              << " rsq? " << rsq << " cut neigh sq? " << cutneighsq[itype][jtype]
                              << std::endl;
                }

                if (j >= nlocal) {
                    bool debug = false;
                    double debug_lo[3] = {0};
                    double debug_hi[3] = {0};
                    bool debug_ghost_edge[3] = {false, false, false};
                    bool shrinking_out_of_bounds = false;
                    bool expanding_out_of_bounds = false;

                    for (int dim = 0; dim < 3; dim++) {
                        double lo = domain_->sublo[dim];
                        double hi = domain_->subhi[dim];
                        bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                        bool out_of_bounds = (x[j][dim] < lo || x[j][dim] > hi);
                        if (my_dim_shrinking && out_of_bounds) {
                            shrinking_out_of_bounds = true;
                        } else if (!my_dim_shrinking && out_of_bounds) {
                            expanding_out_of_bounds = true;
                        }
                        debug_lo[dim] = lo;
                        debug_hi[dim] = hi;
                    }

                    bool add_ghost_edge = shrinking_out_of_bounds;

                    if (debug) {
                        std::cout << GREEN << "zoid: " << zoid.num << " other bin my tag: " << atom_->tag[i] << " other tag: " << atom_->tag[j] << " my pos: " << x[i][0] << " " << x[i][1] << " " << x[i][2] << " other pos: " << x[j][0] << " " << x[j][1] << " " << x[j][2]
                                  << " lo: " << debug_lo[0] << " " << debug_lo[1] << " " << debug_lo[2]
                                  << " hi: " << debug_hi[0] << " " << debug_hi[1] << " " << debug_hi[2]
                                  << " add ghost edge? " << debug_ghost_edge[0] << " " << debug_ghost_edge[1] << " " << debug_ghost_edge[2]
                                  << " final res? " << add_ghost_edge << RESET_COLOR << std::endl;
                    }

                    if (!add_ghost_edge) {
                        continue;
                    }
                }

                jtype = type[j];
                if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

                if (debug2) {
                    std::cout << "PASSED YET ANOTHER TEST FOUND TARGET ATOM in stencil? idx i: "
                              << i << " tag i: " << atom_->tag[i] << " idx j: " << j << " tag j: " << atom_->tag[j] << " nlocal? " << nlocal
                              << " rsq? " << rsq << " cut neigh sq? " << cutneighsq[itype][jtype]
                              << std::endl;
                }

                delx = xtmp - x[j][0];
                dely = ytmp - x[j][1];
                delz = ztmp - x[j][2];
                rsq = delx*delx + dely*dely + delz*delz;

                if (rsq <= cutneighsq[itype][jtype]) {
                    if (neighbor_idxs.find(j) != neighbor_idxs.end()) {
                        std::cout << RED << "zoid: " << zoid.num << " src idx: " << i << " repeated idx: " << j << RESET_COLOR << std::endl;
                    }
                    assert(neighbor_idxs.find(j) == neighbor_idxs.end());
                    neighbor_idxs.insert(j);
                    neighptr[n++] = j;
                    if (debug2) {
                        std::cout << "GOTTEM TARGET ATOM in stencil? idx i: "
                                  << i << " tag i: " << atom_->tag[i] << " idx j: " << j << " tag j: " << atom_->tag[j] << " nlocal? " << nlocal
                                  << " rsq? " << rsq << " cut neigh sq? " << cutneighsq[itype][jtype]
                                  << " list? " << list << std::endl;
                    }
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

    if (zoid.num == 24) {
        // assert(false);
    }

    list->inum = inum;
}
