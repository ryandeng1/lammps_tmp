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

#include "ntopo_bond_all.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "output.h"
#include "thermo.h"
#include "update.h"

using namespace LAMMPS_NS;

#define DELTA 10000

constexpr int target_tag = 14968;

/* ---------------------------------------------------------------------- */

NTopoBondAll::NTopoBondAll(LAMMPS *lmp) : NTopo(lmp)
{
  allocate_bond();
}

/* ---------------------------------------------------------------------- */

void NTopoBondAll::build()
{
  int i, m, atom1;

  int nlocal = atom->nlocal;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  tagint *tag = atom->tag;
  int newton_bond = force->newton_bond;

  int lostbond = output->thermo->lostbond;
  int nmissing = 0;
  nbondlist = 0;

  for (i = 0; i < nlocal; i++)
    for (m = 0; m < num_bond[i]; m++) {
      atom1 = atom->map(bond_atom[i][m]);
      if (atom1 == -1) {
        nmissing++;
        if (lostbond == Thermo::ERROR)
          error->one(FLERR,
                     "Bond atoms {} {} missing on "
                     "proc {} at step {}",
                     tag[i], bond_atom[i][m], me, update->ntimestep);
        continue;
      }
      atom1 = domain->closest_image(i, atom1);
      if (newton_bond || i < atom1) {
        if (nbondlist == maxbond) {
          maxbond += DELTA;
          memory->grow(bondlist, maxbond, 3, "neigh_topo:bondlist");
        }
        bondlist[nbondlist][0] = i;
        bondlist[nbondlist][1] = atom1;
        bondlist[nbondlist][2] = bond_type[i][m];
        nbondlist++;
      }
    }

  if (cluster_check) bond_check();
  if (lostbond == Thermo::IGNORE) return;

  int all;
  MPI_Allreduce(&nmissing, &all, 1, MPI_INT, MPI_SUM, world);
  if (all && (me == 0)) error->warning(FLERR, "Bond atoms missing at step {}", update->ntimestep);
}

int closest_image(Atom* atom_, int i, int j) {
    if (j < 0) return j;

    int *sametag = atom_->sametag;
    double **x = atom_->x;
    double *xi = x[i];

    int closest = j;
    double delx = xi[0] - x[j][0];
    double dely = xi[1] - x[j][1];
    double delz = xi[2] - x[j][2];
    double rsqmin = delx*delx + dely*dely + delz*delz;
    double rsq;

    while (sametag[j] >= 0) {
        j = sametag[j];
        delx = xi[0] - x[j][0];
        dely = xi[1] - x[j][1];
        delz = xi[2] - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < rsqmin) {
            rsqmin = rsq;
            closest = j;
        }
    }

    return closest;
}

// Need to do similar shenanigans to ensure bond list respects dependency levels
void NTopoBondAll::build_stencil_md(Atom* atom_, Domain* domain_, queue_info& zoid) {
    if (zoid.num == 0) {
        std::cout << "zoid: " << zoid.num << " ntopobondall stencilmd build start" << std::endl;
    }
    int i, m, atom1;

    int nlocal = atom_->nlocal;
    int *num_bond = atom_->num_bond;
    tagint **bond_atom = atom_->bond_atom;
    int **bond_type = atom_->bond_type;
    tagint *tag = atom_->tag;
    int newton_bond = force->newton_bond;

    assert(newton_bond);

    int lostbond = output->thermo->lostbond;
    int nmissing = 0;
    nbondlist = 0;

    /*
    for (i = 0; i < nlocal; i++) {
        for (m = 0; m < num_bond[i]; m++) {
            atom1 = atom_->map(bond_atom[i][m]);
            if (atom1 == -1) {
                nmissing++;
                if (lostbond == Thermo::ERROR)
                    error->one(FLERR,
                               "Bond atoms {} {} missing on "
                               "proc {} at step {}",
                               tag[i], bond_atom[i][m], me, update->ntimestep);
                continue;
            }
            // atom1 = domain->closest_image(i, atom1);
            int tmp = atom1;
            atom1 = closest_image(atom_, i, atom1);
            assert(tmp == atom1);

            if (newton_bond || i < atom1) {
                if (nbondlist == maxbond) {
                    maxbond += DELTA;
                    memory->grow(bondlist, maxbond, 3, "neigh_topo:bondlist");
                }
                bondlist[nbondlist][0] = i;
                bondlist[nbondlist][1] = atom1;
                bondlist[nbondlist][2] = bond_type[i][m];
                nbondlist++;
            }
        }
    }
    */

    for (i = 0; i < nlocal + atom_->nghost; i++) {
        for (m = 0; m < num_bond[i]; m++) {
            atom1 = atom_->map(bond_atom[i][m]);
            if (atom1 == -1) {
                nmissing++;
                /*
                if (lostbond == Thermo::ERROR) {
                    error->one(FLERR,
                               "Bond atoms {} {} missing on "
                               "proc {} at step {}",
                               tag[i], bond_atom[i][m], me, update->ntimestep);
                }
                */
                continue;
            }
            // atom1 = domain->closest_image(i, atom1);
            int tmp = atom1;
            atom1 = closest_image(atom_, i, atom1);
            assert(tmp == atom1);

            if (i >= nlocal && atom1 >= nlocal) {
                continue;
            }

            if (i >= nlocal) {
                bool shrinking_out_of_bounds = false;
                bool expanding_out_of_bounds = false;

                for (int dim = 0; dim < 3; dim++) {
                    double lo = domain_->sublo[dim];
                    double hi = domain_->subhi[dim];
                    bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                    bool out_of_bounds = (atom_->x[i][dim] < lo || atom_->x[i][dim] >= hi);
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

            if (atom1 >= nlocal) {
                bool shrinking_out_of_bounds = false;
                bool expanding_out_of_bounds = false;

                for (int dim = 0; dim < 3; dim++) {
                    double lo = domain_->sublo[dim];
                    double hi = domain_->subhi[dim];
                    bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                    bool out_of_bounds = (atom_->x[atom1][dim] < lo || atom_->x[atom1][dim] >= hi);
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

            if (newton_bond || i < atom1) {
                if (nbondlist == maxbond) {
                    maxbond += DELTA;
                    memory->grow(bondlist, maxbond, 3, "neigh_topo:bondlist");
                }
                bondlist[nbondlist][0] = i;
                bondlist[nbondlist][1] = atom1;
                bondlist[nbondlist][2] = bond_type[i][m];
                nbondlist++;
            }
        }
    }


    if (cluster_check) {
        bond_check_stencil_md(atom_);
    }
    if (lostbond == Thermo::IGNORE) return;

    int all;
    MPI_Allreduce(&nmissing, &all, 1, MPI_INT, MPI_SUM, world);
    if (all && (me == 0)) error->warning(FLERR, "Bond atoms missing at step {}", update->ntimestep);
}
