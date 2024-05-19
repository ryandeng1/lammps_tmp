//
// Created by Ryan Deng on 2/27/24.
//

#pragma once

#include "pointers.h"
#include "force.h"
#include "modify.h"
#include "pair.h"
#include "bond.h"
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "neigh_list.h"

namespace LAMMPS_NS {

class StencilMD : protected Pointers {
public:
    StencilMD(class LAMMPS *lmp) : Pointers(lmp) {}

    void MODIFY_ADD_FIX_STENCIL_MD(int narg, char **arg);

    void MODIFY_ADD_FIX_PACKAGE_STENCIL_MD(const std::string& fixcmd);

    void MODIFY_ADD_COMPUTE_STENCIL_MD(int narg, char **arg);

    void ATOM_STYLE(const std::string &style, int narg, char **arg, int trysuffix);

    void ATOM_SETTINGS();

    void FORCE_SET_SPECIAL(int narg, char **arg);

    void FORCE_CREATE_BOND(const std::string& style, int trysuffix);

    void FORCE_BOND_SETTINGS(int narg, char **arg);

    void FORCE_BOND_COEFF(int narg, char **arg);

    void FORCE_PAIR_COEFF(int narg, char **arg);

    void FORCE_MODIFY_PARAMS(int narg, char **arg);

    void FORCE_CREATE_PAIR(const std::string& style, int trysuffix);

    void FORCE_PAIR_SETTINGS(int narg, char **arg);

    void CREATE();

    void CREATE_NEXT_DT();

    void INIT_ZOIDS();

    void INIT_ZOID_DATA();

    void INIT_ZOID_NEIGHBORS();

    void INIT_DOMAIN_BOUNDS();

    void INIT_ALL();

    void SETUP();

    void MODIFY_PRE_FORCE_SETUP(int);

    void MODIFY_SETUP(int);

    void GET_LOCAL_ATOMS_ZOID();

    void GET_GHOST_ATOMS_ZOID();

    void BUILD_NEIGHBOR_LIST();
    void BUILD_NEIGHBOR_LIST_NEXT_DT();

    void COMPUTE_NUM_SEND_RECV_PROCESS();

    void COMPARE_POS_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_x);

    void COMPARE_FORCE_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);
};

}