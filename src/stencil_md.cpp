//
// Created by Ryan Deng on 3/7/24.
//
#include <algorithm>
#include <set>

#include "stencil_md.h"
#include "comm.h"
#include "comm_brick.h"
#include "domain.h"
#include "accelerator_omp.h"
#include "modify.h"
#include "fix_langevin.h"
#include "bond_fene.h"
#include "pair_lj_cut.h"
#include <unordered_set>
#include <sstream>

using namespace LAMMPS_NS;

void StencilMD::ATOM_STYLE(const std::string &style, int narg, char **arg, int trysuffix) {
    for (int i = 0; i < lmp->atom_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->atom_stencil_md[i].size(); j++) {
                lmp->atom_stencil_md[i][j]->create_avec_stencil_md(style, narg, arg, trysuffix);
            }
        }
    }
}

void StencilMD::ATOM_SETTINGS() {
    for (int i = 0; i < lmp->atom_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->atom_stencil_md[i].size(); j++) {
                lmp->atom_stencil_md[i][j]->bond_per_atom = atom->bond_per_atom;
                lmp->atom_stencil_md[i][j]->maxspecial = atom->maxspecial;
            }
        }
    }
}

void StencilMD::MODIFY_ADD_FIX_STENCIL_MD(int narg, char **arg) {
#ifdef LMP_OPENMP
    for (int i = 0; i < lmp->modify_stencil_md_omp.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->modify_stencil_md_omp[i].size(); j++) {
                lmp->modify_stencil_md_omp[i][j]->add_fix(narg, arg, 1, true);
            }
        }
    }
#else
    for (int i = 0; i < lmp->modify_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            lmp->modify_stencil_md[i]->add_fix(narg, arg, 1, true);
        }
    }
#endif
}

void StencilMD::MODIFY_ADD_FIX_PACKAGE_STENCIL_MD(const std::string& fixcmd) {
#ifdef LMP_OPENMP
    for (int i = 0; i < lmp->modify_stencil_md_omp.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->modify_stencil_md_omp[i].size(); j++) {
                lmp->modify_stencil_md_omp[i][j]->add_fix(fixcmd, 1, true);
            }
        }
    }
#else
    for (int i = 0; i < lmp->modify_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            lmp->modify_stencil_md[i]->add_fix(fixcmd, 1, true);
        }
    }
#endif
}

void StencilMD::MODIFY_ADD_COMPUTE_STENCIL_MD(int narg, char **arg) {
#ifdef LMP_OPENMP
    for (int i = 0; i < lmp->modify_stencil_md_omp.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->modify_stencil_md_omp[i].size(); j++) {
                lmp->modify_stencil_md_omp[i][j]->add_compute(narg, arg, 1, true);
            }
        }
    }
#else
    for (int i = 0; i < lmp->modify_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            lmp->modify_stencil_md[i]->add_compute(narg, arg, 1, true);
        }
    }
#endif
}

void StencilMD::FORCE_SET_SPECIAL(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                lmp->force_stencil_md[i][j]->set_special(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->set_special(narg, arg);
            }
        }
    }
}

void StencilMD::FORCE_PAIR_COEFF(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                lmp->force_stencil_md[i][j]->pair->coeff(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->pair->coeff(narg, arg);
            }
        }
    }
}

void StencilMD::FORCE_BOND_COEFF(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                lmp->force_stencil_md[i][j]->bond->coeff(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->bond->coeff(narg, arg);
            }
        }
    }
}

void StencilMD::FORCE_MODIFY_PARAMS(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                lmp->force_stencil_md[i][j]->pair->modify_params(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->pair->modify_params(narg, arg);
            }
        }
    }
}

void StencilMD::FORCE_CREATE_PAIR(const std::string& style, int trysuffix) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                // curr-dt and next-dt force use the same modify. Will there be issues? Hopefully not?
#ifdef LMP_OPENMP
                lmp->force_stencil_md[i][j]->create_pair(style, trysuffix, true, lmp->modify_stencil_md_omp[i][j]);

                // TODO RYAN: force_stencil_md_next_dt does not behave the same as the other stuff. Since there is no modify_next_dt, we gotta do something about it.
                lmp->force_stencil_md_next_dt[i][j]->create_pair(style, trysuffix, true, lmp->modify_stencil_md_omp[i][NUM_TIMESTEPS_IN_PARALLEL - j]);
#else
                lmp->force_stencil_md[i][j]->create_pair(style, trysuffix, true, lmp->modify_stencil_md[i]);
                lmp->force_stencil_md_next_dt[i][j]->create_pair(style, trysuffix, true, lmp->modify_stencil_md[i]);
#endif
            }
        }
    }
}

void StencilMD::FORCE_CREATE_BOND(const std::string& style, int trysuffix) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                // curr-dt and next-dt force use the same modify. Will there be issues? Hopefully not?
#ifdef LMP_OPENMP
                lmp->force_stencil_md[i][j]->create_bond(style, trysuffix, true, lmp->modify_stencil_md_omp[i][j]);

                // TODO RYAN: force_stencil_md_next_dt does not behave the same as the other stuff. Since there is no modify_next_dt, we gotta do something about it.
                lmp->force_stencil_md_next_dt[i][j]->create_bond(style, trysuffix, true, lmp->modify_stencil_md_omp[i][NUM_TIMESTEPS_IN_PARALLEL - j]);
#else
                lmp->force_stencil_md[i][j]->create_bond(style, trysuffix, true, lmp->modify_stencil_md[i]);
                lmp->force_stencil_md_next_dt[i][j]->create_bond(style, trysuffix, true, lmp->modify_stencil_md[i]);
#endif
            }
        }
    }
}

void StencilMD::FORCE_PAIR_SETTINGS(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                assert(lmp->force_stencil_md[i][j]->pair != nullptr);
                lmp->force_stencil_md[i][j]->pair->settings(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->pair->settings(narg, arg);
            }
        }
    }
}

void StencilMD::FORCE_BOND_SETTINGS(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                assert(lmp->force_stencil_md[i][j]->bond != nullptr);
                lmp->force_stencil_md[i][j]->bond->settings(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->bond->settings(narg, arg);
            }
        }
    }
}

void StencilMD::CREATE() {
    assert(!lmp->kokkos);
    for (int i = 0; i < NUM_ZOIDS; i++) {
        std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_atom;
        lmp->atom_stencil_md.push_back(arr_atom);

        std::array<Domain *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_domain;
        lmp->domain_stencil_md.push_back(arr_domain);

        std::array<Neighbor *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_neighbor;
        lmp->neighbor_stencil_md.push_back(arr_neighbor);

        std::array<Force *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_force;
        lmp->force_stencil_md.push_back(arr_force);

#ifdef LMP_OPENMP
        std::array<Modify*, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_modify_omp;
        lmp->modify_stencil_md_omp.push_back(arr_modify_omp);
#endif
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
            Force* force_ = new Force(lmp);
            lmp->force_stencil_md[i][j] = force_;
        }

#ifdef LMP_OPENMP
        for (int j = 0; j < lmp->modify_stencil_md_omp[i].size(); j++) {
            Modify* modify_ = new Modify(lmp);
            lmp->modify_stencil_md_omp[i][j] = modify_;
        }
#else
        Modify* modify_;
        if (lmp->kokkos) {
            // modify_ = new ModifyKokkos(lmp);
            modify_ = nullptr;
        } else {
            modify_ = new Modify(lmp);
        }
        lmp->modify_stencil_md.push_back(modify_);
#endif

        for (int j = 0; j < lmp->atom_stencil_md[i].size(); j++) {
            Atom* atom_;
            if (lmp->kokkos) {
                // atom_ = new AtomKokkos(lmp);
                atom_ = nullptr;
            } else {
                atom_ = new Atom(lmp);
                // This is to help with some shenanigans?
            }

            if (lmp->kokkos) {
                atom_->create_avec_stencil_md("atomic/kk",0,nullptr,1);
            } else {
                atom_->create_avec_stencil_md("atomic", 0, nullptr, 1);
            }
            lmp->atom_stencil_md[i][j] = atom_;
        }

        Comm* comm_;
        if (lmp->kokkos) {
            // comm_ = new CommKokkos(lmp);
            comm_ = nullptr;
        } else {
            comm_ = new CommBrick(lmp);
        }

        for (int j = 0; j < lmp->domain_stencil_md[i].size(); j++) {
            Domain* domain_;
            if (lmp->kokkos) {
                // domain_ = new DomainKokkos(lmp);
                domain_ = nullptr;
            } else {
                domain_ = new Domain(lmp);
            }
            lmp->domain_stencil_md[i][j] = domain_;
        }

        for (int j = 0; j < lmp->neighbor_stencil_md[i].size(); j++) {
            Neighbor* neighbor_;
            if (lmp->kokkos) {
                // neighbor_ = new NeighborKokkos(lmp);
                neighbor_ = nullptr;
            } else {
                neighbor_ = new Neighbor(lmp);
            }
            lmp->neighbor_stencil_md[i][j] = neighbor_;
        }

        lmp->comm_stencil_md.push_back(comm_);
    }
}

void StencilMD::CREATE_NEXT_DT() {
    for (int i = 0; i < NUM_ZOIDS; i++) {
        std::array<Domain *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_domain;
        lmp->domain_stencil_md_next_dt.push_back(arr_domain);

        std::array<Neighbor *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_neighbor;
        lmp->neighbor_stencil_md_next_dt.push_back(arr_neighbor);

        std::array<Force *, NUM_TIMESTEPS_IN_PARALLEL + 1> arr_force;
        lmp->force_stencil_md_next_dt.push_back(arr_force);
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
            Force *force_ = new Force(lmp);
            lmp->force_stencil_md_next_dt[i][j] = force_;
        }

        for (int j = 0; j < lmp->domain_stencil_md[i].size(); j++) {
            Domain *domain_;
            if (lmp->kokkos) {
                // domain_ = new DomainKokkos(lmp);
                domain_ = nullptr;
            } else {
                domain_ = new Domain(lmp);
            }
            lmp->domain_stencil_md_next_dt[i][j] = domain_;
        }

        for (int j = 0; j < lmp->neighbor_stencil_md_next_dt[i].size(); j++) {
            Neighbor *neighbor_;
            if (lmp->kokkos) {
                // neighbor_ = new NeighborKokkos(lmp);
                neighbor_ = nullptr;
            } else {
                neighbor_ = new Neighbor(lmp);
            }
            lmp->neighbor_stencil_md_next_dt[i][j] = neighbor_;
        }
    }
}

void StencilMD::INIT_ZOIDS() {
    if (comm->me == 0) {
        std::cout << BOLDYELLOW << "INIT ZOIDS. LO: " << domain->boxlo[0] << " " << domain->boxlo[1] << " " << domain->boxlo[2] << RESET_COLOR << std::endl;
        std::cout << BOLDYELLOW << "INIT ZOIDS. HI: " << domain->boxhi[0] << " " << domain->boxhi[1] << " " << domain->boxhi[2] << RESET_COLOR << std::endl;
        std::cout << "map size: " << zoid_to_num_map.size() << std::endl;
    }

    get_zoids(ALLEGRO_SLOPE, domain->boxlo, domain->boxhi, lmp->queues);

    assert(zoid_to_num_map.size() == NUM_ZOIDS);
    std::set<int> zoid_nums;
    for (auto& [k, v] : zoid_to_num_map) {
        zoid_nums.insert(v);
    }
    assert(zoid_nums.size() == NUM_ZOIDS);

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            IDX_3D key = {zoid.where[0], zoid.where[1], zoid.where[2]};
            if (zoid_to_num_map.find(key) == zoid_to_num_map.end()) {
                std::cout << "error. key: " << std::get<0>(key) << " "
                          << std::get<1>(key) << " " << std::get<2>(key)
                          << std::endl;
                assert(false);
            }
            lmp->queues[dep][j].num = zoid_to_num_map.at(key);
            assert(zoid.num >= 0 && zoid.num < NUM_ZOIDS);
        }
    }

    if (comm->me == 0) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::vector<int> nums;
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                nums.push_back(lmp->queues[dep][j].num);
            }

            // std::cout << "DEP: " << dep << " zoid nums: " << nums << std::endl;
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            const queue_info& zoid = lmp->queues[dep][j];
            int new_dep = NUM_DEPS - 1 - dep;
            queue_info new_zoid;
            cuts_t new_cuts_t;
            for (int dim = 0; dim < 3; dim++) {
                double new_start =
                        zoid.zoid.cuts[dim].lower +
                        NUM_TIMESTEPS_IN_PARALLEL * zoid.zoid.cuts[dim].slope_lower;
                double new_end =
                        zoid.zoid.cuts[dim].upper +
                        NUM_TIMESTEPS_IN_PARALLEL * zoid.zoid.cuts[dim].slope_upper;
                new_cuts_t.cuts[dim].lower = new_start;
                new_cuts_t.cuts[dim].upper = new_end;
                new_cuts_t.cuts[dim].slope_lower =
                        -1 * zoid.zoid.cuts[dim].slope_lower;
                new_cuts_t.cuts[dim].slope_upper =
                        -1 * zoid.zoid.cuts[dim].slope_upper;
            }
            new_zoid.zoid = new_cuts_t;
            new_zoid.num = zoid.num;
            for (int i = 0; i < 3; i++) {
                new_zoid.where[i] = zoid.where[i];
            }
            lmp->queues_next_dt[new_dep].push_back(new_zoid);
        }
    }
}

void StencilMD::INIT_ZOID_DATA() {
    std::map<int, std::pair<int, int>> zoid_num_to_coord;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
                zoid_num_to_coord[zoid.num] = {dep, j};
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
                /* start stuff for 2 timesteps */
                zoid.per_worker_force_updates = new std::pair<int, dbl3_t_stencil_md>*[__cilkrts_get_nworkers()];
                for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
                    zoid.per_worker_force_updates[i] = new std::pair<int, dbl3_t_stencil_md>[MODIFY_GRAINSIZE * MAX_NEIGHBORS_PER_ATOM];
                }

                zoid.space_cut_idxs = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.x_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                // zoid.v_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                // zoid.f_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                // zoid.eval_f_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                zoid.v_stencil_md = new std::vector<dbl3_t_stencil_md>[1];
                zoid.f_stencil_md = new std::vector<dbl3_t_stencil_md>[1];
                zoid.eval_f_stencil_md = new std::vector<dbl3_t_stencil_md>[1];

                zoid.tag_stencil_md = new std::vector<int>[1];
                zoid.type_stencil_md = new std::vector<int>[1];
                zoid.mask_stencil_md = new std::vector<int>[1];
                zoid.image_stencil_md = new std::vector<int>[1];
                zoid.spinlocks_stencil_md = new spinlock*[1];
                zoid.claimed_flags_stencil_md = new std::atomic_flag*[1];

                zoid.local_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.neighbor_list = new std::vector<std::vector<int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bond_list = new std::vector<std::vector<std::pair<int, int>>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.neighbor_list = new std::vector<int>*[1];
                // zoid.bond_list = new std::vector<std::pair<int, int>>*[1];

                zoid.send_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_pos_local_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_pos_ghost_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                /* end stuff for 2 timesteps */

                int num_bins_3d = NUM_BINS * NUM_BINS * NUM_BINS;

                zoid.local_bins_comm = new std::vector<bool>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.no_comm_local_bins = new std::vector<IDX_3D>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.comm_local_bins = new std::vector<IDX_3D>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_force_comm = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_pos_vel_comm = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.bin_to_num_send_zoids = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_send_zoids = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.bin_to_idx = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_size = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.bin_to_idx[t] = new int[num_bins_3d];
                    zoid.bin_to_size[t] = new int[num_bins_3d];

                    zoid.bin_to_num_send_zoids[t] = new int[num_bins_3d];
                    zoid.bin_to_send_zoids[t] = new int*[num_bins_3d];

                    for (int i = 0; i < num_bins_3d; i++) {
                        zoid.bin_to_idx[t][i] = -1;
                        zoid.bin_to_size[t][i] = -1;
                        zoid.bin_to_num_send_zoids[t][i] = -1;
                    }
                }

                zoid.send_force_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.send_force_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.send_pos_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_vel_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.send_vel_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_vel_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_force_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.recv_force_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_force_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_pos_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.recv_pos_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_pos_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_vel_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.recv_vel_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_vel_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.inum_per_timestep = new int[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.num_send_process_timestep = new int*[comm->nprocs];
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    zoid.num_send_process_timestep[proc] = new int[NUM_TIMESTEPS_IN_PARALLEL + 1];
                }

                zoid.num_send_process = new int[comm->nprocs];
                zoid.num_recv_process = new int[comm->nprocs];
                // debugging
                zoid.debug_atom_pos =
                        new double*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.can_eval_center = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.can_eval_pos = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_size =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for send list
                zoid.send_force_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_total_num_elems = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_total_num_elems = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_force_only =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_num_force_only =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_force_pos =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_num_force_pos =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for second send list
                zoid.send_segment_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_types =
                        new bool**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_idxs =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_local_list = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // need to init this so that "copies" can be made
                zoid.num_elems_send = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.reverse_atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.reverse_atom_idx_mapping_idxs = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_process_segment_types =
                        new bool**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_idxs =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_process_segment_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_idxs =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_types =
                        new bool**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_local_list =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_process_segment_sizes[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_idxs[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_types[t] = new bool*[comm->nprocs];
                    zoid.send_process_num_segments[t] = new int[comm->nprocs];
                    zoid.send_process_local_list[t] = new int*[comm->nprocs];
                }

                zoid.num_elems_send_process =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv_process =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send_process[t] = new int[comm->nprocs];
                    zoid.num_elems_recv_process[t] = new int[comm->nprocs];
                }

                zoid.recv_process_force_offset =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_vel_offset =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_pos_offset =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.relevant_atom_idxs =
                        new std::set<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.relevant_atom_tags =
                        new std::set<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.can_eval_center_tags =
                        new std::set<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
                /* start stuff for 2 timesteps */
                zoid.per_worker_force_updates = new std::pair<int, dbl3_t_stencil_md>*[__cilkrts_get_nworkers()];
                for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
                    zoid.per_worker_force_updates[i] = new std::pair<int, dbl3_t_stencil_md>[MODIFY_GRAINSIZE * MAX_NEIGHBORS_PER_ATOM];
                }
                zoid.space_cut_idxs = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // Copy the main data from the curr_dt zoid
                auto coord = zoid_num_to_coord[zoid.num];

                zoid.x_stencil_md = lmp->queues[coord.first][coord.second].x_stencil_md;
                zoid.v_stencil_md = lmp->queues[coord.first][coord.second].v_stencil_md;
                zoid.f_stencil_md = lmp->queues[coord.first][coord.second].f_stencil_md;
                zoid.eval_f_stencil_md = lmp->queues[coord.first][coord.second].eval_f_stencil_md;
                zoid.tag_stencil_md = lmp->queues[coord.first][coord.second].tag_stencil_md;
                zoid.type_stencil_md = lmp->queues[coord.first][coord.second].type_stencil_md;
                zoid.mask_stencil_md = lmp->queues[coord.first][coord.second].mask_stencil_md;
                zoid.image_stencil_md = lmp->queues[coord.first][coord.second].image_stencil_md;
                zoid.spinlocks_stencil_md = lmp->queues[coord.first][coord.second].spinlocks_stencil_md;
                zoid.claimed_flags_stencil_md = lmp->queues[coord.first][coord.second].claimed_flags_stencil_md;

                zoid.local_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.neighbor_list = new std::vector<std::vector<int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bond_list = new std::vector<std::vector<std::pair<int, int>>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.neighbor_list = lmp->queues[coord.first][coord.second].neighbor_list;
                // zoid.bond_list = lmp->queues[coord.first][coord.second].bond_list;

                zoid.send_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_pos_local_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_pos_ghost_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                /* end stuff for 2 timesteps */

                int num_bins_3d = NUM_BINS * NUM_BINS * NUM_BINS;
                zoid.local_bins_comm = new std::vector<bool>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.no_comm_local_bins = new std::vector<IDX_3D>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.comm_local_bins = new std::vector<IDX_3D>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_force_comm = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_pos_vel_comm = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.bin_to_num_send_zoids = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_send_zoids = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.bin_to_idx = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.bin_to_size = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.bin_to_idx[t] = new int[num_bins_3d];
                    zoid.bin_to_size[t] = new int[num_bins_3d];

                    zoid.bin_to_num_send_zoids[t] = new int[num_bins_3d];
                    zoid.bin_to_send_zoids[t] = new int*[num_bins_3d];

                    for (int i = 0; i < num_bins_3d; i++) {
                        zoid.bin_to_idx[t][i] = -1;
                        zoid.bin_to_size[t][i] = -1;
                        zoid.bin_to_num_send_zoids[t][i] = -1;
                    }
                }

                zoid.send_force_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.send_force_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.send_pos_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_vel_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.send_vel_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_vel_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_force_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.recv_force_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_force_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_pos_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.recv_pos_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_pos_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_vel_num_bins = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                // zoid.recv_vel_bins = new std::tuple<int, int, int>**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_vel_bins = new IDX_3D**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.inum_per_timestep = new int[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.num_send_process_timestep = new int*[comm->nprocs];
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    zoid.num_send_process_timestep[proc] = new int[NUM_TIMESTEPS_IN_PARALLEL + 1];
                }

                zoid.num_send_process = new int[comm->nprocs];
                zoid.num_recv_process = new int[comm->nprocs];

                zoid.debug_atom_pos =
                        new double*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.can_eval_center = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.can_eval_pos = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_size =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for send list
                zoid.send_force_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_total_num_elems = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_total_num_elems = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_force_only =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_force_pos =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_num_force_only =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_num_force_pos =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for second send list
                zoid.send_segment_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_types =
                        new bool**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_idxs =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_local_list = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.num_elems_send = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.reverse_atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.reverse_atom_idx_mapping_idxs = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_process_segment_types =
                        new bool**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_idxs =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_process_segment_sizes =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_idxs =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_types =
                        new bool**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_local_list =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_process_segment_sizes[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_idxs[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_types[t] = new bool*[comm->nprocs];
                    zoid.send_process_num_segments[t] = new int[comm->nprocs];
                    zoid.send_process_local_list[t] = new int*[comm->nprocs];
                }

                zoid.num_elems_send_process =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv_process =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send_process[t] = new int[comm->nprocs];
                    zoid.num_elems_recv_process[t] = new int[comm->nprocs];
                }

                zoid.recv_process_force_offset =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_vel_offset =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_pos_offset =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.relevant_atom_idxs =
                        new std::set<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.relevant_atom_tags =
                        new std::set<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.can_eval_center_tags =
                        new std::set<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
            }
        }
    }

    lmp->zoid_num_to_zoid = new queue_info[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            lmp->zoid_num_to_zoid[zoid_num] = lmp->queues[dep][j];
        }
    }

    lmp->zoid_num_to_zoid_next_dt = new queue_info[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            int zoid_num = lmp->queues_next_dt[dep][j].num;
            lmp->zoid_num_to_zoid_next_dt[zoid_num] = lmp->queues_next_dt[dep][j];
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->my_queues[dep].push_back(lmp->queues[dep][j]);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            int zoid_num = lmp->queues_next_dt[dep][j].num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->my_queues_next_dt[dep].push_back(lmp->queues_next_dt[dep][j]);
            }
        }
    }
}

void StencilMD::INIT_ZOID_NEIGHBORS() {
    // for neighbors
    lmp->send_to_neighbors = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_neighbors = new std::vector<int>[NUM_ZOIDS];

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& zoid = lmp->zoid_num_to_zoid[i];
        int zoid_dep = get_zoid_dep(zoid.num);
        for (int j = 0; j < NUM_ZOIDS; j++) {
            int zoid_dep_neighbor = get_zoid_dep(j);
            if (zoid_dep_neighbor > zoid_dep &&
                is_close_test(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // if (zoid_dep_neighbor > zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // TODO: test this extra condition
                if (zoid_dep_neighbor == zoid_dep + 1 || true) {
                    lmp->send_to_neighbors[i].push_back(j);
                }
            }

            // if (zoid_dep_neighbor < zoid_dep && is_close(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
            if (zoid_dep_neighbor < zoid_dep &&
                is_close_test(lmp->zoid_num_to_zoid[j].where, zoid.where)) {
                // if (zoid_dep_neighbor < zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                if (zoid_dep_neighbor == zoid_dep - 1 || true) {
                    lmp->recv_from_neighbors[i].push_back(j);
                }
            }
        }
    }

    for (int recv_zoid_num = 0; recv_zoid_num < NUM_ZOIDS; recv_zoid_num++) {
        for (int my_zoid_num = 0; my_zoid_num < NUM_ZOIDS; my_zoid_num++) {
            if (my_zoid_num % comm->nprocs == comm->me) {
                auto& recv_from = lmp->recv_from_neighbors[my_zoid_num];
                if (std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) != recv_from.end()) {
                    int recv_idx = std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) - recv_from.begin();
                    lmp->recv_zoid_to_my_zoids[recv_zoid_num].push_back({my_zoid_num, recv_idx});
                }
            }
        }
    }

    //  for next dt
    lmp->send_to_neighbors_next_dt = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_neighbors_next_dt = new std::vector<int>[NUM_ZOIDS];

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& zoid = lmp->zoid_num_to_zoid_next_dt[i];
        int zoid_dep = get_zoid_dep(zoid.num);
        zoid_dep = NUM_DEPS - 1 - zoid_dep;
        for (int j = 0; j < NUM_ZOIDS; j++) {
            int zoid_dep_neighbor = get_zoid_dep(j);
            zoid_dep_neighbor = NUM_DEPS - 1 - zoid_dep_neighbor;
            if (zoid_dep_neighbor > zoid_dep &&
                is_close_test_next_dt(zoid.where,
                                      lmp->zoid_num_to_zoid_next_dt[j].where)) {
                // if (zoid_dep_neighbor > zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // TODO: test this extra condition
                if (zoid_dep_neighbor == zoid_dep + 1 || true) {
                    lmp->send_to_neighbors_next_dt[i].push_back(j);
                }
            }

            // if (zoid_dep_neighbor < zoid_dep && is_close(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
            if (zoid_dep_neighbor < zoid_dep &&
                is_close_test_next_dt(lmp->zoid_num_to_zoid_next_dt[j].where,
                                      zoid.where)) {
                // if (zoid_dep_neighbor < zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                if (zoid_dep_neighbor == zoid_dep - 1 || true) {
                    lmp->recv_from_neighbors_next_dt[i].push_back(j);
                }
            }
        }
    }

    for (int recv_zoid_num = 0; recv_zoid_num < NUM_ZOIDS; recv_zoid_num++) {
        for (int my_zoid_num = 0; my_zoid_num < NUM_ZOIDS; my_zoid_num++) {
            if (my_zoid_num % comm->nprocs == comm->me) {
                auto& recv_from = lmp->recv_from_neighbors_next_dt[my_zoid_num];
                if (std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) != recv_from.end()) {
                    int recv_idx = std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) - recv_from.begin();
                    lmp->recv_zoid_to_my_zoids_next_dt[recv_zoid_num].push_back({my_zoid_num, recv_idx});
                }
            }
        }
    }

    lmp->send_to_neighbors_procs = new std::unordered_set<int>[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;

            auto& send_to = lmp->send_to_neighbors[zoid_num];
            std::set<int> send_procs;
            for (int send_zoid : send_to) {
                send_procs.insert(send_zoid % comm->nprocs);
            }

            for (int proc : send_procs) {
                lmp->send_to_neighbors_procs[zoid_num].insert(proc);

                if (proc == comm->me &&
                    std::find(lmp->recv_from_neighbors_procs.begin(),
                              lmp->recv_from_neighbors_procs.end(), zoid_num) ==
                    lmp->recv_from_neighbors_procs.end()) {
                    lmp->recv_from_neighbors_procs.push_back(zoid_num);
                }
            }
        }
    }

    for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
        lmp->recv_from_neighbors_procs_idxs[lmp->recv_from_neighbors_procs[i]] = i;
    }

    lmp->send_to_neighbors_procs_next_dt = new std::unordered_set<int>[NUM_ZOIDS];
    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& send_to = lmp->send_to_neighbors_next_dt[i];
        std::set<int> send_procs;
        for (int send_zoid : send_to) {
            send_procs.insert(send_zoid % comm->nprocs);
        }

        for (int proc : send_procs) {
            lmp->send_to_neighbors_procs_next_dt[i].insert(proc);

            if (proc == comm->me &&
                std::find(lmp->recv_from_neighbors_procs_next_dt.begin(),
                          lmp->recv_from_neighbors_procs_next_dt.end(),
                          i) == lmp->recv_from_neighbors_procs_next_dt.end()) {
                lmp->recv_from_neighbors_procs_next_dt.push_back(i);
            }
        }
    }

    for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
        lmp->recv_from_neighbors_procs_idxs_next_dt[lmp->recv_from_neighbors_procs_next_dt[i]] = i;
    }

}

void StencilMD::INIT_DOMAIN_BOUNDS() {
    // set the domains for each zoid
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            for (int k = 0; k < lmp->domain_stencil_md[j].size(); k++) {
                Domain* domain_ = lmp->domain_stencil_md[zoid_num][k];
                for (int dim = 0; dim < 3; dim++) {
                    domain_->sublo[dim] = zoid.zoid.cuts[dim].lower +
                                          k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->subhi[dim] = zoid.zoid.cuts[dim].upper +
                                          k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->boxlo[dim] = zoid.zoid.cuts[dim].lower +
                                          k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->boxhi[dim] = zoid.zoid.cuts[dim].upper +
                                          k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->prd[dim] = domain->prd[dim];
                    assert(domain_->sublo[dim] < domain_->subhi[dim]);
                }
            }
        }
    }

    // set the domains for each zoid
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            for (int k = 0; k < lmp->domain_stencil_md_next_dt[j].size(); k++) {
                Domain* domain_ = lmp->domain_stencil_md_next_dt[zoid_num][k];
                for (int dim = 0; dim < 3; dim++) {
                    domain_->sublo[dim] = zoid.zoid.cuts[dim].lower +
                                          k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->subhi[dim] = zoid.zoid.cuts[dim].upper +
                                          k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->boxlo[dim] = zoid.zoid.cuts[dim].lower +
                                          k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->boxhi[dim] = zoid.zoid.cuts[dim].upper +
                                          k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->prd[dim] = domain->prd[dim];
                }
            }
        }
    }
}

void StencilMD::INIT_ALL() {
    // init
    int use_omp = 0;
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                Force *force_ = lmp->force_stencil_md[i][j];
                Neighbor *neighbor_ = lmp->neighbor_stencil_md[i][j];
                Domain *domain_ = lmp->domain_stencil_md[i][j];
                force_->init_stencil_md(neighbor_);
                domain_->init();
            }

            Comm *comm_ = lmp->comm_stencil_md[i];

            for (int j = 0; j < lmp->atom_stencil_md[i].size(); j++) {
                Atom *atom_ = lmp->atom_stencil_md[i][j];
                atom_->init();
            }

#ifdef LMP_OPENMP
            for (int j = 0; j < lmp->modify_stencil_md_omp[i].size(); j++) {
                Modify* modify_ = lmp->modify_stencil_md_omp[i][j];
                // TODO: watch out. neighbor has a next_dt as well. This right now is only meant to propagate the npair-omp'ness over
                // so that the neighbor list will call the omp-version of the build method
                modify_->init_stencil_md(lmp->atom_stencil_md[i][j], lmp->neighbor_stencil_md[i][j]);
                use_omp = lmp->neighbor_stencil_md[i][j]->get_omp_neighbor();
            }
#else
            Modify *modify_ = lmp->modify_stencil_md[i];
            modify_->init_stencil_md(lmp->atom_stencil_md[i][0]);
#endif

            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                Neighbor *neighbor_ = lmp->neighbor_stencil_md[i][j];
                Domain *domain_ = lmp->domain_stencil_md[i][j];
                neighbor_->init_stencil_md(domain_);
            }
            comm_->init();
        }
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md_next_dt[i].size(); j++) {
                Force *force_ = lmp->force_stencil_md_next_dt[i][j];
                Neighbor *neighbor_next_dt = lmp->neighbor_stencil_md_next_dt[i][j];
                Domain *domain_ = lmp->domain_stencil_md_next_dt[i][j];
                force_->init_stencil_md(neighbor_next_dt);
                domain_->init();
            }

            for (int j = 0; j < lmp->force_stencil_md_next_dt[i].size(); j++) {
                Neighbor *neighbor_next_dt = lmp->neighbor_stencil_md_next_dt[i][j];
                Domain *domain_ = lmp->domain_stencil_md_next_dt[i][j];
                if (use_omp) {
                    neighbor_next_dt->set_omp_neighbor(use_omp);
                }
                neighbor_next_dt->init_stencil_md(domain_);
            }
        }
    }
}

void StencilMD::SETUP() {
    // atom setup
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                for (int k = 0; k < lmp->atom_stencil_md[zoid_num].size();
                     k++) {
                    lmp->atom_stencil_md[zoid_num][k]->setup_stencil_md(
                            lmp->domain_stencil_md[zoid_num][k]);
                }
            }
        }
    }
}

void StencilMD::MODIFY_PRE_FORCE_SETUP(int vflag) {
    // atom setup
#ifdef LMP_OPENMP
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Modify* modify_ = lmp->modify_stencil_md_omp[zoid_num][t];
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                modify_->setup_pre_force_stencil_md(vflag, atom_);
            }
        }
    }
#endif
}

void StencilMD::MODIFY_SETUP(int vflag) {
    assert(false);
#ifdef LMP_OPENMP
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Modify* modify_ = lmp->modify_stencil_md_omp[zoid_num][t];
                modify_->setup(vflag);
                // modify_->setup_stencil_md(vflag, lmp->atom_stencil_md[zoid_num][t]);
            }
        }
    }
#endif
}

void StencilMD::GET_LOCAL_ATOMS_ZOID() {
    // get local atoms for each zoid
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        std::vector<MPI_Request> r(2 * NUM_ZOIDS, MPI_REQUEST_NULL);
        comm->exchange_stencil_md_initial_send(r);
        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* first = lmp->atom_stencil_md[zoid_num][t];
                    lmp->comm_stencil_md[zoid_num]
                            ->exchange_stencil_md_initial_receive(
                                    first, lmp->domain_stencil_md[zoid_num][t], zoid);
                    first->sort_stencil_md();
                }
            }
        }

        MPI_Barrier(world);
        MPI_Waitall(r.size(), r.data(), MPI_STATUSES_IGNORE);
    }
}

void StencilMD::GET_LOCAL_ATOMS_ZOID_DOUBLE_BUFFERING() {
    // get local atoms for each zoid using double buffering
    for (int t = 0; t < DOUBLE_BUFFERING; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    std::set<tagint> all_tags;
                    for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][t2];
                        for (int i = 0; i < atom_->nlocal; i++) {
                            tagint tag_ = atom_->tag[i];
                            if (all_tags.find(tag_) == all_tags.end()) {
                                all_tags.insert(tag_);

                                double* x = atom_->x[i];
                                double* v = atom_->v[i];
                                int type_ = atom_->type[i];
                                int mask_ = atom_->mask[i];
                                int image_ = atom_->image[i];

                                zoid.x_stencil_md[t].push_back({x[0], x[1], x[2]});
                                // zoid.v_stencil_md[t].push_back({v[0], v[1], v[2]});
                                // zoid.f_stencil_md[t].push_back({0.0, 0.0, 0.0});
                                // zoid.eval_f_stencil_md[t].push_back({0.0, 0.0, 0.0});

                                if (t == 0) {
                                    zoid.tag_stencil_md[t].push_back(tag_);
                                    zoid.type_stencil_md[t].push_back(type_);
                                    zoid.mask_stencil_md[t].push_back(mask_);
                                    zoid.image_stencil_md[t].push_back(image_);
                                    zoid.f_stencil_md[t].push_back({0.0, 0.0, 0.0});
                                    zoid.eval_f_stencil_md[t].push_back({0.0, 0.0, 0.0});
                                    zoid.v_stencil_md[t].push_back({v[0], v[1], v[2]});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void StencilMD::GET_GHOST_ATOMS_ZOID() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        std::vector<MPI_Request> r(2 * NUM_ZOIDS, MPI_REQUEST_NULL);
        // comm->exchange_stencil_md_initial_send(r);
        comm->borders_stencil_md_initial_send(r);
        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* first = lmp->atom_stencil_md[zoid_num][t];
                    lmp->comm_stencil_md[zoid_num]->borders_stencil_md_initial_receive_from_lammps(
                            first, lmp->domain_stencil_md[zoid_num][t], zoid, t);
                }
            }
        }

        MPI_Barrier(world);
        MPI_Waitall(r.size(), r.data(), MPI_STATUSES_IGNORE);
    }
}

void StencilMD::GET_GHOST_ATOMS_ZOID_DOUBLE_BUFFERING() {
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                auto& my_tags = zoid.tag_stencil_md[0];
                int num_ghosts_added = 0;
                for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t2];
                    for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
                        auto tag_ = atom_->tag[i];
                        if (std::find(my_tags.begin(), my_tags.end(), tag_) == my_tags.end()) {
                            num_ghosts_added++;
                            double* x = atom_->x[i];
                            double* v = atom_->v[i];
                            int type_ = atom_->type[i];
                            int mask_ = atom_->mask[i];
                            int image_ = atom_->image[i];

                            for (int t = 0; t < DOUBLE_BUFFERING; t++) {
                                zoid.x_stencil_md[t].push_back({x[0], x[1], x[2]});
                                // zoid.v_stencil_md[t].push_back({v[0], v[1], v[2]});
                                // zoid.f_stencil_md[t].push_back({0.0, 0.0, 0.0});
                                // zoid.eval_f_stencil_md[t].push_back({0.0, 0.0, 0.0});
                            }

                            zoid.tag_stencil_md[0].push_back(tag_);
                            zoid.type_stencil_md[0].push_back(type_);
                            zoid.mask_stencil_md[0].push_back(mask_);
                            zoid.image_stencil_md[0].push_back(image_);
                            zoid.f_stencil_md[0].push_back({0.0, 0.0, 0.0});
                            zoid.eval_f_stencil_md[0].push_back({0.0, 0.0, 0.0});
                            zoid.v_stencil_md[0].push_back({v[0], v[1], v[2]});
                        }
                    }
                }

                // zoid.spinlocks_stencil_md[0] = new std::vector<spinlock>();
                // zoid.spinlocks_stencil_md[0].resize(zoid.x_stencil_md[0].size());
                zoid.spinlocks_stencil_md[0] = new spinlock[zoid.x_stencil_md[0].size()];
            }
        }
    }
}

void StencilMD::SORT_LOCAL_ATOMS_BINS() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    if (comm->nprocs == 1) {
                        auto& atom_arr = lmp->atom_stencil_md[zoid_num];
                        Atom* first = atom_arr[t];
                        auto& bin_bounds = GET_BOUNDS(true, t);
                        Atom* next;
                        if (t < NUM_TIMESTEPS_IN_PARALLEL) {
                            next = atom_arr[t + 1];
                        } else {
                            next = atom_arr[t - 1];
                        }
                        stencilMD->sort_local_bins(zoid, t, first, next);
                        auto& sorted_local_bin_indices = first->sorted_local_bin_indices;
                        first->sort_local_stencil_md_bins(bin_bounds, sorted_local_bin_indices);
                        first->setup_stencil_md_pair_bins(zoid, t);
                    }
                }
            }
        }
    }
}

void StencilMD::SORT_LOCAL_ATOMS_DOUBLE_BUFFERING() {
    // setup lammps code
    double binsize = 0.5 * neighbor->cutneighmax;
    double bininv = 1.0/binsize;

    int nbinx = static_cast<int> ((domain->boxhi[0] - domain->boxlo[0]) * bininv);
    int nbiny = static_cast<int> ((domain->boxhi[1] - domain->boxlo[1]) * bininv);
    int nbinz = static_cast<int> ((domain->boxhi[2] - domain->boxlo[2]) * bininv);

    double bininvx = nbinx / (domain->boxhi[0] - domain->boxlo[0]);
    double bininvy = nbiny / (domain->boxhi[1] - domain->boxlo[1]);
    double bininvz = nbinz / (domain->boxhi[2] - domain->boxlo[2]);

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                std::map<tagint, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                auto permutation = sort_permutation(zoid.tag_stencil_md[0],
                                          [&](const tagint& tag_a, const tagint& tag_b){
                      int idx_a = tag_to_idx[tag_a];
                      int idx_b = tag_to_idx[tag_b];

                      // sort based on last timestep they are in the zoid
                      // int last_timestep_a = -1;
                      // int last_timestep_b = -1;
                      int last_timestep_a = 0;
                      int last_timestep_b = 0;

                      std::set<int> timesteps_a;
                      std::set<int> timesteps_b;

                      for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                          bool in_zoid = true;
                          double pos[3] = {zoid.x_stencil_md[0][idx_a].x,
                                           zoid.x_stencil_md[0][idx_a].y,
                                           zoid.x_stencil_md[0][idx_a].z};

                          for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                              double lo = zoid.zoid.cuts[dim].lower + t2 * zoid.zoid.cuts[dim].slope_lower;
                              double hi = zoid.zoid.cuts[dim].upper + t2 * zoid.zoid.cuts[dim].slope_upper;
                              if (!(pos[dim] >= lo && pos[dim] < hi)) {
                                  in_zoid = false;
                              }
                          }

                          if (in_zoid) {
                              // last_timestep_a = t2;
                              last_timestep_a++;
                              timesteps_a.insert(t2);
                          }
                      }

                      for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                          bool in_zoid = true;
                          double pos[3] = {zoid.x_stencil_md[0][idx_b].x,
                                           zoid.x_stencil_md[0][idx_b].y,
                                           zoid.x_stencil_md[0][idx_b].z};

                          for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                              double lo = zoid.zoid.cuts[dim].lower + t2 * zoid.zoid.cuts[dim].slope_lower;
                              double hi = zoid.zoid.cuts[dim].upper + t2 * zoid.zoid.cuts[dim].slope_upper;
                              if (!(pos[dim] >= lo && pos[dim] < hi)) {
                                  in_zoid = false;
                              }
                          }

                          if (in_zoid) {
                              // last_timestep_b = t2;
                              last_timestep_b++;
                              timesteps_b.insert(t2);
                          }
                      }

                      if (last_timestep_a != last_timestep_b) {
                          return last_timestep_a > last_timestep_b;
                      }

                      if (timesteps_a != timesteps_b) {
                          return timesteps_a > timesteps_b;
                      }

                      /*
                      if (get_zoid_dep(zoid.num) == 0 || get_zoid_dep(zoid.num) == NUM_DEPS - 1) {
                          std::set<int> timesteps_local_a;
                          std::set<int> timesteps_local_b;

                          int num_timesteps_to_eval = (zoid.zoid.cuts[0].upper - zoid.zoid.cuts[0].lower) / (2 * fabs(zoid.zoid.cuts[0].slope_lower));

                          for (int t2 = 0; t2 < num_timesteps_to_eval; t2++) {
                              bool in_zoid = true;
                              double pos[3] = {zoid.x_stencil_md[0][idx_a].x,
                                               zoid.x_stencil_md[0][idx_a].y,
                                               zoid.x_stencil_md[0][idx_a].z};

                              for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                                  double lo = zoid.zoid.cuts[dim].lower + t2 * zoid.zoid.cuts[dim].slope_lower;
                                  double hi = zoid.zoid.cuts[dim].upper + t2 * zoid.zoid.cuts[dim].slope_upper;
                                  if (!(pos[dim] >= lo && pos[dim] < hi)) {
                                      in_zoid = false;
                                  }
                              }

                              if (in_zoid) {
                                  timesteps_local_a.insert(t2);
                              }
                          }

                          for (int t2 = 0; t2 < num_timesteps_to_eval; t2++) {
                              bool in_zoid = true;
                              double pos[3] = {zoid.x_stencil_md[0][idx_b].x,
                                               zoid.x_stencil_md[0][idx_b].y,
                                               zoid.x_stencil_md[0][idx_b].z};

                              for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                                  double lo = zoid.zoid.cuts[dim].lower + t2 * zoid.zoid.cuts[dim].slope_lower;
                                  double hi = zoid.zoid.cuts[dim].upper + t2 * zoid.zoid.cuts[dim].slope_upper;
                                  if (!(pos[dim] >= lo && pos[dim] < hi)) {
                                      in_zoid = false;
                                  }
                              }

                              if (in_zoid) {
                                  timesteps_local_b.insert(t2);
                              }
                          }

                          if (timesteps_local_a != timesteps_local_b) {
                              return timesteps_local_a > timesteps_local_b;
                          }
                      }
                      */

                      // USE LAMMPS SORTING
                      const auto& pos_a = zoid.x_stencil_md[0][idx_a];
                      int ix_a = static_cast<int> ((pos_a.x - domain->boxlo[0]) * bininvx);
                      int iy_a = static_cast<int> ((pos_a.y - domain->boxlo[1]) * bininvy);
                      int iz_a = static_cast<int> ((pos_a.z - domain->boxlo[2]) * bininvz);

                      ix_a = MAX(ix_a,0);
                      iy_a = MAX(iy_a,0);
                      iz_a = MAX(iz_a,0);
                      ix_a = MIN(ix_a,nbinx-1);
                      iy_a = MIN(iy_a,nbiny-1);
                      iz_a = MIN(iz_a,nbinz-1);
                      int ibin_a = iz_a*nbiny*nbinx + iy_a*nbinx + ix_a;

                      const auto& pos_b = zoid.x_stencil_md[0][idx_b];
                      int ix_b = static_cast<int> ((pos_b.x - domain->boxlo[0]) * bininvx);
                      int iy_b = static_cast<int> ((pos_b.y - domain->boxlo[1]) * bininvy);
                      int iz_b = static_cast<int> ((pos_b.z - domain->boxlo[2]) * bininvz);

                      ix_b = MAX(ix_b,0);
                      iy_b = MAX(iy_b,0);
                      iz_b = MAX(iz_b,0);
                      ix_b = MIN(ix_b,nbinx-1);
                      iy_b = MIN(iy_b,nbiny-1);
                      iz_b = MIN(iz_b,nbinz-1);
                      int ibin_b = iz_b*nbiny*nbinx + iy_b*nbinx + ix_b;

                      return ibin_a < ibin_b;
                });

                for (int t = 0; t < DOUBLE_BUFFERING; t++) {
                    apply_permutation_in_place(zoid.x_stencil_md[t], permutation);
                    // apply_permutation_in_place(zoid.v_stencil_md[t], permutation);
                }

                apply_permutation_in_place(zoid.tag_stencil_md[0], permutation);
                apply_permutation_in_place(zoid.type_stencil_md[0], permutation);
                apply_permutation_in_place(zoid.image_stencil_md[0], permutation);
                apply_permutation_in_place(zoid.mask_stencil_md[0], permutation);
                apply_permutation_in_place(zoid.v_stencil_md[0], permutation);

                if (DOUBLE_BUFFERING == 2) {
                    assert(zoid.x_stencil_md[0].size() == zoid.x_stencil_md[1].size());
                    // assert(zoid.v_stencil_md[0].size() == zoid.v_stencil_md[1].size());

                    for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                        assert(fabs(zoid.x_stencil_md[0][i].x - zoid.x_stencil_md[1][i].x) < 1e-6);
                        assert(fabs(zoid.x_stencil_md[0][i].y - zoid.x_stencil_md[1][i].y) < 1e-6);
                        assert(fabs(zoid.x_stencil_md[0][i].z - zoid.x_stencil_md[1][i].z) < 1e-6);
                    }
                }
            }
        }
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& send_to = lmp->send_to_neighbors[i];
        int num_in_different_proc = 0;
        for (auto& send_zoid_num : send_to) {
            if (true || get_zoid_dep(send_zoid_num) == get_zoid_dep(i) + 1) {
                if (send_zoid_num % comm->nprocs != i % comm->nprocs) {
                    num_in_different_proc++;
                }
            }
        }

        if (comm->me == 0) {
            std::cout << "zoid: " << i << " num send to different proc: " << num_in_different_proc << std::endl;
        }
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& send_to = lmp->send_to_neighbors_next_dt[i];
        int num_in_different_proc = 0;
        for (auto& send_zoid_num : send_to) {
            if (true || get_zoid_dep_next_dt(send_zoid_num) == get_zoid_dep_next_dt(i) + 1) {
                if (send_zoid_num % comm->nprocs != i % comm->nprocs) {
                    num_in_different_proc++;
                }
            }
        }

        if (comm->me == 0) {
            std::cout << "NEXT DT zoid: " << i << " num send to different proc: " << num_in_different_proc << std::endl;
        }
    }

    MPI_Barrier(world);
    // assert(false);
}

void StencilMD::CREATE_ATOM_IDXS_DOUBLE_BUFFERING() {
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                std::map<tagint, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    std::vector<int> tags;
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    for (int i = 0; i < atom_->nlocal; i++) {
                        tags.push_back(atom_->tag[i]);
                    }

                    std::sort(tags.begin(), tags.end(), [&](auto& tag_a, auto& tag_b) {
                        return tag_to_idx[tag_a] < tag_to_idx[tag_b];
                    });

                    std::vector<int> real_idxs;
                    for (int i = 0; i < tags.size(); i++) {
                        assert(std::find(real_idxs.begin(), real_idxs.end(), tag_to_idx[tags[i]]) == real_idxs.end());
                        real_idxs.push_back(tag_to_idx[tags[i]]);
                    }

                    zoid.local_idxs_per_timestep[t] = real_idxs;
                    assert(zoid.local_idxs_per_timestep[t].size() == atom_->nlocal);

                    std::vector<int> tmp_segments_idxs;
                    std::vector<int> tmp_segments_sizes;
                    int num_segments = get_segments(zoid.local_idxs_per_timestep[t], tmp_segments_idxs, tmp_segments_sizes);
                    std::cout << BOLDYELLOW << "zoid: " << zoid.num << " total size: " << tag_to_idx.size()
                              << " time: " << t << " num segments: " << num_segments << RESET_COLOR << std::endl;
                    /*
                    for (int k2 = 0; k2 < tmp_segments_idxs.size(); k2++) {
                        std::cout << "segment: " << k2 << " idx: " << tmp_segments_idxs[k2] << " size: " << tmp_segments_sizes[k2] << std::endl;
                    }
                    */
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                std::map<tagint, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    std::vector<int> tags;
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    for (int i = 0; i < atom_->nlocal; i++) {
                        tags.push_back(atom_->tag[i]);
                    }

                    std::sort(tags.begin(), tags.end(), [&](auto& tag_a, auto& tag_b) {
                        return tag_to_idx[tag_a] < tag_to_idx[tag_b];
                    });

                    std::vector<int> real_idxs;
                    for (int i = 0; i < tags.size(); i++) {
                        assert(std::find(real_idxs.begin(), real_idxs.end(), tag_to_idx[tags[i]]) == real_idxs.end());
                        real_idxs.push_back(tag_to_idx[tags[i]]);
                    }

                    zoid.local_idxs_per_timestep[t] = real_idxs;
                    assert(zoid.local_idxs_per_timestep[t].size() == atom_->nlocal);

                    std::vector<int> tmp_segments_idxs;
                    std::vector<int> tmp_segments_sizes;
                    int num_segments = get_segments(zoid.local_idxs_per_timestep[t], tmp_segments_idxs, tmp_segments_sizes);
                    std::cout << BOLDYELLOW << "NEXT DT zoid: " << zoid.num << " total size: " << tag_to_idx.size()
                              << " time: " << t << " num segments: " << num_segments << RESET_COLOR << std::endl;
                }
            }
        }
    }

    /*
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info &zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                if (zoid_num == 58) {
                    std::cout << "POINTER COMPARISON: " << zoid.local_idxs_per_timestep << " " << lmp->zoid_num_to_zoid_next_dt[zoid_num].local_idxs_per_timestep << std::endl;
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        std::cout << "POINTER COMPARISON: " << &(zoid.local_idxs_per_timestep[t]) << " " << &(lmp->zoid_num_to_zoid_next_dt[zoid_num].local_idxs_per_timestep[t]) << std::endl;
                    }
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        std::cout << "SIZE COMPARISON time: " << t << " " << zoid.local_idxs_per_timestep[t].size() << " " << lmp->zoid_num_to_zoid_next_dt[zoid_num].local_idxs_per_timestep[t].size() << std::endl;
                    }
                    assert(false);
                }
            }
        }
    }
    */
}

void StencilMD::CREATE_ATOM_IDX_MAPPING() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    auto& atom_arr = lmp->atom_stencil_md[zoid_num];
                    Atom* atom_ = atom_arr[t];
                    if (comm->nprocs == 1) {
                        std::vector<int> idxs;
                        for (int i = 0; i < atom_->nlocal; i++) {
                            int next_idx = zoid.atom_idx_mapping[t][i];
                            idxs.push_back(next_idx);
                        }

                        int num_segments = get_segments(idxs, atom_->atom_idx_mapping_segment_idxs, atom_->atom_idx_mapping_segment_sizes);
                    }
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    auto& atom_arr = lmp->atom_stencil_md[zoid_num];
                    Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                    if (comm->nprocs == 1) {
                        std::vector<int> idxs;
                        for (int i = 0; i < atom_->nlocal; i++) {
                            int next_idx = zoid.atom_idx_mapping[t][i];
                            idxs.push_back(next_idx);
                        }

                        int num_segments = get_segments(idxs, atom_->atom_idx_mapping_segment_idxs, atom_->atom_idx_mapping_segment_sizes);
                    }
                }
            }
        }
    }
}

void StencilMD::BUILD_NEIGHBOR_LIST() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Neighbor* neigh = lmp->neighbor_stencil_md[zoid_num][t];
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    Domain* domain_ = lmp->domain_stencil_md[zoid_num][t];
                    Comm* comm_ = lmp->comm_stencil_md[zoid_num];
                    neigh->setup_bins_stencil_md(atom_, domain_, comm_);
                    neigh->build_stencil_md(1, atom_, domain_,
                                            comm_, zoid);
                    neigh->ncalls = 0;

                    neigh->setup_stencil_md_bond_bins(atom_);

                    Force* force_ = lmp->force_stencil_md[zoid_num][t];
                    force_->setup();
                }
            }
        }
    }
}

void StencilMD::BUILD_NEIGHBOR_LIST_NEXT_DT() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_next_dt = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    Domain* domain_next_dt = lmp->domain_stencil_md_next_dt[zoid_num][t];
                    Neighbor* neigh_next_dt = lmp->neighbor_stencil_md_next_dt[zoid_num][t];
                    Comm* comm_ = lmp->comm_stencil_md[zoid_num];
                    neigh_next_dt->setup_bins_stencil_md(atom_next_dt,
                                                 domain_next_dt,
                                                 comm_);

                    neigh_next_dt->build_stencil_md(1, atom_next_dt, domain_next_dt,
                                                    comm_, zoid);
                    neigh_next_dt->ncalls = 0;

                    neigh_next_dt->setup_stencil_md_bond_bins(atom_next_dt);

                    Force* force_ = lmp->force_stencil_md_next_dt[zoid_num][t];
                    force_->setup();
                }
            }
        }
    }
}

void StencilMD::BUILD_NEIGHBOR_LIST_DOUBLE_BUFFERING() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Force* force_ = lmp->force_stencil_md[zoid_num][t];
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    Neighbor* neigh = lmp->neighbor_stencil_md[zoid_num][t];

                    const int * _noalias const ilist = force_->pair->list->ilist;
                    const int * _noalias const numneigh = force_->pair->list->numneigh;
                    const int * const * const firstneigh = force_->pair->list->firstneigh;

                    zoid.neighbor_list[t].resize(zoid.x_stencil_md[0].size());

                    std::map<tagint, int> tag_to_idx;
                    for (int i = 0; i < zoid.x_stencil_md[t % DOUBLE_BUFFERING].size(); i++) {
                        tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                    }

                    // cut in largest dimension
                    double widest_width = -1;
                    int widest_dim = -1;

                    for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                        double lo = zoid.zoid.cuts[dim].lower + 0 * zoid.zoid.cuts[dim].slope_lower;
                        double hi = zoid.zoid.cuts[dim].upper + 0 * zoid.zoid.cuts[dim].slope_upper;
                        double width = hi - lo;
                        if (width > widest_width) {
                            widest_dim = dim;
                            widest_width = width;
                        }
                    }

                    int dt = NUM_TIMESTEPS_IN_PARALLEL;
                    double xm = 0.5 * (zoid.zoid.cuts[widest_dim].lower + zoid.zoid.cuts[widest_dim].upper)
                                + 0.25 * dt * (zoid.zoid.cuts[widest_dim].slope_lower + zoid.zoid.cuts[widest_dim].slope_upper) + 0.5 * dt;
                    double boundary = xm - t * ALLEGRO_SLOPE;

                    for (int ii = 0; ii < atom_->nlocal; ii++) {
                        int i = ilist[ii];
                        int src_idx = tag_to_idx[atom_->tag[i]];

                        auto neigh_list = firstneigh[i];

                        for (int k = 0; k < numneigh[i]; k++) {
                            auto neigh_atom = neigh_list[k];
                            int dst_idx = tag_to_idx[atom_->tag[neigh_atom]];

                            /*
                            if (get_zoid_dep(zoid.num) == 0 || get_zoid_dep(zoid.num) == NUM_DEPS - 1) {
                                if (src_idx < dst_idx) {
                                    zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                                } else {
                                    zoid.neighbor_list[t][dst_idx].push_back(src_idx);
                                }
                            } else {
                                zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                            }
                            */

                            double pos_src = atom_->x[i][widest_dim];
                            double pos_dst = atom_->x[neigh_atom][widest_dim];

                            if (pos_dst < boundary && neigh_atom < atom_->nlocal) {
                                zoid.neighbor_list[t][dst_idx].push_back(src_idx);
                            } else {
                                zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                            }

                            // zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                        }
                    }

                    /*
                    int num_pairs = 0;
                    int num_diff = 0;

                    const auto& local_idxs = zoid.local_idxs_per_timestep[t];
                    for (int i = 0; i < local_idxs.size(); i += MODIFY_GRAINSIZE) {
                        std::set<int> idxs_accessed_in_chunk;
                        for (int k = i; k < i + MODIFY_GRAINSIZE && k < local_idxs.size(); k++) {
                            int idx = local_idxs[k];
                            idxs_accessed_in_chunk.insert(idx);
                        }
                        for (int k = i; k < i + MODIFY_GRAINSIZE && k < local_idxs.size(); k++) {
                            int idx = local_idxs[k];
                            num_pairs += zoid.neighbor_list[t][idx].size();
                            for (int neigh_idx : zoid.neighbor_list[t][idx]) {
                                if (idxs_accessed_in_chunk.find(neigh_idx) == idxs_accessed_in_chunk.end()) {
                                    num_diff++;
                                }
                            }
                        }
                    }

                    std::cout << BOLDCYAN << "PAY ATTENTION zoid: " << zoid.num << " timestep: " << t << " nlocal: " << local_idxs.size() << " total num pairs: " << num_pairs
                              << " num big diff: " << num_diff << RESET_COLOR << std::endl;
                    */
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Force* force_ = lmp->force_stencil_md_next_dt[zoid_num][t];
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    Neighbor* neigh = lmp->neighbor_stencil_md_next_dt[zoid_num][t];

                    const int * _noalias const ilist = force_->pair->list->ilist;
                    const int * _noalias const numneigh = force_->pair->list->numneigh;
                    const int * const * const firstneigh = force_->pair->list->firstneigh;

                    zoid.neighbor_list[t].resize(zoid.x_stencil_md[0].size());

                    std::map<tagint, int> tag_to_idx;
                    for (int i = 0; i < zoid.x_stencil_md[t % DOUBLE_BUFFERING].size(); i++) {
                        tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                    }

                    // cut in largest dimension
                    double widest_width = -1;
                    int widest_dim = -1;

                    for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                        double lo = zoid.zoid.cuts[dim].lower + 0 * zoid.zoid.cuts[dim].slope_lower;
                        double hi = zoid.zoid.cuts[dim].upper + 0 * zoid.zoid.cuts[dim].slope_upper;
                        double width = hi - lo;
                        if (width > widest_width) {
                            widest_dim = dim;
                            widest_width = width;
                        }
                    }

                    int dt = NUM_TIMESTEPS_IN_PARALLEL;
                    double xm = 0.5 * (zoid.zoid.cuts[widest_dim].lower + zoid.zoid.cuts[widest_dim].upper)
                                + 0.25 * dt * (zoid.zoid.cuts[widest_dim].slope_lower + zoid.zoid.cuts[widest_dim].slope_upper) + 0.5 * dt;
                    double boundary = xm - t * ALLEGRO_SLOPE;

                    for (int ii = 0; ii < atom_->nlocal; ii++) {
                        int i = ilist[ii];
                        auto neigh_list = firstneigh[i];
                        int src_idx = tag_to_idx[atom_->tag[i]];

                        for (int k = 0; k < numneigh[i]; k++) {
                            auto neigh_atom = neigh_list[k];
                            int dst_idx = tag_to_idx[atom_->tag[neigh_atom]];
                            assert(src_idx >= 0 && src_idx < zoid.x_stencil_md[0].size());

                            double pos_src = atom_->x[i][widest_dim];
                            double pos_dst = atom_->x[neigh_atom][widest_dim];

                            if (pos_dst < boundary && neigh_atom < atom_->nlocal) {
                                zoid.neighbor_list[t][dst_idx].push_back(src_idx);
                            } else {
                                zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                            }

                            /*
                            if (get_zoid_dep(zoid.num) == 0 || get_zoid_dep(zoid.num) == NUM_DEPS - 1) {
                                if (src_idx < dst_idx) {
                                    zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                                } else {
                                    zoid.neighbor_list[t][dst_idx].push_back(src_idx);
                                }
                            } else {
                                zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                            }
                            */
                            // zoid.neighbor_list[t][src_idx].push_back(dst_idx);
                        }
                    }
                }
            }
        }
    }
}

void StencilMD::BUILD_BOND_LIST_DOUBLE_BUFFERING() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Neighbor* neigh = lmp->neighbor_stencil_md[zoid_num][t];
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];

                    zoid.bond_list[t].resize(zoid.x_stencil_md[0].size());

                    std::map<tagint, int> tag_to_idx;
                    for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                        tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                    }

                    // cut in largest dimension
                    double widest_width = -1;
                    int widest_dim = -1;

                    for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                        double lo = zoid.zoid.cuts[dim].lower + 0 * zoid.zoid.cuts[dim].slope_lower;
                        double hi = zoid.zoid.cuts[dim].upper + 0 * zoid.zoid.cuts[dim].slope_upper;
                        double width = hi - lo;
                        if (width > widest_width) {
                            widest_dim = dim;
                            widest_width = width;
                        }
                    }

                    int dt = NUM_TIMESTEPS_IN_PARALLEL;
                    double xm = 0.5 * (zoid.zoid.cuts[widest_dim].lower + zoid.zoid.cuts[widest_dim].upper)
                                + 0.25 * dt * (zoid.zoid.cuts[widest_dim].slope_lower + zoid.zoid.cuts[widest_dim].slope_upper) + 0.5 * dt;
                    double boundary = xm - t * ALLEGRO_SLOPE;

                    for (int ii = 0; ii < atom_->nlocal; ii++) {
                        auto& lst_bonds = neigh->atom_bondlist[ii];
                        int src_idx = tag_to_idx[atom_->tag[ii]];

                        for (int k = 0; k < lst_bonds.size(); k++) {
                            auto bond_info = lst_bonds[k];
                            auto neigh_atom = bond_info.first;
                            auto bond_type = bond_info.second;

                            int dst_idx = tag_to_idx[atom_->tag[neigh_atom]];

                            /*
                            if (get_zoid_dep(zoid.num) == 0 || get_zoid_dep(zoid.num) == NUM_DEPS - 1) {
                                if (src_idx < dst_idx) {
                                    zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                                } else {
                                    zoid.bond_list[t][dst_idx].push_back({src_idx, bond_type});
                                }
                                if (neigh_atom >= atom_->nlocal && src_idx > dst_idx) {
                                    std::cout << "CURR DT. weird shit: " << zoid.num << " time: " << t << " nlocal: " << atom_->nlocal
                                              << " idx: " << ii << " " << neigh_atom
                                              << " real idx: " << src_idx << " " << dst_idx
                                              << " tag: " << atom_->tag[ii] << " " << atom_->tag[neigh_atom] << std::endl;
                                    assert(false);
                                }
                            } else {
                                zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                            }
                            */

                            double pos_src = atom_->x[ii][widest_dim];
                            double pos_dst = atom_->x[neigh_atom][widest_dim];

                            if (pos_dst < boundary && neigh_atom < atom_->nlocal) {
                                zoid.bond_list[t][dst_idx].push_back({src_idx, bond_type});
                            } else {
                                zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                            }

                            // zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                        }
                    }
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    Neighbor* neigh = lmp->neighbor_stencil_md_next_dt[zoid_num][t];

                    zoid.bond_list[t].resize(zoid.x_stencil_md[0].size());

                    std::map<tagint, int> tag_to_idx;
                    for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                        tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                    }

                    // cut in largest dimension
                    double widest_width = -1;
                    int widest_dim = -1;

                    for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                        double lo = zoid.zoid.cuts[dim].lower + 0 * zoid.zoid.cuts[dim].slope_lower;
                        double hi = zoid.zoid.cuts[dim].upper + 0 * zoid.zoid.cuts[dim].slope_upper;
                        double width = hi - lo;
                        if (width > widest_width) {
                            widest_dim = dim;
                            widest_width = width;
                        }
                    }

                    int dt = NUM_TIMESTEPS_IN_PARALLEL;
                    double xm = 0.5 * (zoid.zoid.cuts[widest_dim].lower + zoid.zoid.cuts[widest_dim].upper)
                                + 0.25 * dt * (zoid.zoid.cuts[widest_dim].slope_lower + zoid.zoid.cuts[widest_dim].slope_upper) + 0.5 * dt;
                    double boundary = xm - t * ALLEGRO_SLOPE;

                    for (int ii = 0; ii < atom_->nlocal; ii++) {
                        auto& lst_bonds = neigh->atom_bondlist[ii];
                        for (int k = 0; k < lst_bonds.size(); k++) {
                            auto bond_info = lst_bonds[k];
                            auto neigh_atom = bond_info.first;
                            auto bond_type = bond_info.second;

                            int src_idx = tag_to_idx[atom_->tag[ii]];
                            int dst_idx = tag_to_idx[atom_->tag[neigh_atom]];

                            /*
                            if (get_zoid_dep(zoid.num) == 0 || get_zoid_dep(zoid.num) == NUM_DEPS - 1) {
                                if (src_idx < dst_idx) {
                                    zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                                } else {
                                    zoid.bond_list[t][dst_idx].push_back({src_idx, bond_type});
                                }
                                if (neigh_atom >= atom_->nlocal && src_idx > dst_idx) {
                                    std::cout << "NEXT DT. weird shit: " << zoid.num << " time: " << t << " nlocal: " << atom_->nlocal
                                              << " idx: " << ii << " " << neigh_atom
                                              << " real idx: " << src_idx << " " << dst_idx
                                              << " tag: " << atom_->tag[ii] << " " << atom_->tag[neigh_atom] << std::endl;
                                    assert(false);
                                }
                            } else {
                                zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                            }
                            */
                            // zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                            double pos_src = atom_->x[ii][widest_dim];
                            double pos_dst = atom_->x[neigh_atom][widest_dim];

                            if (pos_dst < boundary && neigh_atom < atom_->nlocal) {
                                zoid.bond_list[t][dst_idx].push_back({src_idx, bond_type});
                            } else {
                                zoid.bond_list[t][src_idx].push_back({dst_idx, bond_type});
                            }
                        }
                    }
                }
            }
        }
    }
}

void StencilMD::INIT_DOUBLE_BUFFERING_SEND_RECV_DATA() {
    stencilMD->CONSTRUCT_SEND_FORCE_DOUBLE_BUFFERING<true>();
    stencilMD->CONSTRUCT_SEND_FORCE_DOUBLE_BUFFERING<false>();
    stencilMD->CONSTRUCT_RECV_FORCE_DOUBLE_BUFFERING<true>();
    stencilMD->CONSTRUCT_RECV_FORCE_DOUBLE_BUFFERING<false>();

    stencilMD->CONSTRUCT_SEND_POS_DOUBLE_BUFFERING<true>();
    stencilMD->CONSTRUCT_SEND_POS_DOUBLE_BUFFERING<false>();
    stencilMD->CONSTRUCT_RECV_POS_DOUBLE_BUFFERING<true>();
    stencilMD->CONSTRUCT_RECV_POS_DOUBLE_BUFFERING<false>();

    stencilMD->CONSTRUCT_SEND_VEL_DOUBLE_BUFFERING<true>();
    stencilMD->CONSTRUCT_SEND_VEL_DOUBLE_BUFFERING<false>();
    stencilMD->CONSTRUCT_RECV_VEL_DOUBLE_BUFFERING<true>();
    stencilMD->CONSTRUCT_RECV_VEL_DOUBLE_BUFFERING<false>();
}

// TODO: what to do with inum per timestep
void StencilMD::SET_CLAIMED_ATOMIC_BOOLS() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    int num_chunks = atom_->nlocal / MODIFY_GRAINSIZE + 1;
                    int chunk_size = MODIFY_GRAINSIZE;

                    atom_->claimed = new std::atomic<bool>[num_chunks];
                    atom_->claimed_int = new std::atomic<int>[num_chunks];
                    atom_->claimed_flag = new std::atomic_flag[num_chunks];
                    atom_->claimed_flag_struct = new Atom::ClaimedFlag[num_chunks];
                    atom_->spinlocks = new spinlock[atom_->nlocal + atom_->nghost];
                    atom_->num_chunks = num_chunks;
                    atom_->chunk_size = chunk_size;

                    for (int i = 0; i < num_chunks; i++) {
                        atom_->claimed[i] = false;
                        atom_->claimed_int[i] = 0;
                        atom_->claimed_flag[i].clear();
                        atom_->claimed_flag_struct[i].m.clear();
                    }
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                int num_chunks = zoid.x_stencil_md[0].size() / MODIFY_GRAINSIZE + 1;
                int chunk_size = MODIFY_GRAINSIZE;

                zoid.claimed_flags_stencil_md[0] = new std::atomic_flag[num_chunks];

                for (int i = 0; i < num_chunks; i++) {
                    zoid.claimed_flags_stencil_md[0]->clear();
                }
            }
        }
    }

}

// TODO: what to do with inum per timestep
void StencilMD::SET_INUM_PER_TIMESTEP() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    Neighbor* neighbor_ = lmp->neighbor_stencil_md[zoid_num][t];

                    int nlocal = atom_->nlocal;
                    int inum = neighbor_->nbondlist;

                    auto max_inum = std::max({nlocal, inum});

                    int nthreads_to_use = max_inum / NUM_WORKERS_PER_THREAD;
                    if (nthreads_to_use < 1) {
                        nthreads_to_use = 1;
                    }

                    if (nthreads_to_use > comm->nthreads) {
                        nthreads_to_use = comm->nthreads;
                    }
                    zoid.inum_per_timestep[t] = nthreads_to_use;
                }
            }
        }
    }
}

void StencilMD::SET_INUM_PER_TIMESTEP_NEXT_DT() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    Neighbor* neighbor_ = lmp->neighbor_stencil_md[zoid_num][t];

                    int nlocal = atom_->nlocal;
                    int inum = neighbor_->nbondlist;

                    auto max_inum = std::max({nlocal, inum});

                    int nthreads_to_use = max_inum / NUM_WORKERS_PER_THREAD;
                    if (nthreads_to_use < 1) {
                        nthreads_to_use = 1;
                    }

                    if (nthreads_to_use > comm->nthreads) {
                        nthreads_to_use = comm->nthreads;
                    }
                    zoid.inum_per_timestep[t] = nthreads_to_use;
                }
            }
        }
    }
}

void StencilMD::COMPUTE_NUM_SEND_RECV_PROCESS() {
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            auto &send_to_neighbors = lmp->send_to_neighbors[zoid_num];

            for (int proc = 0; proc < comm->nprocs; proc++) {
                int nsend_force = 0;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_force_segments = zoid.send_force_num_segments[t][i];
                            for (int j = 0; j < num_force_segments; j++) {
                                nsend_force += zoid.send_force_sizes[t][i][j];
                            }
                        }
                    }
                }

                int nsend_vel = 0;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_vel_segments = zoid.send_pos_num_segments[t][i];
                            for (int j = 0; j < num_vel_segments; j++) {
                                nsend_vel += zoid.send_pos_sizes[t][i][j];
                            }
                        }
                    }
                }

                // count positions
                int nsend_pos = 0;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    nsend_pos += zoid.num_elems_send_process[t][proc];
                }

                int num_elems_send;
                if (DEBUG_SEND_RECV_DATA) {
                    num_elems_send = nsend_force * (3 + 1) + nsend_pos * (3 + 1) + nsend_vel * (3 + 1);
                } else {
                    num_elems_send = nsend_force * (3) + nsend_pos * (3) + nsend_vel * (3);
                }
                zoid.num_send_process[proc] = num_elems_send;
            }


            // compute number sent to each proc at each timestep to help parallelize across timesteps
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    int nsend_force = 0;
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_force_segments = zoid.send_force_num_segments[t][i];
                            for (int j = 0; j < num_force_segments; j++) {
                                nsend_force += zoid.send_force_sizes[t][i][j];
                            }
                        }
                    }

                    int nsend_vel = 0;
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_vel_segments = zoid.send_pos_num_segments[t][i];
                            for (int j = 0; j < num_vel_segments; j++) {
                                nsend_vel += zoid.send_pos_sizes[t][i][j];
                            }
                        }
                    }

                    // count positions
                    int nsend_pos = 0;
                    nsend_pos += zoid.num_elems_send_process[t][proc];

                    int num_elems_send_timestep;
                    if (DEBUG_SEND_RECV_DATA) {
                        num_elems_send_timestep = nsend_force * (3 + 1) + nsend_pos * (3 + 1) + nsend_vel * (3 + 1);
                    } else {
                        num_elems_send_timestep = nsend_force * (3) + nsend_pos * (3) + nsend_vel * (3);
                    }

                    zoid.num_send_process_timestep[proc][t] = num_elems_send_timestep;
                }
            }
        }
    }

    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            queue_info& zoid = lmp->zoid_num_to_zoid_next_dt[zoid_num];
            auto &send_to_neighbors = lmp->send_to_neighbors_next_dt[zoid_num];

            for (int proc = 0; proc < comm->nprocs; proc++) {
                int nsend_force = 0;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_force_segments = zoid.send_force_num_segments[t][i];
                            for (int j = 0; j < num_force_segments; j++) {
                                nsend_force += zoid.send_force_sizes[t][i][j];
                            }
                        }
                    }
                }

                int nsend_vel = 0;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_vel_segments = zoid.send_pos_num_segments[t][i];
                            for (int j = 0; j < num_vel_segments; j++) {
                                nsend_vel += zoid.send_pos_sizes[t][i][j];
                            }
                        }
                    }
                }

                // count positions
                int nsend_pos = 0;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    nsend_pos += zoid.num_elems_send_process[t][proc];
                }

                int num_elems_send;
                if (DEBUG_SEND_RECV_DATA) {
                    num_elems_send = nsend_force * (3 + 1) + nsend_pos * (3 + 1) + nsend_vel * (3 + 1);
                } else {
                    num_elems_send = nsend_force * (3) + nsend_pos * (3) + nsend_vel * (3);
                }
                zoid.num_send_process[proc] = num_elems_send;
            }

            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    int nsend_force = 0;
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_force_segments = zoid.send_force_num_segments[t][i];
                            for (int j = 0; j < num_force_segments; j++) {
                                nsend_force += zoid.send_force_sizes[t][i][j];
                            }
                        }
                    }

                    int nsend_vel = 0;
                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int neighbor = send_to_neighbors[i];
                        if (neighbor % comm->nprocs == proc) {
                            int num_vel_segments = zoid.send_pos_num_segments[t][i];
                            for (int j = 0; j < num_vel_segments; j++) {
                                nsend_vel += zoid.send_pos_sizes[t][i][j];
                            }
                        }
                    }

                    // count positions
                    int nsend_pos = 0;
                    nsend_pos += zoid.num_elems_send_process[t][proc];

                    int num_elems_send_timestep;
                    if (DEBUG_SEND_RECV_DATA) {
                        num_elems_send_timestep = nsend_force * (3 + 1) + nsend_pos * (3 + 1) + nsend_vel * (3 + 1);
                    } else {
                        num_elems_send_timestep = nsend_force * (3) + nsend_pos * (3) + nsend_vel * (3);
                    }

                    zoid.num_send_process_timestep[proc][t] = num_elems_send_timestep;
                }
            }
        }
    }
}

void StencilMD::CONSTRUCT_START_END_BIG_ZOIDS_HELPER_SHRINKING(queue_info& zoid) {
    constexpr int NUM_STAGES = 2;

    int num_timesteps_to_eval = (zoid.zoid.cuts[0].upper - zoid.zoid.cuts[0].lower) / (2 * fabs(zoid.zoid.cuts[0].slope_lower));

    auto& local_idxs = zoid.local_idxs_per_timestep[0];
    std::unordered_map<int, std::set<int>> local_idx_to_timesteps_local;

    for (auto& idx : local_idxs) {
        for (int t2 = 0; t2 < num_timesteps_to_eval; t2++) {
            bool in_zoid = true;
            double pos[3] = {zoid.x_stencil_md[0][idx].x,
                             zoid.x_stencil_md[0][idx].y,
                             zoid.x_stencil_md[0][idx].z};

            for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                double lo = zoid.zoid.cuts[dim].lower + t2 * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + t2 * zoid.zoid.cuts[dim].slope_upper;
                if (!(pos[dim] >= lo && pos[dim] < hi)) {
                    in_zoid = false;
                }
            }

            if (in_zoid) {
                local_idx_to_timesteps_local[idx].insert(t2);
            }
        }
    }

    std::vector<int> mapping(num_timesteps_to_eval + 1, 0);

    for (auto& [idx, timesteps_local] : local_idx_to_timesteps_local) {
        mapping[timesteps_local.size()]++;
    }

    int total = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        total += zoid.local_idxs_per_timestep[t].size();
    }

    int timestep_to_start = 0;
    int sum_so_far = 0;
    for (int idx = 0; idx < num_timesteps_to_eval + 1; idx++) {
        sum_so_far += mapping[idx];
        if (sum_so_far >= zoid.local_idxs_per_timestep[0].size() / 2) {
            timestep_to_start = idx;
            break;
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        int new_timestep = t + timestep_to_start;

        std::vector<int> stage0_idxs;
        std::vector<int> stage1_idxs;

        for (auto& idx: zoid.local_idxs_per_timestep[t]) {
            if (local_idx_to_timesteps_local[idx].find(new_timestep) != local_idx_to_timesteps_local[idx].end()) {
                stage0_idxs.push_back(idx);
            } else {
                stage1_idxs.push_back(idx);
            }
        }

        zoid.space_cut_idxs[t] = new std::vector<int>[NUM_STAGES];
        zoid.space_cut_idxs[t][0] = stage0_idxs;
        zoid.space_cut_idxs[t][1] = stage1_idxs;
    }
}

void StencilMD::CONSTRUCT_START_END_BIG_ZOIDS_HELPER_EXPANDING(queue_info& zoid) {
    constexpr int NUM_STAGES = 2;

    int num_timesteps_to_eval = (zoid.zoid.cuts[0].upper - zoid.zoid.cuts[0].lower) / (2 * fabs(zoid.zoid.cuts[0].slope_lower));

    auto& initial_local_idxs = zoid.local_idxs_per_timestep[0];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        std::vector<int> stage0_idxs;
        std::vector<int> stage1_idxs;

        auto& curr_local_idxs = zoid.local_idxs_per_timestep[t];

        for (int j = 0; j < curr_local_idxs.size(); j++) {
            if (j < initial_local_idxs.size()) {
                stage0_idxs.push_back(curr_local_idxs[j]);
            } else {
                stage1_idxs.push_back(curr_local_idxs[j]);
            }
        }

        zoid.space_cut_idxs[t] = new std::vector<int>[NUM_STAGES];
        zoid.space_cut_idxs[t][0] = stage0_idxs;
        zoid.space_cut_idxs[t][1] = stage1_idxs;
    }
}

void StencilMD::CONSTRUCT_START_END_BIG_ZOIDS_HELPER(queue_info& zoid) {
    constexpr int NUM_STAGES = 2;

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        std::vector<int> stage0_idxs;
        std::vector<int> stage1_idxs;

        auto& curr_local_idxs = zoid.local_idxs_per_timestep[t];

        // cut in largest dimension
        double widest_width = -1;
        int widest_dim = -1;

        for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
            double lo = zoid.zoid.cuts[dim].lower + 0 * zoid.zoid.cuts[dim].slope_lower;
            double hi = zoid.zoid.cuts[dim].upper + 0 * zoid.zoid.cuts[dim].slope_upper;
            double width = hi - lo;
            assert(width > 0);
            if (width > widest_width) {
                widest_dim = dim;
                widest_width = width;
            }
        }

        int dt = NUM_TIMESTEPS_IN_PARALLEL;
        double xm = 0.5 * (zoid.zoid.cuts[widest_dim].lower + zoid.zoid.cuts[widest_dim].upper)
                    + 0.25 * dt * (zoid.zoid.cuts[widest_dim].slope_lower + zoid.zoid.cuts[widest_dim].slope_upper) + 0.5 * dt;
        double boundary = xm - t * ALLEGRO_SLOPE;

        for (int local_idx : curr_local_idxs) {
            // double pos = zoid.x_stencil_md[0][local_idx][widest_dim];
            double pos;
            if (widest_dim == 0) {
                pos = zoid.x_stencil_md[0][local_idx].x;
            } else if (widest_dim == 1) {
                pos = zoid.x_stencil_md[0][local_idx].y;
            } else if (widest_dim == 2) {
                pos = zoid.x_stencil_md[0][local_idx].z;
            } else {
                assert(false);
            }

            if (pos < boundary) {
                stage0_idxs.push_back(local_idx);
            } else {
                stage1_idxs.push_back(local_idx);
            }
        }

        zoid.space_cut_idxs[t] = new std::vector<int>[NUM_STAGES];
        zoid.space_cut_idxs[t][0] = stage0_idxs;
        zoid.space_cut_idxs[t][1] = stage1_idxs;

        std::vector<int> tmp1;
        std::vector<int> tmp2;
        std::vector<int> tmp3;
        std::vector<int> tmp4;

        int num_segments_stage0 = get_segments(stage0_idxs, tmp1, tmp2);
        int num_segments_stage1 = get_segments(stage1_idxs, tmp3, tmp4);
        std::cout << "zoid: " << zoid.num << " timestep: " << t << " space cut num segments: " << num_segments_stage0 << " " << num_segments_stage1 << std::endl;
    }
}

void StencilMD::CONSTRUCT_START_END_BIG_ZOIDS() {
    constexpr int NUM_STAGES = 2;

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            auto &zoid = lmp->queues[dep][j];
            if (zoid.num % comm->nprocs != comm->me) {
                continue;
            }

            CONSTRUCT_START_END_BIG_ZOIDS_HELPER(zoid);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            auto &zoid = lmp->queues_next_dt[dep][j];
            if (zoid.num % comm->nprocs != comm->me) {
                continue;
            }

            CONSTRUCT_START_END_BIG_ZOIDS_HELPER(zoid);
        }
    }

    /*
    // setup zoids with shrinking in every dimension
    for (int j = 0; j < lmp->queues[0].size(); j++) {
        auto &zoid = lmp->queues[0][j];
        if (zoid.num % comm->nprocs != comm->me) {
            continue;
        }

        CONSTRUCT_START_END_BIG_ZOIDS_HELPER_SHRINKING(zoid);
    }

    // setup zoids with shrinking in every dimension
    for (int j = 0; j < lmp->queues_next_dt[0].size(); j++) {
        auto &zoid = lmp->queues_next_dt[0][j];
        if (zoid.num % comm->nprocs != comm->me) {
            continue;
        }
        CONSTRUCT_START_END_BIG_ZOIDS_HELPER_SHRINKING(zoid);
    }

    // setup zoids with shrinking in every dimension
    for (int j = 0; j < lmp->queues[NUM_DEPS - 1].size(); j++) {
        auto &zoid = lmp->queues[NUM_DEPS - 1][j];
        if (zoid.num % comm->nprocs != comm->me) {
            continue;
        }

        CONSTRUCT_START_END_BIG_ZOIDS_HELPER_EXPANDING(zoid);
    }

    // setup zoids with shrinking in every dimension
    for (int j = 0; j < lmp->queues_next_dt[NUM_DEPS - 1].size(); j++) {
        auto &zoid = lmp->queues_next_dt[NUM_DEPS - 1][j];
        if (zoid.num % comm->nprocs != comm->me) {
            continue;
        }
        CONSTRUCT_START_END_BIG_ZOIDS_HELPER_EXPANDING(zoid);
    }
    */
}

void StencilMD::COMPARE_POS_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info &zoid,
                                           double **test_x) {
    int zoid_num = zoid.num;
    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
        int tag = atom_->tag[k];
        double *x_ = atom_->x[k];
        for (int dim = 0; dim < 3; dim++) {
            double val = x_[dim];
            if (val < 0) {
                val += domain->prd[dim];
            } else if (val >= domain->prd[dim]) {
                val -= domain->prd[dim];
            }

            double test_val = test_x[timestep][tag * 3 + dim];
            if (test_val < 0) {
                test_val += domain->prd[dim];
            } else if (test_val >= domain->prd[dim]) {
                test_val -= domain->prd[dim];
            }

            if (fabs(val - test_val) > 1e-6) {
                if (curr_dt) {
                    std::cout << "-------POS DIFF--------"
                              << std::endl;
                } else {
                    std::cout << "-------NEXT DT POS DIFF--------"
                              << std::endl;
                }
                std::cout << "idx: " << k
                          << " out of nlocal: " << atom_->nlocal << " and total: " << atom_->nlocal + atom_->nghost
                          << std::endl;
                std::cout
                        << "Dim: " << dim << " Zoid: " << zoid_num
                        << " timestep: " << timestep << " tag: " << tag
                        << " different. " << std::endl;
                std::cout << "What I have: " << x_[0] << " "
                          << x_[1] << " " << x_[2] << std::endl;
                std::cout << "What does LAMMPS have? "
                          << test_x[timestep][tag * 3 + 0] << " "
                          << test_x[timestep][tag * 3 + 1] << " "
                          << test_x[timestep][tag * 3 + 2]
                          << std::endl;
                std::cout
                        << "Diff: "
                        << fabs(x_[dim] - test_x[timestep][tag * 3 + dim])
                        << std::endl;
                std::cout << "pos: " << atom_->x[k][0] << " "
                          << atom_->x[k][1] << " "
                          << atom_->x[k][2] << std::endl;

                /*
                for (int tmp = 0; tmp < 3; tmp++) {
                    std::cout
                            << "lo: "
                            << zoid.zoid.cuts[tmp].lower +
                               zoid.zoid.cuts[tmp].slope_lower *
                               t
                            << std::endl;
                    std::cout
                            << "hi: "
                            << zoid.zoid.cuts[tmp].upper +
                               zoid.zoid.cuts[tmp].slope_upper *
                               t
                            << std::endl;
                }
                */
                assert(false);
            }
        }
    }
}

void StencilMD::COMPARE_FORCE_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f) {
    int zoid_num = zoid.num;

    for (int k = 0; k < atom_->nlocal; k++) {
        // compare forces on local atoms?
        int tag = atom_->tag[k];
        for (int dim = 0; dim < 3; dim++) {
            double my_force = atom_->f[k][dim] +
                              atom_->eval_f_stencil_md[k][dim];
            if (fabs(my_force - test_f[timestep][tag * 3 + dim]) >
                1e-6) {
                if (curr_dt) {
                    std::cout << "------FORCE DIFF--------"
                              << std::endl;
                } else {
                    std::cout << "------NEXT DT FORCE DIFF--------"
                              << std::endl;
                }
                std::cout << "idx: " << k
                          << " out of: " << atom_->nlocal << " atom: " << atom_
                          << std::endl;
                std::cout
                        << "Dim: " << dim << " Zoid: " << zoid_num
                        << " timestep: " << timestep << " tag: " << tag
                        << " different. " << std::endl;
                std::cout << "what I have f: " << atom_->f[k][0]
                          << " " << atom_->f[k][1] << " "
                          << atom_->f[k][2] << std::endl;
                std::cout
                        << "what I have eval "
                        << atom_->eval_f_stencil_md[k][0] << " "
                        << atom_->eval_f_stencil_md[k][1] << " "
                        << atom_->eval_f_stencil_md[k][2]
                        << std::endl;
                std::cout << "what I have: "
                          << atom_->f[k][0] +
                             atom_->eval_f_stencil_md[k][0]
                          << " "
                          << atom_->f[k][1] +
                             atom_->eval_f_stencil_md[k][1]
                          << " "
                          << atom_->f[k][2] +
                             atom_->eval_f_stencil_md[k][2]
                          << std::endl;
                std::cout << "What does LAMMPS have? "
                          << test_f[timestep][tag * 3 + 0] << " "
                          << test_f[timestep][tag * 3 + 1] << " "
                          << test_f[timestep][tag * 3 + 2]
                          << std::endl;
                /*
                std::cout << "What does LAMMPS have prev? "
                          << test_f[timestep - 1][tag * 3 + 0] << " "
                          << test_f[timestep - 1][tag * 3 + 1] << " "
                          << test_f[timestep - 1][tag * 3 + 2]
                          << std::endl;
                */
                std::cout
                        << "Diff: "
                        << fabs(my_force - test_f[timestep][tag * 3 + dim])
                        << std::endl;
                std::cout << "pos: " << atom_->x[k][0] << " "
                          << atom_->x[k][1] << " "
                          << atom_->x[k][2] << std::endl;

                for (int tmp = 0; tmp < 3; tmp++) {
                    std::cout
                            << "lo: "
                            << zoid.zoid.cuts[tmp].lower +
                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                            << std::endl;
                    std::cout
                            << "hi: "
                            << zoid.zoid.cuts[tmp].upper +
                               zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                            << std::endl;
                }

                assert(false);
            }
        }
    }
}

void StencilMD::COMPARE_VEL_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_v) {
    int zoid_num = zoid.num;

    for (int k = 0; k < atom_->nlocal; k++) {
        // compare forces on local atoms?
        int tag = atom_->tag[k];
        for (int dim = 0; dim < 3; dim++) {
            double my_vel = atom_->v[k][dim];
            if (fabs(my_vel - test_v[timestep][tag * 3 + dim]) >
                1e-6) {
                if (curr_dt) {
                    std::cout << "------VEL DIFF--------"
                              << std::endl;
                } else {
                    std::cout << "------NEXT DT VEL DIFF--------"
                              << std::endl;
                }
                std::cout << "idx: " << k
                          << " out of: " << atom_->nlocal << " atom: " << atom_
                          << std::endl;
                std::cout
                        << "Dim: " << dim << " Zoid: " << zoid_num
                        << " timestep: " << timestep << " tag: " << tag
                        << " different. " << std::endl;
                std::cout << "what I have v: " << atom_->v[k][0]
                          << " " << atom_->v[k][1] << " "
                          << atom_->v[k][2] << std::endl;
                std::cout << "What does LAMMPS have? "
                          << test_v[timestep][tag * 3 + 0] << " "
                          << test_v[timestep][tag * 3 + 1] << " "
                          << test_v[timestep][tag * 3 + 2]
                          << std::endl;
                std::cout
                        << "Diff: "
                        << fabs(my_vel - test_v[timestep][tag * 3 + dim])
                        << std::endl;
                std::cout << "pos: " << atom_->x[k][0] << " "
                          << atom_->x[k][1] << " "
                          << atom_->x[k][2] << std::endl;

                for (int tmp = 0; tmp < 3; tmp++) {
                    std::cout
                            << "lo: "
                            << zoid.zoid.cuts[tmp].lower +
                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                            << std::endl;
                    std::cout
                            << "hi: "
                            << zoid.zoid.cuts[tmp].upper +
                               zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                            << std::endl;
                }

                assert(false);
            }
        }
    }
}

std::vector<double>& StencilMD::GET_BOUNDS(bool curr_dt, int timestep) {
    auto& bounds_at_timestep = curr_dt ? bounds[timestep] : bounds[NUM_TIMESTEPS_IN_PARALLEL - timestep];

    if (bounds_at_timestep.size() > 0) {
        /*
        std::stringstream s;
        for (auto& b : bounds_at_timestep) {
            s << b << " ";
        }
        std::stringstream s2;
        for (int i = 1; i < bounds_at_timestep.size(); i++) {
            s2 << bounds_at_timestep[i] - bounds_at_timestep[i - 1] << " ";
        }
        std::cout << "num bounds: " << bounds_at_timestep.size() << std::endl;
        std::cout << "bounds: " << s.str() << std::endl;
        std::cout << "diffs: " << s2.str() << std::endl;
        assert(false);
        */
        return bounds_at_timestep;
    }

    std::vector<int> check_domains = {-1, 0, 1};
    std::vector<int> ts;

    // insert current bounds
    std::vector<double> try_bounds;

    for (int i = 0; i < NUM_ZOIDS; i++) {
        queue_info& zoid = curr_dt ? lmp->zoid_num_to_zoid[i] : lmp->zoid_num_to_zoid_next_dt[i];

        double lo_curr = zoid.zoid.cuts[0].lower + zoid.zoid.cuts[0].slope_lower * timestep;
        double hi_curr = zoid.zoid.cuts[0].upper + zoid.zoid.cuts[0].slope_upper * timestep;

        // std::vector<double> try_bounds = {lo_curr, lo_curr - ALLEGRO_SLOPE, lo_curr + ALLEGRO_SLOPE, hi_curr, hi_curr - ALLEGRO_SLOPE, hi_curr + ALLEGRO_SLOPE};
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        // for (int t = 0; t < 5; t++) {
            std::vector<int> try_different_vals = {-1, 0, 1};
            for (int try_val : try_different_vals) {
                double lo = zoid.zoid.cuts[0].lower + zoid.zoid.cuts[0].slope_lower * t + try_val * ALLEGRO_SLOPE;
                double hi = zoid.zoid.cuts[0].upper + zoid.zoid.cuts[0].slope_upper * t + try_val * ALLEGRO_SLOPE;
                try_bounds.push_back(lo);
                try_bounds.push_back(hi);
            }
        }
    }

    for (int i = 0; i < try_bounds.size(); i++) {
        if (try_bounds[i] < domain->boxlo[0]) {
            try_bounds[i] += domain->prd[0];
        }
        if (try_bounds[i] >= domain->boxhi[0]) {
            try_bounds[i] -= domain->prd[0];
        }
    }

    for (auto& try_val : try_bounds) {
        bool close_to_existing = false;
        for (auto& b : bounds_at_timestep) {
            // if (fabs(try_val - b) <= 1e-5) {
            if (fabs(try_val - b) < ALLEGRO_SLOPE - 1e-3) {
                close_to_existing = true;
            }
        }

        if (!close_to_existing) {
            bounds_at_timestep.push_back(try_val);
        }
    }

    std::sort(bounds_at_timestep.begin(), bounds_at_timestep.end());

    std::vector<double> points_coords;
    for (int i = 0; i < bounds_at_timestep.size() - 1; i++) {
        double lo = bounds_at_timestep[i];
        double hi = bounds_at_timestep[i + 1];
        assert(hi - lo > 1e-5);
        double avg = (lo + hi) / 2;
        points_coords.push_back(avg);
    }

    // last one
    double last_lo = bounds_at_timestep[bounds_at_timestep.size() - 1];
    double last_hi = bounds_at_timestep[0] + domain->prd[0];
    assert(last_hi - last_lo > 1e-5);
    double last_avg = (last_lo + last_hi) / 2;
    points_coords.push_back(last_avg);

    std::vector<Point> points(NUM_BINS * NUM_BINS * NUM_BINS);
    // points.reserve(points_coords.size() * points_coords.size() * points_coords.size());
    for (int bin_x = 0; bin_x < points_coords.size(); bin_x++) {
        for (int bin_y = 0; bin_y < points_coords.size(); bin_y++) {
            for (int bin_z = 0; bin_z < points_coords.size(); bin_z++) {
                IDX_3D bin = {bin_x, bin_y, bin_z};
                int bin_idx = get_bin_idx(bin);
                // points.emplace_back(points_coords[bin_x], points_coords[bin_y], points_coords[bin_z]);
                points[bin_idx] = Point(points_coords[bin_x], points_coords[bin_y], points_coords[bin_z]);
            }
        }
    }

    auto& indices = curr_dt ? sorted_bin_indices[timestep] : sorted_bin_indices[NUM_TIMESTEPS_IN_PARALLEL - timestep];
    assert(indices.size() == 0);
    indices.reserve(points.size());
    for (int i = 0; i < points.size(); i++) {
        indices.push_back(i);
    }

    /*
    CGAL::spatial_sort(indices.begin(),
                       indices.end(),
                       Search_traits(CGAL::make_property_map(points)));
    */

    // assert(false);
    return bounds_at_timestep;
}

std::vector<double>& StencilMD::LAMMPS_GET_BOUNDS(bool curr_dt, int timestep) {
    auto& bounds_at_timestep = curr_dt ? lammps_bounds[timestep] : lammps_bounds[NUM_TIMESTEPS_IN_PARALLEL - timestep];

    if (bounds_at_timestep.size() > 0) {
        return bounds_at_timestep;
    }

    int dim = 0;

    for (int i = 0; i < LAMMPS_NUM_REGIONS; i++) {
        double width = (domain->boxhi[dim] - domain->boxlo[dim]) / LAMMPS_NUM_REGIONS;
        bounds_at_timestep.push_back(domain->boxlo[dim] + width * i);
    }

    bounds_at_timestep.push_back(domain->boxhi[dim]);

    std::vector<double> points_coords;
    for (int i = 0; i < bounds_at_timestep.size() - 1; i++) {
        double lo = bounds_at_timestep[i];
        double hi = bounds_at_timestep[i + 1];
        assert(hi - lo > 1e-5);
        double avg = (lo + hi) / 2;
        points_coords.push_back(avg);
    }

    std::vector<Point> points(LAMMPS_NUM_REGIONS * LAMMPS_NUM_REGIONS * LAMMPS_NUM_REGIONS);
    for (int bin_x = 0; bin_x < points_coords.size(); bin_x++) {
        for (int bin_y = 0; bin_y < points_coords.size(); bin_y++) {
            for (int bin_z = 0; bin_z < points_coords.size(); bin_z++) {
                IDX_3D bin = {bin_x, bin_y, bin_z};
                int bin_idx = lammps_get_bin_idx(bin);
                points[bin_idx] = Point(points_coords[bin_x], points_coords[bin_y], points_coords[bin_z]);
            }
        }
    }

    auto& indices = curr_dt ? lammps_sorted_bin_indices[timestep] : lammps_sorted_bin_indices[NUM_TIMESTEPS_IN_PARALLEL - timestep];
    assert(indices.size() == 0);
    indices.reserve(points.size());
    for (int i = 0; i < points.size(); i++) {
        indices.push_back(i);
    }

    CGAL::spatial_sort(indices.begin(),
                       indices.end(),
                       Search_traits(CGAL::make_property_map(points)));

    // assert(false);
    return bounds_at_timestep;
}

