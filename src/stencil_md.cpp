//
// Created by Ryan Deng on 3/7/24.
//
#include <algorithm>

#include "stencil_md.h"
#include "comm.h"
#include "comm_brick.h"
#include "domain.h"
#include "accelerator_omp.h"
#include "neigh_list.h"
#include "modify.h"

using namespace LAMMPS_NS;

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
void StencilMD::FORCE_MODIFY_PARAMS(int narg, char **arg) {
    assert(false);
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
            lmp->force_stencil_md[i][j]->pair->modify_params(narg, arg);
            lmp->force_stencil_md_next_dt[i][j]->pair->modify_params(narg, arg);
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

void StencilMD::FORCE_PAIR_SETTINGS(int narg, char **arg) {
    for (int i = 0; i < lmp->force_stencil_md.size(); i++) {
        if (i % comm->nprocs == comm->me) {
            for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
                assert(lmp->force_stencil_md[i][j]->pair != NULL);
                lmp->force_stencil_md[i][j]->pair->settings(narg, arg);
                lmp->force_stencil_md_next_dt[i][j]->pair->settings(narg, arg);
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
            }
#ifdef LMP_OPENMP
                else {
            domain_ = new DomainOMP(lmp);
        }
#else
            else {
                domain_ = new Domain(lmp);
            }
#endif
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
            }
#ifdef LMP_OPENMP
                else {
            domain_ = new DomainOMP(lmp);
        }
#else
            else {
                domain_ = new Domain(lmp);
            }
#endif
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
            auto key =
                    std::make_tuple(zoid.where[0], zoid.where[1], zoid.where[2]);
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
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
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

                zoid.send_pos_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

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
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
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

                zoid.recv_process_segment_types =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
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
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_local_list =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_process_segment_sizes[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_idxs[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_types[t] = new int*[comm->nprocs];
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

                zoid.send_pos_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

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
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
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

                zoid.recv_process_segment_types =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
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
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_num_segments =
                        new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_local_list =
                        new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_process_segment_sizes[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_idxs[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_types[t] = new int*[comm->nprocs];
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
            lmp->zoid_num_to_zoid_next_dt[zoid_num] =
                    lmp->queues_next_dt[dep][j];
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

    lmp->send_to_neighbors_procs = new std::vector<int>[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;

            auto& send_to = lmp->send_to_neighbors[zoid_num];
            std::set<int> send_procs;
            for (int send_zoid : send_to) {
                send_procs.insert(send_zoid % comm->nprocs);
            }

            for (int proc : send_procs) {
                lmp->send_to_neighbors_procs[zoid_num].push_back(proc);

                if (proc == comm->me &&
                    std::find(lmp->recv_from_neighbors_procs.begin(),
                              lmp->recv_from_neighbors_procs.end(), zoid_num) ==
                    lmp->recv_from_neighbors_procs.end()) {
                    lmp->recv_from_neighbors_procs.push_back(zoid_num);
                }
            }
        }
    }

    lmp->send_to_neighbors_procs_next_dt = new std::vector<int>[NUM_ZOIDS];
    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& send_to = lmp->send_to_neighbors_next_dt[i];
        std::set<int> send_procs;
        for (int send_zoid : send_to) {
            send_procs.insert(send_zoid % comm->nprocs);
        }

        for (int proc : send_procs) {
            lmp->send_to_neighbors_procs_next_dt[i].push_back(proc);

            if (proc == comm->me &&
                std::find(lmp->recv_from_neighbors_procs_next_dt.begin(),
                          lmp->recv_from_neighbors_procs_next_dt.end(),
                          i) == lmp->recv_from_neighbors_procs_next_dt.end()) {
                lmp->recv_from_neighbors_procs_next_dt.push_back(i);
            }
        }
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
                modify_->init_stencil_md(lmp->atom_stencil_md[i][j]);
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
    // atom setup
#ifdef LMP_OPENMP
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Modify* modify_ = lmp->modify_stencil_md_omp[zoid_num][t];
                modify_->setup_stencil_md(vflag, lmp->atom_stencil_md[zoid_num][t]);
            }
        }
    }
#endif
}

void StencilMD::GET_LOCAL_ATOMS_ZOID() {
    // get local atoms for each zoid
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        comm->exchange_stencil_md_initial_send();
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
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
    }
}

void StencilMD::GET_GHOST_ATOMS_ZOID() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        comm->exchange_stencil_md_initial_send();
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* first = lmp->atom_stencil_md[zoid_num][t];
                    lmp->comm_stencil_md[zoid_num]
                            ->borders_stencil_md_initial_receive_from_lammps(
                                    first, lmp->domain_stencil_md[zoid_num][t], zoid,
                                    t);
                }
            }
        }

        MPI_Barrier(world);
    }
}

void StencilMD::BUILD_NEIGHBOR_LIST() {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->neighbor_stencil_md[zoid_num][t]
                            ->setup_bins_stencil_md(
                                    lmp->atom_stencil_md[zoid_num][t],
                                    lmp->domain_stencil_md[zoid_num][t],
                                    lmp->comm_stencil_md[zoid_num]);
                    lmp->neighbor_stencil_md[zoid_num][t]->build_stencil_md(
                            1, lmp->atom_stencil_md[zoid_num][t],
                            lmp->domain_stencil_md[zoid_num][t],
                            lmp->comm_stencil_md[zoid_num], zoid);
                    lmp->neighbor_stencil_md[zoid_num][t]->ncalls = 0;

                    MPI_Barrier(world);

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

                    lmp->neighbor_stencil_md_next_dt[zoid_num][t]
                            ->setup_bins_stencil_md(
                                    atom_next_dt,
                                    domain_next_dt,
                                    lmp->comm_stencil_md[zoid_num]);

                    lmp->neighbor_stencil_md_next_dt[zoid_num][t]->build_stencil_md(
                            1, atom_next_dt, domain_next_dt,
                            lmp->comm_stencil_md[zoid_num], zoid);
                    lmp->neighbor_stencil_md_next_dt[zoid_num][t]->ncalls = 0;

                    MPI_Barrier(world);

                    Force* force_ = lmp->force_stencil_md_next_dt[zoid_num][t];
                    force_->setup();
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
        }
    }
}
