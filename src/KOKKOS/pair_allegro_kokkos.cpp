/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <cmath>
#include "kokkos.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "error.h"
#include "atom_masks.h"
#include "math_const.h"

#include <pair_allegro_kokkos.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <algorithm>
#include <random>

using namespace LAMMPS_NS;
using namespace MathConst;
namespace Kokkos {
  template <>
  struct reduction_identity<s_FEV_FLOAT> {
    KOKKOS_FORCEINLINE_FUNCTION static s_FEV_FLOAT sum() {
      return s_FEV_FLOAT();
    }
  };
}

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template<Precision precision>
PairAllegroKokkos<precision>::PairAllegroKokkos(LAMMPS *lmp) : PairAllegro<precision>(lmp)
{
  this->respa_enable = 0;


  this->atomKK = (AtomKokkos *) this->atom;
  this->execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  this->datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  this->datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<Precision precision>
PairAllegroKokkos<precision>::~PairAllegroKokkos()
{
  if (!this->copymode) {
    this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
    this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
    this->eatom = NULL;
    this->vatom = NULL;
  }
}

/* ---------------------------------------------------------------------- */

template<Precision precision>
void PairAllegroKokkos<precision>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) this->no_virial_fdotr_compute = 1;

  this->ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (this->eflag_atom) {
    this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
    this->memoryKK->create_kokkos(k_eatom,this->eatom,this->maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (this->vflag_atom) {
    this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
    this->memoryKK->create_kokkos(k_vatom,this->vatom,this->maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  this->atomKK->sync(this->execution_space,this->datamask_read);
  if (eflag || vflag) this->atomKK->modified(this->execution_space,this->datamask_modify);
  else this->atomKK->modified(this->execution_space,F_MASK);

  x = this->atomKK->k_x.template view<DeviceType>();
  f = this->atomKK->k_f.template view<DeviceType>();
  tag = this->atomKK->k_tag.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  nlocal = this->atom->nlocal;
  newton_pair = this->force->newton_pair;
  nall = this->atom->nlocal + this->atom->nghost;

  const int inum = this->list->inum;
  const int ignum = inum + this->list->gnum;
  std::cout << "inum: " << inum << " ignum: " << ignum << " nlocal: " << nlocal << std::endl;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(this->list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  this->copymode = 1;


  // build short neighbor list

  const int max_neighs = d_neighbors.extent(1);
  // TODO: check inum/ignum here
  const int n_atoms = neighflag == FULL ? inum : inum;



  if(d_numneigh_short.extent(0) < inum){
    d_numneigh_short = decltype(d_numneigh_short)();
    d_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::numneighs_short") ,inum);
    d_cumsum_numneigh_short = decltype(d_cumsum_numneigh_short)();
    d_cumsum_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::cumsum_numneighs_short") ,inum);
  }
  if(d_neighbors_short.extent(0) < inum || d_neighbors_short.extent(1) < max_neighs){
    d_neighbors_short = decltype(d_neighbors_short)();
    d_neighbors_short = Kokkos::View<int**,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::neighbors_short") ,inum,max_neighs);
  }

  // compute short neighbor list
  auto d_numneigh_short = this->d_numneigh_short;
  auto d_neighbors_short = this->d_neighbors_short;
  auto d_cumsum_numneigh_short = this->d_cumsum_numneigh_short;
  double cutoff = this->cutoff;
  std::cout << "CUTOFF: " << cutoff << std::endl;
  auto x = this->x;
  auto d_type = this->type;
  auto d_ilist = this->d_ilist;
  auto d_numneigh = this->d_numneigh;
  auto d_neighbors = this->d_neighbors;
  auto f = this->f;
  auto d_eatom = this->d_eatom;
  auto d_type_mapper = this->d_type_mapper;

    int target_atom_idx = -1;
    for (int k = 0; k < inum; k++) {
        if (tag(k) == 48139) {
            assert(target_atom_idx == -1);
            target_atom_idx = k;
        }
    }

  Kokkos::parallel_for("Allegro: Short neighlist", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii){
      const int i = d_ilist[ii];
      const X_FLOAT xtmp = x(i,0);
      const X_FLOAT ytmp = x(i,1);
      const X_FLOAT ztmp = x(i,2);

      assert(ii == i);

      const int si = d_type[i] - 1;

      const int jnum = d_numneigh[i];
      int inside = 0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;

        const X_FLOAT delx = xtmp - x(j,0);
        const X_FLOAT dely = ytmp - x(j,1);
        const X_FLOAT delz = ztmp - x(j,2);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < cutoff*cutoff) {
            d_neighbors_short(ii,inside) = j;
            inside++;
            /*
            if (target_atom_idx != -1) {
                if (tag(j) == 48139 || tag(i) == 48139) {
                    d_neighbors_short(ii,inside) = j;
                    inside++;
                }
            } else {
                d_neighbors_short(ii,inside) = j;
                inside++;
            }
            */
        }
      }
      d_numneigh_short(ii) = inside;
  });

  std::set<int> outgoing_edges;
  for (int ii = 0; ii < inum; ii++) {
      if (tag(ii) == 48139) {
          // std::cout << "regular md. " << "idx: " << i << " " << tag(i) << " has num neighbors: " << inside << " pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
          for (int k = 0; k < d_numneigh_short(ii); k++) {
              int neighbor = d_neighbors_short(ii,k);
              outgoing_edges.insert(neighbor);
              // std::cout << "regular md
              // Neighbor idx " << neighbor << "Neighbor tag: " << tag(neighbor) << " pos: " << x(neighbor, 0) << " " << x(neighbor, 1) << " " << x(neighbor, 2) << std::endl;
          }
      }
  }

  std::cout << "outgoing edges: " << outgoing_edges << std::endl;

  Kokkos::deep_copy(d_cumsum_numneigh_short, d_numneigh_short);

  Kokkos::parallel_scan("Allegro: cumsum shortneighs", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
      const int curr_val = d_cumsum_numneigh_short(ii);
      update += curr_val;
      if(is_final) d_cumsum_numneigh_short(ii) = update;
  });
  int nedges = 0;
  Kokkos::View<int*, Kokkos::HostSpace> nedges_view("Allegro: nedges",1);
  Kokkos::deep_copy(nedges_view, Kokkos::subview(d_cumsum_numneigh_short, Kokkos::make_pair(inum-1, inum)));
  nedges = nedges_view(0);

  auto nn = Kokkos::create_mirror_view(d_numneigh_short);
  Kokkos::deep_copy(nn, d_numneigh_short);
  auto cs = Kokkos::create_mirror_view(d_cumsum_numneigh_short);
  Kokkos::deep_copy(cs, d_cumsum_numneigh_short);
  //printf("INUM=%d, GNUM=%d, IGNUM=%d\n", inum, list->gnum, ignum);
  //printf("NEDGES: %d\nnumneigh_short cumsum\n",nedges);
  //for(int i = 0; i < inum; i++){
  //  printf("%d %d\n", nn(i), cs(i));
  //}


  if(d_edges.extent(1) < nedges){
    d_edges = decltype(d_edges)();
    d_edges = decltype(d_edges)("Allegro: edges", 2, nedges);
  }
  if(d_ij2type.extent(0) < ignum){
    d_ij2type = decltype(d_ij2type)();
    d_ij2type = decltype(d_ij2type)("Allegro: ij2type", ignum);
    d_xfloat = decltype(d_xfloat)();
    d_xfloat = decltype(d_xfloat)("Allegro: xfloat", ignum, 3);
  }

  auto d_edges = this->d_edges;
  auto d_ij2type = this->d_ij2type;
  auto d_xfloat = this->d_xfloat;

  Kokkos::parallel_for("Allegro: store type mask and x", Kokkos::RangePolicy<DeviceType>(0, ignum), KOKKOS_LAMBDA(const int i){
      d_ij2type(i) = d_type_mapper(d_type(i)-1);
      d_xfloat(i,0) = x(i,0);
      d_xfloat(i,1) = x(i,1);
      d_xfloat(i,2) = x(i,2);
  });

  Kokkos::parallel_for("Allegro: create edges", Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      const int startedge = ii==0 ? 0 : d_cumsum_numneigh_short(ii-1);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_short(ii)), [&] (const int jj){
          d_edges(0, startedge + jj) = i;
          d_edges(1, startedge + jj) = d_neighbors_short(ii,jj);
      });
  });

  torch::Tensor ij2type_tensor = torch::from_blob(d_ij2type.data(), {ignum}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
  torch::Tensor edges_tensor = torch::from_blob(d_edges.data(), {2,nedges}, {(long) d_edges.extent(1),1}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
  torch::Tensor pos_tensor = torch::from_blob(d_xfloat.data(), {ignum,3}, {3,1}, torch::TensorOptions().device(this->device).dtype(this->inputtorchtype));

  if (this->debug_mode) {
    /*
    printf("Allegro edges: i j rij\n");
    for (long i = 0; i < nedges; i++) {
      printf(
        "%ld %ld %.10g\n",
        edges_tensor.index({0, i}).item<long>(),
        edges_tensor.index({1, i}).item<long>(),
        (pos_tensor[edges_tensor.index({0, i}).item<long>()] - pos_tensor[edges_tensor.index({1, i}).item<long>()]).square().sum().sqrt().item<inputtype>()
      );
    }
    printf("end Allegro edges\n");
    */
  }

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor);
  input.insert("edge_index", edges_tensor);
  input.insert("atom_types", ij2type_tensor);
  std::vector<torch::IValue> input_vector(1, input);
  //std::cout << "NequIP model input:\n";
  //std::cout << "pos:\n" << pos_tensor.cpu() << "\n";
  //std::cout << "edge_index:\n" << edges_tensor.cpu() << "\n";
  //std::cout << "atom_types:\n" << ij2type_tensor.cpu() << "\n";

  auto output = this->model.forward(input_vector).toGenericDict();
  torch::Tensor forces_tensor = output.at("forces").toTensor();
  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();

  for (auto iter = output.begin(); iter != output.end(); iter++) {
    std::cout << "iter key: " << iter->key() << std::endl;
  }

  UnmanagedFloatView1D d_atomic_energy(atomic_energy_tensor.data_ptr<outputtype>(), inum);
  UnmanagedFloatView2D d_forces(forces_tensor.data_ptr<outputtype>(), ignum, 3);

  //std::cout << "NequIP model output:\n";
  //std::cout << "forces:\n" << forces_tensor.cpu() << "\n";
  //std::cout << "atomic_energy:\n" << atomic_energy_tensor.cpu() << "\n";

  this->eng_vdwl = 0.0;
  auto eflag_atom = this->eflag_atom;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  Kokkos::parallel_reduce("Allegro: store forces",
      Kokkos::RangePolicy<DeviceType>(0, ignum),
      KOKKOS_LAMBDA(const int i, double &eng_vdwl){
        f(i,0) = d_forces(i,0);
        f(i,1) = d_forces(i,1);
        f(i,2) = d_forces(i,2);
        if(eflag_atom && i < inum){
          d_eatom(i) = d_atomic_energy(i);
        }
        if(i < inum){
          eng_vdwl += d_atomic_energy(i);
        }
        if (std::isnan(f(i, 0))) {
          std::cout << "REGULAR MD. Me: " << world_rank << " FORCE IS NAN" << " for atom: " << i << std::endl;
          assert(false);
        }
        if (tag(i) == 48139) {
          std::cout << "me: " << world_rank << " test regular md Force for atom: " << tag(i) << " is: " << d_forces(i, 0) << " " << d_forces(i, 1) << " " << d_forces(i, 2) << " idx: " << i << " out of: " << inum << std::endl;
          std::cout << "Pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
          if (i < inum) {
            std::cout << "ATOMIC ENERGY: " << d_atomic_energy(i) << std::endl;
          }
        }
      },
      this->eng_vdwl
      );

  if (target_atom_idx != -1) {
      auto edge_features = output.at("edge_energy").toTensor();
      std::cout << "edge features size: " << edge_features.sizes() << std::endl;

      auto edges = edges_tensor.accessor<int64_t, 2>();
      auto pos = pos_tensor.accessor<inputtype, 2>();

      double energy = 0.0;
      for (int i = 0; i < nedges; i++) {
          int64_t src = edges[0][i];
          int64_t dst = edges[1][i];

          if (tag(src) == 48139 || tag(dst) == 48139) {
              std::cout << "REGULAR MD EDGE: " << src << " to: " << dst << " target atom idx: " << target_atom_idx << " tag src: " << tag(src) << " tag dst: " << tag(dst) << std::endl;
              std::cout << "EDGE NUMBER: " << i << std::endl;
              std::cout << "edge feature: " << edge_features.index({i}) << std::endl;
              // std::cout << "edge features: " << edge_features.index({src}) << std::endl;
              // std::cout << "edge features: " << edge_features.index({dst}) << std::endl;
              // std::cout << "Pos src: " << pos[src][0] << " " << pos[src][1] << " " << pos[src][2] << " Pos dst: " << pos[dst][0] << " " << pos[dst][1] << " " << pos[dst][2] << std::endl;
              // std::cout << "Type src: " << d_ij2type(i) << " Type dst: " << d_ij2type(d_neighbors_short(ii,jj)) << std::endl;
              double x_src = pos[src][0];
              double y_src = pos[src][1];
              double z_src = pos[src][2];
              double x_dst = pos[dst][0];
              double y_dst = pos[dst][1];
              double z_dst = pos[dst][2];
              double x_diff = x_src - x_dst;
              double y_diff = y_src - y_dst;
              double z_diff = z_src - z_dst;
              double dist = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
              // std::cout << "Dist: " << sqrt(dist) << std::endl;
          }
      }
      std::map<int, int> atom_to_incoming_count;
      std::map<int, int> atom_to_outgoing_count;
      for (int ii = 0; ii < nedges; ii++) {
          int src = d_edges(0, ii);
          int dst = d_edges(1, ii);
          if (!atom_to_incoming_count.count(tag(dst))) {
              atom_to_incoming_count[tag(dst)] = 0;
          }
          if (!atom_to_outgoing_count.count(tag(src))) {
              atom_to_outgoing_count[tag(src)] = 0;
          }
          atom_to_incoming_count[tag(dst)] = atom_to_incoming_count[tag(dst)] + 1;
          atom_to_outgoing_count[tag(src)] = atom_to_outgoing_count[tag(src)] + 1;
      }

      std::set<int> outgoing_edges;
      for (int ii = 0; ii < inum; ii++) {
          if (tag(ii) == 48139) {
              for (int k = 0; k < d_numneigh_short(ii); k++) {
                  int neighbor = d_neighbors_short(ii,k);
                  outgoing_edges.insert(tag(neighbor));
              }
          }
      }
      outgoing_edges.insert(48139);

      if (true) {
          for (auto& [k, v] : atom_to_incoming_count) {
              if (outgoing_edges.find(k) != outgoing_edges.end()) {
                  std::cout << "regular md tag: " << k << " num incoming edges: " << v << std::endl;
              }
          }
          for (auto& [k, v] : atom_to_outgoing_count) {
              if (outgoing_edges.find(k) != outgoing_edges.end()) {
                  std::cout << "regular md tag: " << k << " num outgoing edges: " << v << std::endl;
              }
          }
      }

      /*
      std::set<int> outgoing_edges_neighbor;
      for (int ii = 0; ii < inum; ii++) {
          if (tag(ii) == 48020) {
              for (int k = 0; k < d_numneigh_short(ii); k++) {
                  int neighbor = d_neighbors_short(ii,k);
                  outgoing_edges_neighbor.insert(neighbor);
              }
          }
      }
      std::cout << "outgoing edges neighbor size: " << outgoing_edges_neighbor.size() << std::endl;
      for (int edge_neighbor : outgoing_edges_neighbor) {
          std::cout << "Pos neighbor: " << edge_neighbor << x(edge_neighbor, 0) << " " << x(edge_neighbor, 1) << " " << x(edge_neighbor, 2) << " tag: " << tag(edge_neighbor) << std::endl;
      }
      */

      int idx = -1;
      std::vector<int> incoming_edges_neighbor;
      for (int ii = 0; ii < nedges; ii++) {
          if (tag(d_edges(1, ii)) == 48020) {
              incoming_edges_neighbor.push_back(d_edges(0, ii));
              idx = d_edges(1, ii);
          }
      }
      std::cout << "outgoing edges neighbor size: " << incoming_edges_neighbor.size() << std::endl;
      for (int edge_neighbor : incoming_edges_neighbor) {
          std::cout << "Pos neighbor: " << edge_neighbor << " " << x(edge_neighbor, 0) << " " << x(edge_neighbor, 1) << " " << x(edge_neighbor, 2) << " tag: " << tag(edge_neighbor) << std::endl;
      }

      if (idx != -1) {
          std::cout << "Pos: " << x(idx, 0) << " " << x(idx, 1) << " " << x(idx, 2) << std::endl;
      }

  }
  if (eflag_atom) {
    // if (need_dup)
    //   Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if(vflag){
    /*
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<outputtype, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    this->virial[0] = v[0][0][0];
    this->virial[1] = v[0][1][1];
    this->virial[2] = v[0][2][2];
    this->virial[3] = v[0][0][1];
    this->virial[4] = v[0][0][2];
    this->virial[5] = v[0][1][2];
    */
  }
  if(this->vflag_atom) {
    this->error->all(FLERR,"Pair style Allegro does not support per-atom virial");
  }

  if (this->vflag_fdotr) pair_virial_fdotr_compute(this);

  this->copymode = 0;

}

template<Precision precision>
void PairAllegroKokkos<precision>::compute_stencil_md(int eflag_in, int vflag_in, Atom* atom_, Atom* next, bool* mapping, queue_info& zoid, int timestep) {
    for (int i = 0; i < atom_->nlocal; i++) {
        assert(atom_->tag[i] == next->tag[mapping[i]]);
    }
    eflag = eflag_in;
    vflag = vflag_in;

    AtomKokkos* atomKK_ = (AtomKokkos*) atom_;
    if (neighflag == FULL) {
        this->no_virial_fdotr_compute = 1;
    }
    // TODO: stencil_md-ify potentially
    this->ev_init(eflag,vflag,0);

    // reallocate per-atom arrays if necessary

    if (this->eflag_atom) {
        this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
        this->memoryKK->create_kokkos(k_eatom,this->eatom,this->maxeatom,"pair:eatom");
        d_eatom = k_eatom.view<DeviceType>();
    }
    if (this->vflag_atom) {
        this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
        this->memoryKK->create_kokkos(k_vatom,this->vatom,this->maxvatom,"pair:vatom");
        d_vatom = k_vatom.view<DeviceType>();
    }

    // atomKK_->sync(this->execution_space,this->datamask_read);
    atomKK_->sync_stencil_md(this->execution_space,this->datamask_read, atom_);
    if (eflag || vflag) {
        // atomKK_->modified(this->execution_space,this->datamask_modify);
        atomKK_->modified_stencil_md(this->execution_space,this->datamask_modify, atom_);
    } else {
        atomKK_->modified_stencil_md(this->execution_space,F_MASK, atom_);
    }

    x = atomKK_->k_x.template view<DeviceType>();
    f = atomKK_->k_f.template view<DeviceType>();
    tag = atomKK_->k_tag.template view<DeviceType>();
    type = atomKK_->k_type.template view<DeviceType>();
    nlocal = atom_->nlocal;
    newton_pair = this->force->newton_pair;
    nall = atom_->nlocal + atom_->nghost;

    const int inum = this->list->inum;
    const int ignum = inum + this->list->gnum;
    NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(this->list);
    d_ilist = k_list->d_ilist;
    d_numneigh = k_list->d_numneigh;
    d_neighbors = k_list->d_neighbors;

    this->copymode = 1;

    assert(inum == atom_->nlocal);
    assert(ignum == atom_->nlocal + atom_->nghost);

    // build short neighbor list

    // max_neighs right now is around 60
    const int max_neighs = d_neighbors.extent(1);
    // TODO: check inum/ignum here
    const int n_atoms = neighflag == FULL ? inum : inum;
    // std::cout << "max neighs: " << max_neighs << std::endl;

    /*
    if(d_numneigh_short.extent(0) < inum){
        d_numneigh_short = decltype(d_numneigh_short)();
        d_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::numneighs_short") ,inum);
        d_cumsum_numneigh_short = decltype(d_cumsum_numneigh_short)();
        d_cumsum_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::cumsum_numneighs_short") ,inum);
    }
    if(d_neighbors_short.extent(0) < inum || d_neighbors_short.extent(1) < max_neighs){
        d_neighbors_short = decltype(d_neighbors_short)();
        d_neighbors_short = Kokkos::View<int**,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::neighbors_short") ,inum,max_neighs);
    }
    */

    if(d_numneigh_short.extent(0) < ignum){
        d_numneigh_short = decltype(d_numneigh_short)();
        d_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::numneighs_short") ,ignum);
        d_cumsum_numneigh_short = decltype(d_cumsum_numneigh_short)();
        d_cumsum_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::cumsum_numneighs_short") ,ignum);
    }
    if(d_neighbors_short.extent(0) < ignum || d_neighbors_short.extent(1) < max_neighs){
        d_neighbors_short = decltype(d_neighbors_short)();
        d_neighbors_short = Kokkos::View<int**,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::neighbors_short") ,ignum,max_neighs);
    }

    // compute short neighbor list
    auto d_numneigh_short = this->d_numneigh_short;
    auto d_neighbors_short = this->d_neighbors_short;
    auto d_cumsum_numneigh_short = this->d_cumsum_numneigh_short;
    double cutoff = this->cutoff;
    auto x = this->x;
    auto d_type = this->type;
    auto d_ilist = this->d_ilist;
    auto d_numneigh = this->d_numneigh;
    auto d_neighbors = this->d_neighbors;
    auto f = this->f;
    auto d_eatom = this->d_eatom;
    auto d_type_mapper = this->d_type_mapper;

    bool run_shit = false;
    int target_atom_idx = -1;
    for (int k = 0; k < atom_->nlocal; k++) {
        if (atom_->tag[k] == 48139) {
            run_shit = true;
            target_atom_idx = k;
        }
    }

    Kokkos::parallel_for("Allegro: Short neighlist", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii){
        const int i = d_ilist[ii];
        const X_FLOAT xtmp = x(i,0);
        const X_FLOAT ytmp = x(i,1);
        const X_FLOAT ztmp = x(i,2);

        const int si = d_type[i] - 1;

        const int jnum = d_numneigh[i];
        int inside = 0;
        for (int jj = 0; jj < jnum; jj++) {
            int j = d_neighbors(i,jj);
            j &= NEIGHMASK;

            const X_FLOAT delx = xtmp - x(j,0);
            const X_FLOAT dely = ytmp - x(j,1);
            const X_FLOAT delz = ztmp - x(j,2);
            const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

            if (rsq < cutoff*cutoff) {
                /*
                if (target_atom_idx != -1) {
                    if (tag(j) == 48139 || tag(i) == 48139) {
                        d_neighbors_short(ii,inside) = j;
                        inside++;
                    }
                } else {
                    d_neighbors_short(ii,inside) = j;
                    inside++;
                }
                */
                d_neighbors_short(ii,inside) = j;
                inside++;
            }

            assert(tag(i) != tag(j));
        }
        d_numneigh_short(ii) = inside;
    });

    std::map<int, std::vector<int>> idx_to_neighbors;
    int counter = 0;

    if (run_shit) {
        for (int ii = 0; ii < inum; ii++) {
            const int i = d_ilist[ii];
            assert(ii == d_ilist[ii]);

            if (mapping[i] >= next->nlocal && tag(i) == 48139) {
                // if (false) {
                int num_neigh = d_numneigh_short(ii);
                for (int k = 0; k < num_neigh; k++) {
                    int neigh = d_neighbors_short(ii, k);
                    if (neigh >= atom_->nlocal) {
                        idx_to_neighbors[neigh].push_back(i);
                    }
                }
            }
        }
    }

    for (int i = inum; i < ignum; i++) {
        int inside = 0;
        if (idx_to_neighbors.count(i)) {
            std::vector<int>& neighbors = idx_to_neighbors[i];
            for (int neighbor : neighbors) {
                if (true) {
                    std::cout << "Prev neighbor? " << d_neighbors_short(i, inside) << std::endl;
                    d_neighbors_short(i,inside) = neighbor;
                    inside++;
                    counter++;
                }

            }
        }
        d_numneigh_short(i) = inside;
    }

    Kokkos::deep_copy(d_cumsum_numneigh_short, d_numneigh_short);

    int nedges = 0;
    Kokkos::parallel_scan("Allegro: cumsum shortneighs", Kokkos::RangePolicy<DeviceType>(0,ignum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
        const int curr_val = d_cumsum_numneigh_short(ii);
        update += curr_val;
        if(is_final) {
            d_cumsum_numneigh_short(ii) = update;
        }
    });

    Kokkos::View<int*, Kokkos::HostSpace> nedges_view("Allegro: nedges",1);
    Kokkos::deep_copy(nedges_view, Kokkos::subview(d_cumsum_numneigh_short, Kokkos::make_pair(ignum-1, ignum)));
    nedges = nedges_view(0);

    auto nn = Kokkos::create_mirror_view(d_numneigh_short);
    Kokkos::deep_copy(nn, d_numneigh_short);
    auto cs = Kokkos::create_mirror_view(d_cumsum_numneigh_short);
    Kokkos::deep_copy(cs, d_cumsum_numneigh_short);
    // std::cout << "nedges: " << nedges << std::endl;
    /*
    printf("NEDGES: %d\nnumneigh_short cumsum\n",nedges);
    if (run_shit) {
        for(int i = 0; i < ignum; i++){
            printf("%d %d\n", nn(i), cs(i));
        }
    }
    */

    if(d_edges.extent(1) < nedges){
        d_edges = decltype(d_edges)();
        d_edges = decltype(d_edges)("Allegro: edges", 2, nedges);
    }
    if(d_ij2type.extent(0) < ignum){
        d_ij2type = decltype(d_ij2type)();
        d_ij2type = decltype(d_ij2type)("Allegro: ij2type", ignum);
        d_xfloat = decltype(d_xfloat)();
        d_xfloat = decltype(d_xfloat)("Allegro: xfloat", ignum, 3);
    }

    auto d_edges = this->d_edges;
    auto d_ij2type = this->d_ij2type;
    auto d_xfloat = this->d_xfloat;

    Kokkos::parallel_for("Allegro: store type mask and x", Kokkos::RangePolicy<DeviceType>(0, ignum), KOKKOS_LAMBDA(const int i){
        d_ij2type(i) = d_type_mapper(d_type(i)-1);
        assert(d_ij2type(i) == 0 || d_ij2type(i) == 1);
        d_xfloat(i,0) = x(i,0);
        d_xfloat(i,1) = x(i,1);
        d_xfloat(i,2) = x(i,2);
    });

    Kokkos::parallel_for("Allegro: create edges", Kokkos::TeamPolicy<DeviceType>(ignum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        assert(i == ii);
        const int startedge = ii==0 ? 0 : d_cumsum_numneigh_short(ii-1);
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_short(ii)), [&] (const int jj){
            d_edges(0, startedge + jj) = i;
            d_edges(1, startedge + jj) = d_neighbors_short(ii,jj);
        });
    });

    torch::Tensor ij2type_tensor = torch::from_blob(d_ij2type.data(), {ignum}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
    torch::Tensor edges_tensor = torch::from_blob(d_edges.data(), {2,nedges}, {(long) d_edges.extent(1),1}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
    torch::Tensor pos_tensor = torch::from_blob(d_xfloat.data(), {ignum,3}, {3,1}, torch::TensorOptions().device(this->device).dtype(this->inputtorchtype));

    /*
    if (run_shit) {
        int total = 0;
        for (auto& [k, v] : idx_to_neighbors) {
            total += v.size();
        }

        std::vector<int> idxs;
        for (auto& [k, v] : idx_to_neighbors) {
            idxs.push_back(k);
        }

        torch::Tensor a = torch::ones({2, total}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
        int idx = 0;

        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(idxs), std::end(idxs), rng);

        for (int i = 0; i < idxs.size(); i++) {
            int k = idxs[i];
            std::vector<int>& v = idx_to_neighbors[k];
            for (int neighbor : v) {
                a.index_put_({0, idx}, k);
                a.index_put_({1, idx}, neighbor);
                idx++;
            }
        }

        std::cout << "total: " << total << std::endl;

        edges_tensor = torch::cat({edges_tensor, a}, 1);

        // auto r = torch::randperm(edges_tensor.size(1));
        // std::cout << "r size: " << r.sizes() << std::endl;
        // edges_tensor = edges_tensor.index({"...", r});
        // std::cout << "perm edges tensor size: " << edges_tensor.sizes() << std::endl;
    }
    */








    if (this->debug_mode) {
        printf("Allegro edges: i j rij\n");
        for (int i = 0; i < nedges; i++) {
            /*
          printf(
            "%ld %ld %.10g\n",
            edges_tensor.index({0, i}).item<long>(),
            edges_tensor.index({1, i}).item<long>(),
            (pos_tensor[edges_tensor.index({0, i}).item<long>()] - pos_tensor[edges_tensor.index({1, i}).item<long>()]).square().sum().sqrt().item<inputtype>()
          );
             */
        }
        printf("end Allegro edges\n");
    }

    c10::Dict<std::string, torch::Tensor> input;
    input.insert("pos", pos_tensor);
    input.insert("edge_index", edges_tensor);
    input.insert("atom_types", ij2type_tensor);
    std::vector<torch::IValue> input_vector(1, input);

    /*
    std::cout << "NequIP model input:\n";
    std::cout << "pos:\n" << pos_tensor.cpu() << "\n";
    std::cout << "edge_index:\n" << edges_tensor.cpu() << "\n";
    std::cout << "atom_types:\n" << ij2type_tensor.cpu() << "\n";
    */


    auto output = this->model.forward(input_vector).toGenericDict();

    for (auto iter = output.begin(); iter != output.end(); iter++) {
        std::cout << "iter key: " << iter->key() << std::endl;
    }

    // auto total_energy = output.at("total_energy").toTensor();
    // std::cout << "Total energy: " << total_energy << std::endl;
    torch::Tensor forces_tensor = output.at("forces").toTensor();
    torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();

    UnmanagedFloatView1D d_atomic_energy(atomic_energy_tensor.data_ptr<outputtype>(), inum);
    UnmanagedFloatView2D d_forces(forces_tensor.data_ptr<outputtype>(), ignum, 3);

    std::cout << "Atomic energy tensor: " << atomic_energy_tensor.sizes() << " nlocal: " << atom_->nlocal << std::endl;


    UnmanagedFloatView1D d_atomic_energy2(atomic_energy_tensor.data_ptr<outputtype>(), ignum);

    //std::cout << "NequIP model output:\n";
    //std::cout << "forces:\n" << forces_tensor.cpu() << "\n";
    //std::cout << "atomic_energy:\n" << atomic_energy_tensor.cpu() << "\n";

    this->eng_vdwl = 0.0;
    auto eflag_atom = this->eflag_atom;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    Kokkos::parallel_reduce("Allegro: store forces",
                            Kokkos::RangePolicy<DeviceType>(0, ignum),
                            KOKKOS_LAMBDA(const int i, double &eng_vdwl){
                                if (tag(i) == 48139) {
                                    std::cout << "Pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
                                    std::cout << "me: " << " test stencil md Force for atom: " << tag(i) << " is: " << d_forces(i, 0) << " " << d_forces(i, 1) << " " << d_forces(i, 2) << " force rn: " << f(i, 0) << " " << f(i, 1) << " " << f(i, 2) << " idx: " << i << " out of: " << inum << std::endl;
                                    if (i < inum) {
                                        std::cout << "ATOMIC ENERGY: " << d_atomic_energy(i) << std::endl;
                                    }
                                }
                                f(i,0) = d_forces(i,0);
                                f(i,1) = d_forces(i,1);
                                f(i,2) = d_forces(i,2);
                                if(eflag_atom && i < inum){
                                    d_eatom(i) = d_atomic_energy(i);
                                }
                                if(i < inum) {
                                    eng_vdwl += d_atomic_energy(i);
                                }

                                if (std::isnan(f(i, 0))) {
                                    assert(false);
                                }
                            },
                            this->eng_vdwl
    );

    /*
    std::set<int> test_again;
    for (int k = 0; k < ignum; k++) {
        if (test_again.find(tag(k)) != test_again.end()) {
            std::cout << "RYAN REPEAT IDX: " << k << " REPEAT TAG: " << tag(k) << std::endl;
            for (int h = 0; h < k; h++) {
                if (tag(h) == tag(k)) {
                    std::cout << "Me: " << world_rank << " PREV INDEX IS: " << h << " pos: " << atom_->x[h][0] << " " << atom_->x[h][1] << " " << atom_->x[h][2] << std::endl;
                }
            }
            std::cout << "Pos: " << atom_->x[k][0] << " " << atom_->x[k][1] << " " << atom_->x[k][2] << std::endl;
            assert(false);
        }
        test_again.insert(tag(k));
    }
    */



    if (run_shit) {
        std::cout << "Nedges: " << nedges << " size: " << edges_tensor.sizes() << std::endl;
        auto edges = edges_tensor.accessor<int64_t, 2>();
        auto pos = pos_tensor.accessor<inputtype, 2>();

        auto edge_features = output.at("edge_energy").toTensor();
        std::cout << "Edge features size: " << edge_features.sizes() << std::endl;

        std::set<int> target_atom_neighbors;
        for (int i = 0; i < edges_tensor.size(1); i++) {
            int64_t src = edges[0][i];
            int64_t dst = edges[1][i];

            if (tag(src) == 48139 || tag(dst) == 48139) {
                target_atom_neighbors.insert(src);
                target_atom_neighbors.insert(dst);
                std::cout << "STENCIL MD EDGE: " << src << " to: " << dst << " target atom idx: " << target_atom_idx << " tag src: " << tag(src) << " tag dst: " << tag(dst) << std::endl;
                std::cout << "EDGE NUMBER: " << i << std::endl;
                std::cout << "edge feature: " << edge_features.index({i}) << std::endl;
                // std::cout << "edge features: " << edge_features.index({src}) << std::endl;
                // std::cout << "edge features: " << edge_features.index({dst}) << std::endl;
                // std::cout << "Pos src: " << pos[src][0] << " " << pos[src][1] << " " << pos[src][2] << " Pos dst: " << pos[dst][0] << " " << pos[dst][1] << " " << pos[dst][2] << std::endl;
                // std::cout << "Type src: " << d_ij2type(i) << " Type dst: " << d_ij2type(d_neighbors_short(ii,jj)) << std::endl;
                double x_src = pos[src][0];
                double y_src = pos[src][1];
                double z_src = pos[src][2];
                double x_dst = pos[dst][0];
                double y_dst = pos[dst][1];
                double z_dst = pos[dst][2];
                double x_diff = x_src - x_dst;
                double y_diff = y_src - y_dst;
                double z_diff = z_src - z_dst;
                double dist = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
                // std::cout << "Dist: " << sqrt(dist) << std::endl;
            }
        }

        if (false) {
            int idx = 6950;
            std::cout << "tag test? " << tag(idx) << std::endl;
            for (int i = 0; i < nedges; i++) {
                if (d_edges(0, i) == idx) {
                    std::cout << "Pos neighbor: " << d_edges(1, i) << " " << x(d_edges(1, i), 0) << " " << x(d_edges(1, i), 1) << " " << x(d_edges(1, i), 2) << " tag neighbor: " << tag(d_edges(1, i)) << std::endl;
                }
            }

            std::map<int, int> atom_to_incoming_count;
            std::map<int, int> atom_to_outgoing_count;
            for (int ii = 0; ii < nedges; ii++) {
                int src = d_edges(0, ii);
                int dst = d_edges(1, ii);
                if (!atom_to_incoming_count.count(tag(dst))) {
                    atom_to_incoming_count[tag(dst)] = 0;
                }
                if (!atom_to_outgoing_count.count(tag(src))) {
                    atom_to_outgoing_count[tag(src)] = 0;
                }
                atom_to_incoming_count[tag(dst)] = atom_to_incoming_count[tag(dst)] + 1;
                atom_to_outgoing_count[tag(src)] = atom_to_outgoing_count[tag(src)] + 1;
            }

            std::set<int> outgoing_edges;
            for (int ii = 0; ii < inum; ii++) {
                if (tag(ii) == 48139) {
                    for (int k = 0; k < d_numneigh_short(ii); k++) {
                        int neighbor = d_neighbors_short(ii,k);
                        outgoing_edges.insert(tag(neighbor));
                    }
                }
            }
            outgoing_edges.insert(48139);

            if (true) {
                for (auto& [k, v] : atom_to_incoming_count) {
                    if (outgoing_edges.find(k) != outgoing_edges.end()) {
                        std::cout << "stencil md tag: " << k << " num incoming edges: " << v << std::endl;
                    }
                }
                for (auto& [k, v] : atom_to_outgoing_count) {
                    if (outgoing_edges.find(k) != outgoing_edges.end()) {
                        std::cout << "stencil md tag: " << k << " num outgoing edges: " << v << std::endl;
                    }
                }
            }

            std::cout << "NUM EDGES: " << nedges << std::endl;
            /*
            std::set<int> outgoing_edges_neighbor;
            for (int ii = 0; ii < inum; ii++) {
                if (tag(ii) == 48020) {
                    for (int k = 0; k < d_numneigh_short(ii); k++) {
                        int neighbor = d_neighbors_short(ii,k);
                        outgoing_edges_neighbor.insert(neighbor);
                    }
                }
            }
            std::cout << "outgoing edges neighbor size: " << outgoing_edges_neighbor.size() << std::endl;
            for (int edge_neighbor : outgoing_edges_neighbor) {
                std::cout << "Pos neighbor: " << edge_neighbor << x(edge_neighbor, 0) << " " << x(edge_neighbor, 1) << " " << x(edge_neighbor, 2) << " tag: " << tag(edge_neighbor) << std::endl;
            }
            */

            std::vector<int> incoming_edges_neighbor;
            for (int ii = 0; ii < nedges; ii++) {
                if (tag(d_edges(1, ii)) == 48020) {
                    incoming_edges_neighbor.push_back(d_edges(0, ii));
                }
            }
            std::cout << "outgoing edges neighbor size: " << incoming_edges_neighbor.size() << std::endl;
            for (int edge_neighbor : incoming_edges_neighbor) {
                std::cout << "Pos neighbor: " << edge_neighbor << " " << x(edge_neighbor, 0) << " " << x(edge_neighbor, 1) << " " << x(edge_neighbor, 2) << " tag: " << tag(edge_neighbor) << std::endl;
            }

            // assert(false);
        }
    }

    if (eflag_atom) {
        // if (need_dup)
        //   Kokkos::Experimental::contribute(d_eatom, dup_eatom);
        k_eatom.template modify<DeviceType>();
        k_eatom.template sync<LMPHostType>();
    }

    if(vflag){
        /*
        torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
        auto v = v_tensor.accessor<outputtype, 3>();
        // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
        // First [0] index on v is batch
        this->virial[0] = v[0][0][0];
        this->virial[1] = v[0][1][1];
        this->virial[2] = v[0][2][2];
        this->virial[3] = v[0][0][1];
        this->virial[4] = v[0][0][2];
        this->virial[5] = v[0][1][2];
        */
    }
    if(this->vflag_atom) {
        this->error->all(FLERR,"Pair style Allegro does not support per-atom virial");
    }

    if (this->vflag_fdotr) pair_virial_fdotr_compute(this);

    this->copymode = 0;

}






/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<Precision precision>
void PairAllegroKokkos<precision>::coeff(int narg, char **arg)
{
  super::coeff(narg,arg);

  d_type_mapper = IntView1D("Allegro: type_mapper", this->type_mapper.size());
  auto h_type_mapper = Kokkos::create_mirror_view(d_type_mapper);
  for(int i = 0; i < this->type_mapper.size(); i++){
    h_type_mapper(i) = this->type_mapper[i];
  }
  Kokkos::deep_copy(d_type_mapper, h_type_mapper);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<Precision precision>
void PairAllegroKokkos<precision>::init_style()
{
  super::init_style();

  auto request = this->neighbor->find_request(this);

  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
      !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

  neighflag = this->lmp->kokkos->neighflag;
  if (neighflag != FULL) {
    this->error->all(FLERR,"Needs full neighbor list style with pair_allegro/kk");
  }
}

template<Precision precision>
void PairAllegroKokkos<precision>::init_style_stencil_md(Neighbor* neighbor_) {
    super::init_style_stencil_md(neighbor_);

    auto request = neighbor_->find_request(this);

    request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                             !std::is_same<DeviceType,LMPDeviceType>::value);
    request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

    neighflag = this->lmp->kokkos->neighflag;
    if (neighflag != FULL) {
        this->error->all(FLERR,"Needs full neighbor list style with pair_allegro/kk");
    }
}


namespace LAMMPS_NS {
template class PairAllegroKokkos<lowlow>;
template class PairAllegroKokkos<highhigh>;
template class PairAllegroKokkos<lowhigh>;
template class PairAllegroKokkos<highlow>;
}

