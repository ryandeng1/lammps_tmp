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
  std::cout << "pair allegro kokkos compute" << std::endl;
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
  auto x = this->x;
  auto d_type = this->type;
  auto d_ilist = this->d_ilist;
  auto d_numneigh = this->d_numneigh;
  auto d_neighbors = this->d_neighbors;
  auto f = this->f;
  auto d_eatom = this->d_eatom;
  auto d_type_mapper = this->d_type_mapper;

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
          d_neighbors_short(ii,inside) = j;
          inside++;
        }
      }
      d_numneigh_short(ii) = inside;

      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      std::set<int> outgoing_edges;
      // if (i < inum && tag(i) == 3932) {
      if (i < inum && tag(i) == 10675) {
          // std::cout << "regular md. " << "idx: " << i << " " << tag(i) << " has num neighbors: " << inside << " pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
          for (int k = 0; k < inside; k++) {
              int neighbor = d_neighbors_short(ii,k);
              outgoing_edges.insert(neighbor);
              // std::cout << "regular md Neighbor idx " << neighbor << "Neighbor tag: " << tag(neighbor) << " pos: " << x(neighbor, 0) << " " << x(neighbor, 1) << " " << x(neighbor, 2) << std::endl;
          }
      }
      if (outgoing_edges.size() > 0) {
          std::cout << "regular md Rank: " << world_rank << " Num outgoing edges: " << outgoing_edges.size() << std::endl;
      }

      std::set<std::pair<int, int>> incoming_edges;
      for (int k = 0; k < inside; k++) {
          int neighbor = d_neighbors_short(ii,k);
          // if (tag(neighbor) == 3932) {
          if (tag(neighbor) == 10675) {
              incoming_edges.insert({i, tag(i)});
              // std::cout << "regular md. other edge. idx: " << i << " tag: " << tag(i) << " pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
              // std::cout << "Target idx: " << neighbor << " Target tag: " << tag(neighbor) << " Target pos: " << x(neighbor, 0) << " " << x(neighbor, 1) << " " << x(neighbor, 2) << std::endl;
          }
      }

      if (incoming_edges.size() > 0) {
          for (auto& p : incoming_edges) {
              int idx_ = p.first;
              int tag_ = p.second;
              // std::cout << "regular md rank: " << world_rank << " incoming edge: " << tag_ << " pos: " << x(idx_, 0) << " " << x(idx_, 1) << " " << x(idx_, 2) << std::endl;
          }
      }
  });
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
  // auto output = lmp->lmp_model.forward(input_vector).toGenericDict();
  torch::Tensor forces_tensor = output.at("forces").toTensor();
  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();

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
        if (i < inum && tag(i) == 10675) {
          std::cout << "regular md Force for atom: " << tag(i) << " is: " << d_forces(i, 0) << " " << d_forces(i, 1) << " " << d_forces(i, 2) << std::endl;
        }
      },
      this->eng_vdwl
      );

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
void PairAllegroKokkos<precision>::compute_stencil_md(int eflag_in, int vflag_in, Atom* atom_, bool tmp[3]) {
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
    auto x = this->x;
    auto d_type = this->type;
    auto d_ilist = this->d_ilist;
    auto d_numneigh = this->d_numneigh;
    auto d_neighbors = this->d_neighbors;
    auto f = this->f;
    auto d_eatom = this->d_eatom;
    auto d_type_mapper = this->d_type_mapper;

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

            if (rsq < cutoff*cutoff && tag(i) != tag(j)) {
                d_neighbors_short(ii,inside) = j;
                inside++;
            }
        }
        d_numneigh_short(ii) = inside;

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        std::set<int> outgoing_edges;
        // if (i < inum && tag(i) == 3932) {
        if (i < inum && tag(i) == 10675) {
            // std::cout << "stencil md. " << "idx: " << i << " " << tag(i) << " has num neighbors: " << inside << " pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
            for (int k = 0; k < inside; k++) {
                int neighbor = d_neighbors_short(ii,k);
                outgoing_edges.insert(neighbor);
                // std::cout << "stencil md Neighbor idx " << neighbor << " Neighbor tag: " << tag(neighbor) << " pos: " << x(neighbor, 0) << " " << x(neighbor, 1) << " " << x(neighbor, 2) << std::endl;
            }
        }

        if (outgoing_edges.size() > 0) {
            std::cout << "stencil md rank: " << world_rank << " num outgoing edges: " << outgoing_edges.size() << std::endl;
        }

        std::set<std::pair<int, int>> incoming_edges;
        for (int k = 0; k < inside; k++) {
            int neighbor = d_neighbors_short(ii,k);
            // if (tag(neighbor) == 3932) {
            if (tag(neighbor) == 10675) {
                incoming_edges.insert({i, tag(i)});
                // std::cout << "stencil md. other edge. idx: " << i << " tag: " << tag(i) << " pos: " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << std::endl;
                // std::cout << "Target idx: " << neighbor << " Target tag: " << tag(neighbor) << " Target pos: " << x(neighbor, 0) << " " << x(neighbor, 1) << " " << x(neighbor, 2) << std::endl;
            }
        }
        if (incoming_edges.size() > 0) {
            for (auto& p : incoming_edges) {
                int idx_ = p.first;
                int tag_ = p.second;
                // std::cout << "stencil md rank: " << world_rank << " incoming edge: " << tag_ << " pos: " << x(idx_, 0) << " " << x(idx_, 1) << " " << x(idx_, 2) << std::endl;
            }
        }
    });
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

    // TODO: somehow fix this to avoid NaNs
    Kokkos::parallel_for("Allegro: store type mask and x", Kokkos::RangePolicy<DeviceType>(0, ignum), KOKKOS_LAMBDA(const int i){
        d_ij2type(i) = d_type_mapper(d_type(i)-1);
        assert(d_ij2type(i) == 0 || d_ij2type(i) == 1);
        d_xfloat(i,0) = x(i,0);
        d_xfloat(i,1) = x(i,1);
        d_xfloat(i,2) = x(i,2);
        /*
        if (tmp[0]) {
            d_xfloat(i, 0) += 113.2955118;
        }
        if (tmp[1]) {
            d_xfloat(i, 1) += 113.2955118;
        }
        if (tmp[2]) {
            d_xfloat(i, 2) += 113.2955118;
        }
        */
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
    // auto output = lmp->lmp_model.forward(input_vector).toGenericDict();
    torch::Tensor forces_tensor = output.at("forces").toTensor();
    torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();

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
                                if(i < inum) {
                                    eng_vdwl += d_atomic_energy(i);
                                }
                                if (i < inum && tag(i) == 10675) {
                                    std::cout << "stencil md Force for atom: " << tag(i) << " is: " << d_forces(i, 0) << " " << d_forces(i, 1) << " " << d_forces(i, 2) << std::endl;
                                }
                                // TODO: fix
                                if (std::isnan(f(i, 0))) {
                                    std::cout << "STENCIL MD. Me: " << world_rank << " FORCE IS NAN" << " for atom num: " << i << " tag: " << atom_->tag[i] << " nlocal: " << atom_->nlocal << " pos: " << d_xfloat(i, 0) << " " << d_xfloat(i, 1) << " " << d_xfloat(i, 2) << std::endl;
                                    std::cout << "Force: " << d_forces(i, 0) << " " << d_forces(i, 1) << " " << d_forces(i, 2) << " Nedges: " << nedges << std::endl;
                                    // std::cout << "forces: " << forces_tensor << std::endl;
                                    // std::cout << "edge tensor: " << edges_tensor.size(0) << " " << edges_tensor.size(1) << std::endl;
                                    // std::cout << "ij2type tensor: " << ij2type_tensor.size(0) << " nlocal: " << atom_->nlocal << " " << inum << " total? " << ignum << std::endl;
                                    // std::cout << "pos tensor: " << pos_tensor << std::endl;
                                    /*
                                    for (int j = 0; j < nedges; j++) {
                                        int src = d_edges(0, j);
                                        int dst = d_edges(1, j);
                                        if (src == i || dst == i) {
                                            std::cout << "Src: " << src << " pos: " << x(src, 0) << " " << x(src, 1) << " " << x(src, 2) << " tag: " << atom_->tag[src] << std::endl;
                                            std::cout << "Dst: " << dst << " pos: " << x(dst, 0) << " " << x(dst, 1) << " " << x(dst, 2) << " tag: " << atom_->tag[dst] << std::endl;
                                        }
                                    }
                                    */
                                    // assert(false);
                                }
                            },
                            this->eng_vdwl
    );

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

