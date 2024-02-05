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

#include "atom_vec_atomic_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecAtomicKokkos::AtomVecAtomicKokkos(LAMMPS *lmp) : AtomVecKokkos(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  comm_x_only = comm_f_only = 1;
  size_forward = 3;
  size_reverse = 3;
  size_border = 6;
  size_velocity = 3;
  size_data_atom = 5;
  size_data_vel = 4;
  xcol_data = 3;

  k_count = DAT::tdual_int_1d("atom::k_count", 1);
  atomKK = (AtomKokkos *) atom;
  commKK = (CommKokkos *) comm;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by DELTA
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecAtomicKokkos::grow(int n)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA, nmax * 0.01);
  if (n == 0)
    nmax += step;
  else
    nmax = n;
  atomKK->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT) error->one(FLERR, "Per-processor system is too big");

  atomKK->sync(Device, ALL_MASK);
  atomKK->modified(Device, ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_tag, atomKK->tag, nmax, "atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type, atomKK->type, nmax, "atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask, atomKK->mask, nmax, "atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image, atomKK->image, nmax, "atom:image");

  memoryKK->grow_kokkos(atomKK->k_x, atomKK->x, nmax, "atom:x");
  memoryKK->grow_kokkos(atomKK->k_v, atomKK->v, nmax, "atom:v");
  memoryKK->grow_kokkos(atomKK->k_f, atomKK->f, nmax, "atom:f");

  grow_pointers();
  atomKK->sync(Host, ALL_MASK);

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

void AtomVecAtomicKokkos::grow_stencil_md(int n, Atom *atom_)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA, nmax * 0.01);
  if (n == 0) {
    nmax += step;
  } else {
    nmax = n;
  }

  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;

  atomKK_->nmax = nmax;
  if (atomKK_->nmax < 0 || atomKK_->nmax > MAXSMALLINT) {
    error->one(FLERR, "Per-processor system is too big");
  }

  atomKK_->sync_stencil_md(Device, ALL_MASK, atom_);
  atomKK_->modified_stencil_md(Device, ALL_MASK, atom_);

  memoryKK->grow_kokkos(atomKK_->k_tag, atomKK_->tag, nmax, "atom:tag");
  memoryKK->grow_kokkos(atomKK_->k_type, atomKK_->type, nmax, "atom:type");
  memoryKK->grow_kokkos(atomKK_->k_mask, atomKK_->mask, nmax, "atom:mask");
  memoryKK->grow_kokkos(atomKK_->k_image, atomKK_->image, nmax, "atom:image");

  memoryKK->grow_kokkos(atomKK_->k_x, atomKK_->x, nmax, "atom:x");
  memoryKK->grow_kokkos(atomKK_->k_v, atomKK_->v, nmax, "atom:v");
  memoryKK->grow_kokkos(atomKK_->k_f, atomKK_->f, nmax, "atom:f");

  memoryKK->grow_kokkos(atomKK_->k_eval_mask_stencil_md, atomKK_->eval_mask_stencil_md, nmax,
                        "atom:eval_mask_stencil_md");

  memoryKK->grow_kokkos(atomKK_->k_actually_eval_mask_stencil_md, atomKK_->actually_eval_mask_stencil_md, nmax,
                          "atom:actually_eval_mask_stencil_md");
  memoryKK->grow_kokkos(atomKK_->k_eval_f_stencil_md, atomKK_->eval_f_stencil_md, nmax,
                          "atom:k_eval_f_stencil_md");

  grow_pointers_stencil_md(atom_);
  atomKK_->sync_stencil_md(Host, ALL_MASK, atom_);

  if (atom_->nextra_grow) {
    assert(false);
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
  }
}
/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecAtomicKokkos::grow_pointers()
{
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.d_view;
  h_tag = atomKK->k_tag.h_view;

  type = atomKK->type;
  d_type = atomKK->k_type.d_view;
  h_type = atomKK->k_type.h_view;
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.d_view;
  h_mask = atomKK->k_mask.h_view;
  image = atomKK->image;
  d_image = atomKK->k_image.d_view;
  h_image = atomKK->k_image.h_view;

  x = atomKK->x;
  d_x = atomKK->k_x.d_view;
  h_x = atomKK->k_x.h_view;
  v = atomKK->v;
  d_v = atomKK->k_v.d_view;
  h_v = atomKK->k_v.h_view;
  f = atomKK->f;
  d_f = atomKK->k_f.d_view;
  h_f = atomKK->k_f.h_view;
}

void AtomVecAtomicKokkos::grow_pointers_stencil_md(Atom *atom_)
{
  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
  tag = atomKK_->tag;
  d_tag = atomKK_->k_tag.d_view;
  h_tag = atomKK_->k_tag.h_view;

  type = atomKK_->type;
  d_type = atomKK_->k_type.d_view;
  h_type = atomKK_->k_type.h_view;
  mask = atomKK_->mask;
  d_mask = atomKK_->k_mask.d_view;
  h_mask = atomKK_->k_mask.h_view;
  image = atomKK_->image;
  d_image = atomKK_->k_image.d_view;
  h_image = atomKK_->k_image.h_view;

  x = atomKK_->x;
  d_x = atomKK_->k_x.d_view;
  h_x = atomKK_->k_x.h_view;
  v = atomKK_->v;
  d_v = atomKK_->k_v.d_view;
  h_v = atomKK_->k_v.h_view;
  f = atomKK_->f;
  d_f = atomKK_->k_f.d_view;
  h_f = atomKK_->k_f.h_view;

  eval_f_stencil_md = atomKK_->eval_f_stencil_md;
  h_eval_f_stencil_md = atomKK_->k_eval_f_stencil_md.h_view;
  d_eval_f_stencil_md = atomKK_->k_eval_f_stencil_md.d_view;

  eval_mask_stencil_md = atomKK_->eval_mask_stencil_md;
  actually_eval_mask_stencil_md = atomKK_->actually_eval_mask_stencil_md;
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVecAtomicKokkos::copy(int i, int j, int delflag)
{
  if (i != j) { assert(h_tag[i] != h_tag[j]); }
  h_tag[j] = h_tag[i];
  h_type[j] = h_type[i];
  mask[j] = mask[i];
  h_image[j] = h_image[i];
  h_x(j, 0) = h_x(i, 0);
  h_x(j, 1) = h_x(i, 1);
  h_x(j, 2) = h_x(i, 2);
  h_v(j, 0) = h_v(i, 0);
  h_v(j, 1) = h_v(i, 1);
  h_v(j, 2) = h_v(i, 2);

  if (atom->nextra_grow) {
    assert(false);
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i, j, delflag);
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType, int PBC_FLAG> struct AtomVecAtomicKokkos_PackBorder {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d _buf;
  const typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  const typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  const typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  const typename ArrayTypes<DeviceType>::t_int_1d _type;
  const typename ArrayTypes<DeviceType>::t_int_1d _mask;
  X_FLOAT _dx, _dy, _dz;

  AtomVecAtomicKokkos_PackBorder(const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
                                 const typename ArrayTypes<DeviceType>::t_int_2d_const &list,
                                 const int &iswap,
                                 const typename ArrayTypes<DeviceType>::t_x_array &x,
                                 const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
                                 const typename ArrayTypes<DeviceType>::t_int_1d &type,
                                 const typename ArrayTypes<DeviceType>::t_int_1d &mask,
                                 const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz) :
      _buf(buf),
      _list(list), _iswap(iswap), _x(x), _tag(tag), _type(type), _mask(mask), _dx(dx), _dy(dy),
      _dz(dz)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const
  {
    const int j = _list(_iswap, i);
    if (PBC_FLAG == 0) {
      _buf(i, 0) = _x(j, 0);
      _buf(i, 1) = _x(j, 1);
      _buf(i, 2) = _x(j, 2);
      _buf(i, 3) = d_ubuf(_tag(j)).d;
      _buf(i, 4) = d_ubuf(_type(j)).d;
      _buf(i, 5) = d_ubuf(_mask(j)).d;
    } else {
      _buf(i, 0) = _x(j, 0) + _dx;
      _buf(i, 1) = _x(j, 1) + _dy;
      _buf(i, 2) = _x(j, 2) + _dz;
      _buf(i, 3) = d_ubuf(_tag(j)).d;
      _buf(i, 4) = d_ubuf(_type(j)).d;
      _buf(i, 5) = d_ubuf(_mask(j)).d;
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::pack_border_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                                            DAT::tdual_xfloat_2d buf, int iswap, int pbc_flag,
                                            int *pbc, ExecutionSpace space)
{
  X_FLOAT dx, dy, dz;

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (space == Host) {
      AtomVecAtomicKokkos_PackBorder<LMPHostType, 1> f(buf.view<LMPHostType>(),
                                                       k_sendlist.view<LMPHostType>(), iswap, h_x,
                                                       h_tag, h_type, h_mask, dx, dy, dz);
      Kokkos::parallel_for(n, f);
    } else {
      AtomVecAtomicKokkos_PackBorder<LMPDeviceType, 1> f(buf.view<LMPDeviceType>(),
                                                         k_sendlist.view<LMPDeviceType>(), iswap,
                                                         d_x, d_tag, d_type, d_mask, dx, dy, dz);
      Kokkos::parallel_for(n, f);
    }

  } else {
    dx = dy = dz = 0;
    if (space == Host) {
      AtomVecAtomicKokkos_PackBorder<LMPHostType, 0> f(buf.view<LMPHostType>(),
                                                       k_sendlist.view<LMPHostType>(), iswap, h_x,
                                                       h_tag, h_type, h_mask, dx, dy, dz);
      Kokkos::parallel_for(n, f);
    } else {
      AtomVecAtomicKokkos_PackBorder<LMPDeviceType, 0> f(buf.view<LMPDeviceType>(),
                                                         k_sendlist.view<LMPDeviceType>(), iswap,
                                                         d_x, d_tag, d_type, d_mask, dx, dy, dz);
      Kokkos::parallel_for(n, f);
    }
  }
  return n * 6;
}

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::pack_border(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m;
  double dx, dy, dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j, 0);
      buf[m++] = h_x(j, 1);
      buf[m++] = h_x(j, 2);
      buf[m++] = ubuf(h_tag(j)).d;
      buf[m++] = ubuf(h_type(j)).d;
      buf[m++] = ubuf(h_mask(j)).d;
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
      buf[m++] = h_x(j, 0) + dx;
      buf[m++] = h_x(j, 1) + dy;
      buf[m++] = h_x(j, 2) + dz;
      buf[m++] = ubuf(h_tag(j)).d;
      buf[m++] = ubuf(h_type(j)).d;
      buf[m++] = ubuf(h_mask(j)).d;
    }
  }

  if (atom->nextra_border) {
    assert(false);
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n, list, &buf[m]);
  }

  return m;
}

int AtomVecAtomicKokkos::pack_data_to_process_stencil_md(int num_send_force, int* force_idx_list, int* force_size_list,
                                    int num_segments, int* segment_idxs, int* segment_types, int* segment_lengths,
                                    int* local_list, double* buf, int* pbc_flags) {
    if (DEBUG_SEND_RECV_DATA) {
        int m = 0;

        assert(num_send_force >= 0 && num_send_force <= 100000);

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

        int local_list_idx = 0;

        for (int i = 0; i < num_segments; i++) {
            int segment_type = segment_types[i];
            int segment_size = segment_lengths[i];
            if (segment_type == LOCAL_SEGMENT_TYPE) {
                for (int j = 0; j < segment_size; j++) {
                    int idx = local_list[local_list_idx++];
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

        local_list_idx = 0;

        for (int i = 0; i < num_segments; i++) {
            int segment_type = segment_types[i];
            int segment_size = segment_lengths[i];
            if (segment_type == LOCAL_SEGMENT_TYPE) {
                for (int j = 0; j < segment_size; j++) {
                    int idx = local_list[local_list_idx++];
                    tagint tag_ = tag[idx];
                    // TODO: Kokkos-ify
                    buf[m++] = ubuf(tag_).d;
                    buf[m++] = v[idx][0];
                    buf[m++] = v[idx][1];
                    buf[m++] = v[idx][2];
                }
            } else {
                assert(segment_type == GHOST_SEGMENT_TYPE);
                int segment_idx = segment_idxs[i];
                for (int j = 0; j < segment_size; j++) {
                    int idx = segment_idx + j;
                    tagint tag_ = tag[idx];
                    // TODO: Kokkos-ify
                    buf[m++] = ubuf(tag_).d;
                    buf[m++] = v[idx][0];
                    buf[m++] = v[idx][1];
                    buf[m++] = v[idx][2];
                }
            }
        }

        // TODO: maybe add this method in Kokkos
        // modified_stencil_md(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK, this);

        return m;
    }
}

int AtomVecAtomicKokkos::pack_data_stencil_md(int num_send_force, int num_send_pos,
                             int* force_idx_list, int* force_size_list,
                             int* pos_idx_list, int* pos_size_list,
                             int* local_to_ghost_list,
                             int num_segments, int* segment_types, int* segment_idxs, int* segment_sizes,
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
              buf[m++] = ubuf(h_tag(idx)).d;
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
              buf[m++] = ubuf(h_tag(idx)).d;
              buf[m++] = v[idx][0];
              buf[m++] = v[idx][1];
              buf[m++] = v[idx][2];
          }
      }

      int send_pos = m - send_force;

      int local_list_idx = 0;

      for (int i = 0; i < num_segments; i++) {
          int segment_type = segment_types[i];
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

      /*
      local_list_idx = 0;
      for (int i = 0; i < num_segments; i++) {
          int segment_type = segment_types[i];
          int segment_size = segment_sizes[i];
          if (segment_type == LOCAL_SEGMENT_TYPE) {
              for (int j = 0; j < segment_size; j++) {
                  int idx = local_to_ghost_list[local_list_idx++];
                  tagint tag_ = tag[idx];
                  // TODO: Kokkos-ify
                  buf[m++] = ubuf(tag_).d;
                  buf[m++] = v[idx][0];
                  buf[m++] = v[idx][1];
                  buf[m++] = v[idx][2];
              }
          } else {
              assert(segment_type == GHOST_SEGMENT_TYPE);
              int segment_idx = segment_idxs[i];
              for (int j = 0; j < segment_size; j++) {
                  int idx = segment_idx + j;
                  tagint tag_ = tag[idx];
                  // TODO: Kokkos-ify
                  buf[m++] = ubuf(tag_).d;
                  buf[m++] = v[idx][0];
                  buf[m++] = v[idx][1];
                  buf[m++] = v[idx][2];
              }
          }
      }
      */

      // TODO: maybe add this method in Kokkos
      // modified_stencil_md(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK, this);

      return m;
  }
}

void AtomVecAtomicKokkos::unpack_data_stencil_md(int num_recv_force, int num_recv_pos,
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
  }
}

int AtomVecAtomicKokkos::pack_border_stencil_md(int n, int *list, double *buf, int *pbc_flag,
                                                int **pbc) {
  assert(false);
  return 0;
}

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::pack_border_vel(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m;
  double dx, dy, dz, dvx, dvy, dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j, 0);
      buf[m++] = h_x(j, 1);
      buf[m++] = h_x(j, 2);
      buf[m++] = ubuf(h_tag(j)).d;
      buf[m++] = ubuf(h_type(j)).d;
      buf[m++] = ubuf(h_mask(j)).d;
      buf[m++] = h_v(j, 0);
      buf[m++] = h_v(j, 1);
      buf[m++] = h_v(j, 2);
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
        buf[m++] = h_x(j, 0) + dx;
        buf[m++] = h_x(j, 1) + dy;
        buf[m++] = h_x(j, 2) + dz;
        buf[m++] = ubuf(h_tag(j)).d;
        buf[m++] = ubuf(h_type(j)).d;
        buf[m++] = ubuf(h_mask(j)).d;
        buf[m++] = h_v(j, 0);
        buf[m++] = h_v(j, 1);
        buf[m++] = h_v(j, 2);
      }
    } else {
      dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
      dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
      dvz = pbc[2] * h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = h_x(j, 0) + dx;
        buf[m++] = h_x(j, 1) + dy;
        buf[m++] = h_x(j, 2) + dz;
        buf[m++] = ubuf(h_tag(j)).d;
        buf[m++] = ubuf(h_type(j)).d;
        buf[m++] = ubuf(h_mask(j)).d;
        if (mask[i] & deform_groupbit) {
          buf[m++] = h_v(j, 0) + dvx;
          buf[m++] = h_v(j, 1) + dvy;
          buf[m++] = h_v(j, 2) + dvz;
        } else {
          buf[m++] = h_v(j, 0);
          buf[m++] = h_v(j, 1);
          buf[m++] = h_v(j, 2);
        }
      }
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n, list, &buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> struct AtomVecAtomicKokkos_UnpackBorder {
  typedef DeviceType device_type;

  const typename ArrayTypes<DeviceType>::t_xfloat_2d_const _buf;
  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  typename ArrayTypes<DeviceType>::t_int_1d _type;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  int _first;

  AtomVecAtomicKokkos_UnpackBorder(const typename ArrayTypes<DeviceType>::t_xfloat_2d_const &buf,
                                   typename ArrayTypes<DeviceType>::t_x_array &x,
                                   typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
                                   typename ArrayTypes<DeviceType>::t_int_1d &type,
                                   typename ArrayTypes<DeviceType>::t_int_1d &mask,
                                   const int &first) :
      _buf(buf),
      _x(x), _tag(tag), _type(type), _mask(mask), _first(first){};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const
  {
    _x(i + _first, 0) = _buf(i, 0);
    _x(i + _first, 1) = _buf(i, 1);
    _x(i + _first, 2) = _buf(i, 2);
    _tag(i + _first) = (tagint) d_ubuf(_buf(i, 3)).i;
    _type(i + _first) = (int) d_ubuf(_buf(i, 4)).i;
    _mask(i + _first) = (int) d_ubuf(_buf(i, 5)).i;
    //      printf("%i %i %lf %lf %lf %i
    //      BORDER\n",_tag(i+_first),i+_first,_x(i+_first,0),_x(i+_first,1),_x(i+_first,2),_type(i+_first));
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecAtomicKokkos::unpack_border_kokkos(const int &n, const int &first,
                                               const DAT::tdual_xfloat_2d &buf,
                                               ExecutionSpace space)
{
  atomKK->modified(space, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK);
  while (first + n >= nmax) grow(0);
  atomKK->modified(space, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK);
  if (space == Host) {
    struct AtomVecAtomicKokkos_UnpackBorder<LMPHostType> f(buf.view<LMPHostType>(), h_x, h_tag,
                                                           h_type, h_mask, first);
    Kokkos::parallel_for(n, f);
  } else {
    struct AtomVecAtomicKokkos_UnpackBorder<LMPDeviceType> f(buf.view<LMPDeviceType>(), d_x, d_tag,
                                                             d_type, d_mask, first);
    Kokkos::parallel_for(n, f);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecAtomicKokkos::unpack_border(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  while (last > nmax) grow(0);

  for (i = first; i < last; i++) {
    h_x(i, 0) = buf[m++];
    h_x(i, 1) = buf[m++];
    h_x(i, 2) = buf[m++];
    h_tag(i) = (tagint) ubuf(buf[m++]).i;
    h_type(i) = (int) ubuf(buf[m++]).i;
    h_mask(i) = (int) ubuf(buf[m++]).i;
  }

  atomKK->modified(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
}

int AtomVecAtomicKokkos::unpack_border_stencil_md(int n, int first, double *buf, Atom *atom_,
                                                  int zoid_num) {
  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
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

      h_x(insert_idx, 0) = x0;
      h_x(insert_idx, 1) = x1;
      h_x(insert_idx, 2) = x2;
      h_tag(insert_idx) = tag_;
      h_type(insert_idx) = type_;
      h_mask(insert_idx) = mask_;

      h_v(insert_idx, 0) = v0;
      h_v(insert_idx, 1) = v1;
      h_v(insert_idx, 2) = v2;

      inserted_tags.insert(tag_);
      insert_idx++;
    }
  }
  atomKK_->modified_stencil_md(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK, atom_);

  if (atom->nextra_border) {
    assert(false);
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
  }

  return insert_idx - first;
}

/* ---------------------------------------------------------------------- */

void AtomVecAtomicKokkos::unpack_border_vel(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  while (last > nmax) grow(0);

  for (i = first; i < last; i++) {
    h_x(i, 0) = buf[m++];
    h_x(i, 1) = buf[m++];
    h_x(i, 2) = buf[m++];
    h_tag(i) = (tagint) ubuf(buf[m++]).i;
    h_type(i) = (int) ubuf(buf[m++]).i;
    h_mask(i) = (int) ubuf(buf[m++]).i;
    h_v(i, 0) = buf[m++];
    h_v(i, 1) = buf[m++];
    h_v(i, 2) = buf[m++];
  }

  atomKK->modified(Host, X_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | V_MASK);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->unpack_border(n, first, &buf[m]);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> struct AtomVecAtomicKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array_randomread _x;
  typename AT::t_v_array_randomread _v;
  typename AT::t_tagint_1d_randomread _tag;
  typename AT::t_int_1d_randomread _type;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_imageint_1d_randomread _image;
  typename AT::t_x_array _xw;
  typename AT::t_v_array _vw;
  typename AT::t_tagint_1d _tagw;
  typename AT::t_int_1d _typew;
  typename AT::t_int_1d _maskw;
  typename AT::t_imageint_1d _imagew;

  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _nlocal, _dim;
  X_FLOAT _lo, _hi;

  AtomVecAtomicKokkos_PackExchangeFunctor(const AtomKokkos *atom,
                                          const typename AT::tdual_xfloat_2d buf,
                                          typename AT::tdual_int_1d sendlist,
                                          typename AT::tdual_int_1d copylist, int nlocal, int dim,
                                          X_FLOAT lo, X_FLOAT hi) :
      _x(atom->k_x.view<DeviceType>()),
      _v(atom->k_v.view<DeviceType>()), _tag(atom->k_tag.view<DeviceType>()),
      _type(atom->k_type.view<DeviceType>()), _mask(atom->k_mask.view<DeviceType>()),
      _image(atom->k_image.view<DeviceType>()), _xw(atom->k_x.view<DeviceType>()),
      _vw(atom->k_v.view<DeviceType>()), _tagw(atom->k_tag.view<DeviceType>()),
      _typew(atom->k_type.view<DeviceType>()), _maskw(atom->k_mask.view<DeviceType>()),
      _imagew(atom->k_image.view<DeviceType>()), _sendlist(sendlist.template view<DeviceType>()),
      _copylist(copylist.template view<DeviceType>()), _nlocal(nlocal), _dim(dim), _lo(lo), _hi(hi)
  {
    const size_t elements = 11;
    const int maxsendlist =
        (buf.template view<DeviceType>().extent(0) * buf.template view<DeviceType>().extent(1)) /
        elements;

    buffer_view<DeviceType>(_buf, buf, maxsendlist, elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &mysend) const
  {
    const int i = _sendlist(mysend);
    _buf(mysend, 0) = 11;
    _buf(mysend, 1) = _x(i, 0);
    _buf(mysend, 2) = _x(i, 1);
    _buf(mysend, 3) = _x(i, 2);
    _buf(mysend, 4) = _v(i, 0);
    _buf(mysend, 5) = _v(i, 1);
    _buf(mysend, 6) = _v(i, 2);
    _buf(mysend, 7) = d_ubuf(_tag[i]).d;
    _buf(mysend, 8) = d_ubuf(_type[i]).d;
    _buf(mysend, 9) = d_ubuf(_mask[i]).d;
    _buf(mysend, 10) = d_ubuf(_image[i]).d;
    const int j = _copylist(mysend);

    if (j > -1) {
      _xw(i, 0) = _x(j, 0);
      _xw(i, 1) = _x(j, 1);
      _xw(i, 2) = _x(j, 2);
      _vw(i, 0) = _v(j, 0);
      _vw(i, 1) = _v(j, 1);
      _vw(i, 2) = _v(j, 2);
      _tagw[i] = _tag(j);
      _typew[i] = _type(j);
      _maskw[i] = _mask(j);
      _imagew[i] = _image(j);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf,
                                              DAT::tdual_int_1d k_sendlist,
                                              DAT::tdual_int_1d k_copylist, ExecutionSpace space,
                                              int dim, X_FLOAT lo, X_FLOAT hi)
{
  if (nsend >
      (int) (k_buf.view<LMPHostType>().extent(0) * k_buf.view<LMPHostType>().extent(1)) / 11) {
    int newsize = nsend * 11 / k_buf.view<LMPHostType>().extent(1) + 1;
    k_buf.resize(newsize, k_buf.view<LMPHostType>().extent(1));
  }
  if (space == Host) {
    AtomVecAtomicKokkos_PackExchangeFunctor<LMPHostType> f(atomKK, k_buf, k_sendlist, k_copylist,
                                                           atom->nlocal, dim, lo, hi);
    Kokkos::parallel_for(nsend, f);
    return nsend * 11;
  } else {
    AtomVecAtomicKokkos_PackExchangeFunctor<LMPDeviceType> f(atomKK, k_buf, k_sendlist, k_copylist,
                                                             atom->nlocal, dim, lo, hi);
    Kokkos::parallel_for(nsend, f);
    return nsend * 11;
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::pack_exchange(int i, double *buf)
{
  int m = 1;
  buf[m++] = h_x(i, 0);
  buf[m++] = h_x(i, 1);
  buf[m++] = h_x(i, 2);
  buf[m++] = h_v(i, 0);
  buf[m++] = h_v(i, 1);
  buf[m++] = h_v(i, 2);
  buf[m++] = ubuf(h_tag(i)).d;
  buf[m++] = ubuf(h_type(i)).d;
  buf[m++] = ubuf(h_mask(i)).d;
  buf[m++] = ubuf(h_image(i)).d;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> struct AtomVecAtomicKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array _x;
  typename AT::t_v_array _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;

  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _nlocal;
  int _dim;
  X_FLOAT _lo, _hi;

  AtomVecAtomicKokkos_UnpackExchangeFunctor(const AtomKokkos *atom,
                                            const typename AT::tdual_xfloat_2d buf,
                                            typename AT::tdual_int_1d nlocal, int dim, X_FLOAT lo,
                                            X_FLOAT hi) :
      _x(atom->k_x.view<DeviceType>()),
      _v(atom->k_v.view<DeviceType>()), _tag(atom->k_tag.view<DeviceType>()),
      _type(atom->k_type.view<DeviceType>()), _mask(atom->k_mask.view<DeviceType>()),
      _image(atom->k_image.view<DeviceType>()), _nlocal(nlocal.template view<DeviceType>()),
      _dim(dim), _lo(lo), _hi(hi)
  {
    const size_t elements = 11;
    const int maxsendlist =
        (buf.template view<DeviceType>().extent(0) * buf.template view<DeviceType>().extent(1)) /
        elements;

    buffer_view<DeviceType>(_buf, buf, maxsendlist, elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &myrecv) const
  {
    X_FLOAT x = _buf(myrecv, _dim + 1);
    if (x >= _lo && x < _hi) {
      int i = Kokkos::atomic_fetch_add(&_nlocal(0), 1);
      _x(i, 0) = _buf(myrecv, 1);
      _x(i, 1) = _buf(myrecv, 2);
      _x(i, 2) = _buf(myrecv, 3);
      _v(i, 0) = _buf(myrecv, 4);
      _v(i, 1) = _buf(myrecv, 5);
      _v(i, 2) = _buf(myrecv, 6);
      _tag[i] = (tagint) d_ubuf(_buf(myrecv, 7)).i;
      _type[i] = (int) d_ubuf(_buf(myrecv, 8)).i;
      _mask[i] = (int) d_ubuf(_buf(myrecv, 9)).i;
      _image[i] = (imageint) d_ubuf(_buf(myrecv, 10)).i;
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv, int nlocal,
                                                int dim, X_FLOAT lo, X_FLOAT hi,
                                                ExecutionSpace space)
{
  while (nlocal + nrecv / 11 >= nmax) grow(0);

  if (space == Host) {
    k_count.h_view(0) = nlocal;
    AtomVecAtomicKokkos_UnpackExchangeFunctor<LMPHostType> f(atomKK, k_buf, k_count, dim, lo, hi);
    Kokkos::parallel_for(nrecv / 11, f);
    return k_count.h_view(0);
  } else {
    k_count.h_view(0) = nlocal;
    k_count.modify<LMPHostType>();
    k_count.sync<LMPDeviceType>();
    AtomVecAtomicKokkos_UnpackExchangeFunctor<LMPDeviceType> f(atomKK, k_buf, k_count, dim, lo, hi);
    Kokkos::parallel_for(nrecv / 11, f);
    k_count.modify<LMPDeviceType>();
    k_count.sync<LMPHostType>();

    return k_count.h_view(0);
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecAtomicKokkos::unpack_exchange(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);
  atomKK->modified(Host, X_MASK | V_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK);

  int m = 1;
  h_x(nlocal, 0) = buf[m++];
  h_x(nlocal, 1) = buf[m++];
  h_x(nlocal, 2) = buf[m++];
  h_v(nlocal, 0) = buf[m++];
  h_v(nlocal, 1) = buf[m++];
  h_v(nlocal, 2) = buf[m++];
  h_tag(nlocal) = (tagint) ubuf(buf[m++]).i;
  h_type(nlocal) = (int) ubuf(buf[m++]).i;
  h_mask(nlocal) = (int) ubuf(buf[m++]).i;
  h_image(nlocal) = (imageint) ubuf(buf[m++]).i;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->unpack_exchange(nlocal, &buf[m]);

  atom->nlocal++;
  return m;
}

int AtomVecAtomicKokkos::pack_exchange_stencil_md(int i, double *buf, int* pbc) {
    int m = 1;
    double dx = pbc[0] * domain->prd[0];
    double dy = pbc[1] * domain->prd[1];
    double dz = pbc[2] * domain->prd[2];
    buf[m++] = h_x(i, 0) + dx;
    buf[m++] = h_x(i, 1) + dy;
    buf[m++] = h_x(i, 2) + dz;

    buf[m++] = h_v(i, 0);
    buf[m++] = h_v(i, 1);
    buf[m++] = h_v(i, 2);

    buf[m++] = ubuf(h_tag(i)).d;
    buf[m++] = ubuf(h_type(i)).d;
    buf[m++] = ubuf(h_mask(i)).d;
    buf[m++] = ubuf(h_image(i)).d;

    /*
    buf[m++] = atom_->x[i][0] + dx;
    buf[m++] = atom_->x[i][1] + dy;
    buf[m++] = atom_->x[i][2] + dz;

    buf[m++] = atom_->v[i][0];
    buf[m++] = atom_->v[i][1];
    buf[m++] = atom_->v[i][1];

    buf[m++] = ubuf(atom_->tag[i]).d;
    buf[m++] = ubuf(atom_->type[i]).d;
    buf[m++] = ubuf(atom_->mask[i]).d;
    buf[m++] = ubuf(atom_->image[i]).d;
    */

    buf[0] = m;
    return m;
}

int AtomVecAtomicKokkos::unpack_exchange_stencil_md(double *buf, Atom *atom_, Domain *domain_,
                                                    int flag) {
  int last_idx;
  if (flag == LAMMPS_SEND_LOCAL) {
      last_idx = atom_->nlocal;
  } else if (flag == LAMMPS_SEND_GHOST) {
      last_idx = atom_->nlocal + atom_->nghost;
  } else {
      assert(false);
  }

  if (last_idx == nmax) {
      grow_stencil_md(0, atom_);
  }

  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
  atomKK_->modified_stencil_md(Host, X_MASK | V_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK, atom_);

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
  for (int i = 0; i < last_idx; i++) {
      if (atom_->tag[i] == tag_) {
          found_tag = true;
          break;
      }
  }

  if (!found_tag) {
      h_x(last_idx, 0) = x0;
      h_x(last_idx, 1) = x1;
      h_x(last_idx, 2) = x2;

      h_v(last_idx, 0) = v0;
      h_v(last_idx, 1) = v1;
      h_v(last_idx, 2) = v2;

      h_tag(last_idx) = tag_;
      h_type(last_idx) = type_;
      h_mask(last_idx) = mask_;
      h_image(last_idx) = image_;

      if (atom_->nextra_grow) {
          assert(false);
          for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
              m += modify->fix[atom->extra_grow[iextra]]->unpack_exchange(last_idx, &buf[m]);
      }

      if (flag == LAMMPS_SEND_LOCAL) {
          atom_->nlocal++;
      } else if (flag == LAMMPS_SEND_GHOST) {
          atom_->nghost++;
      }
  }

  return m;
  /*
  int nlocal = atom_->nlocal;
  if (nlocal == nmax) { grow_stencil_md(0, atom_); }

  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
  atomKK_->modified_stencil_md(
      Host, X_MASK | V_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK, atom_);

  int m = 1;
  h_x(nlocal, 0) = buf[m++];
  h_x(nlocal, 1) = buf[m++];
  h_x(nlocal, 2) = buf[m++];

  h_v(nlocal, 0) = buf[m++];
  h_v(nlocal, 1) = buf[m++];
  h_v(nlocal, 2) = buf[m++];

  h_tag(nlocal) = (tagint) ubuf(buf[m++]).i;
  h_type(nlocal) = (int) ubuf(buf[m++]).i;
  h_mask(nlocal) = (int) ubuf(buf[m++]).i;
  h_image(nlocal) = (imageint) ubuf(buf[m++]).i;

  if (atom_->nextra_grow) {
    assert(false);
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->unpack_exchange(nlocal, &buf[m]);
  }

  if ()

  atom_->nlocal++;
  return m;
  */
}

void AtomVecAtomicKokkos::add_local_atom_stencil_md(Atom *atom_, Domain *domain_, double *coord,
                                                    double *vel, tagint tag, int type, int mask,
                                                    imageint image)
{
  int nlocal = atom_->nlocal;
  if (nlocal == nmax) { grow_stencil_md(0, atom_); }

  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
  atomKK_->modified_stencil_md(
      Host, X_MASK | V_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK, atom_);

  int m = 1;
  h_x(nlocal, 0) = coord[0];
  h_x(nlocal, 1) = coord[1];
  h_x(nlocal, 2) = coord[2];
  h_v(nlocal, 0) = vel[0];
  h_v(nlocal, 1) = vel[1];
  h_v(nlocal, 2) = vel[2];
  h_tag(nlocal) = tag;
  h_type(nlocal) = type;
  h_mask(nlocal) = mask;
  h_image(nlocal) = image;

  atom_->nlocal++;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVecAtomicKokkos::size_restart()
{
  int i;

  int nlocal = atom->nlocal;
  int n = 11 * nlocal;

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

int AtomVecAtomicKokkos::pack_restart(int i, double *buf)
{
  atomKK->sync(Host, X_MASK | V_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK);

  int m = 1;
  buf[m++] = h_x(i, 0);
  buf[m++] = h_x(i, 1);
  buf[m++] = h_x(i, 2);
  buf[m++] = ubuf(h_tag(i)).d;
  buf[m++] = ubuf(h_type(i)).d;
  buf[m++] = ubuf(h_mask(i)).d;
  buf[m++] = ubuf(h_image(i)).d;
  buf[m++] = h_v(i, 0);
  buf[m++] = h_v(i, 1);
  buf[m++] = h_v(i, 2);

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVecAtomicKokkos::unpack_restart(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store) memory->grow(atom->extra, nmax, atom->nextra_store, "atom:extra");
  }
  atomKK->modified(Host, X_MASK | V_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | IMAGE_MASK);

  int m = 1;
  h_x(nlocal, 0) = buf[m++];
  h_x(nlocal, 1) = buf[m++];
  h_x(nlocal, 2) = buf[m++];
  h_tag(nlocal) = (tagint) ubuf(buf[m++]).i;
  h_type(nlocal) = (int) ubuf(buf[m++]).i;
  h_mask(nlocal) = (int) ubuf(buf[m++]).i;
  h_image(nlocal) = (imageint) ubuf(buf[m++]).i;
  h_v(nlocal, 0) = buf[m++];
  h_v(nlocal, 1) = buf[m++];
  h_v(nlocal, 2) = buf[m++];

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

void AtomVecAtomicKokkos::create_atom(int itype, double *coord)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    // if(nlocal>2) printf("typeA: %i %i\n",type[0],type[1]);
    atomKK->modified(Host, ALL_MASK);
    grow(0);
    // if(nlocal>2) printf("typeB: %i %i\n",type[0],type[1]);
  }
  atomKK->modified(Host, ALL_MASK);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  h_x(nlocal, 0) = coord[0];
  h_x(nlocal, 1) = coord[1];
  h_x(nlocal, 2) = coord[2];
  h_mask[nlocal] = 1;
  h_image[nlocal] = ((tagint) IMGMAX << IMG2BITS) | ((tagint) IMGMAX << IMGBITS) | IMGMAX;
  h_v(nlocal, 0) = 0.0;
  h_v(nlocal, 1) = 0.0;
  h_v(nlocal, 2) = 0.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecAtomicKokkos::data_atom(double *coord, imageint imagetmp,
                                    const std::vector<std::string> &values)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  h_tag[nlocal] = utils::inumeric(FLERR, values[0], true, lmp);
  h_type[nlocal] = utils::inumeric(FLERR, values[1], true, lmp);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR, "Invalid atom type in Atoms section of data file");

  h_x(nlocal, 0) = coord[0];
  h_x(nlocal, 1) = coord[1];
  h_x(nlocal, 2) = coord[2];

  h_image[nlocal] = imagetmp;

  h_mask[nlocal] = 1;
  h_v(nlocal, 0) = 0.0;
  h_v(nlocal, 1) = 0.0;
  h_v(nlocal, 2) = 0.0;

  atomKK->modified(Host, ALL_MASK);

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecAtomicKokkos::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = h_tag[i];
    buf[i][1] = h_type[i];
    buf[i][2] = h_x(i, 0);
    buf[i][3] = h_x(i, 1);
    buf[i][4] = h_x(i, 2);
    buf[i][5] = (h_image[i] & IMGMASK) - IMGMAX;
    buf[i][6] = (h_image[i] >> IMGBITS & IMGMASK) - IMGMAX;
    buf[i][7] = (h_image[i] >> IMG2BITS) - IMGMAX;
  }
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecAtomicKokkos::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp, "%d %d %-1.16e %-1.16e %-1.16e %d %d %d\n", (int) buf[i][0], (int) buf[i][1],
            buf[i][2], buf[i][3], buf[i][4], (int) buf[i][5], (int) buf[i][6], (int) buf[i][7]);
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

double AtomVecAtomicKokkos::memory_usage()
{
  double bytes = 0;

  if (atom->memcheck("tag")) bytes += memory->usage(tag, nmax);
  if (atom->memcheck("type")) bytes += memory->usage(type, nmax);
  if (atom->memcheck("mask")) bytes += memory->usage(mask, nmax);
  if (atom->memcheck("image")) bytes += memory->usage(image, nmax);
  if (atom->memcheck("x")) bytes += memory->usage(x, nmax, 3);
  if (atom->memcheck("v")) bytes += memory->usage(v, nmax, 3);
  if (atom->memcheck("f")) bytes += memory->usage(f, nmax * commKK->nthreads, 3);

  return bytes;
}

/* ---------------------------------------------------------------------- */

void AtomVecAtomicKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.sync<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.sync<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPHostType>();
  }
}

void AtomVecAtomicKokkos::sync_stencil_md(ExecutionSpace space, unsigned int mask, Atom *atom_)
{
  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
  if (space == Device) {
    if (mask & X_MASK) atomKK_->k_x.sync<LMPDeviceType>();
    if (mask & V_MASK) atomKK_->k_v.sync<LMPDeviceType>();
    if (mask & F_MASK) atomKK_->k_f.sync<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK_->k_tag.sync<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK_->k_type.sync<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK_->k_mask.sync<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK_->k_image.sync<LMPDeviceType>();
    atomKK_->k_eval_mask_stencil_md.sync<LMPDeviceType>();
    atomKK_->k_actually_eval_mask_stencil_md.sync<LMPDeviceType>();
    atomKK_->k_eval_f_stencil_md.sync<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK_->k_x.sync<LMPHostType>();
    if (mask & V_MASK) atomKK_->k_v.sync<LMPHostType>();
    if (mask & F_MASK) atomKK_->k_f.sync<LMPHostType>();
    if (mask & TAG_MASK) atomKK_->k_tag.sync<LMPHostType>();
    if (mask & TYPE_MASK) atomKK_->k_type.sync<LMPHostType>();
    if (mask & MASK_MASK) atomKK_->k_mask.sync<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK_->k_image.sync<LMPHostType>();
    atomKK_->k_eval_mask_stencil_md.sync<LMPHostType>();
    atomKK_->k_actually_eval_mask_stencil_md.sync<LMPHostType>();
    atomKK_->k_eval_f_stencil_md.sync<LMPHostType>();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecAtomicKokkos::sync_overlapping_device(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x, space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v, space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f, space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag, space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type, space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask, space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image, space);
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x, space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v, space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f, space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag, space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type, space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask, space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image, space);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecAtomicKokkos::modified(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.modify<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.modify<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPHostType>();
  }
}

void AtomVecAtomicKokkos::modified_stencil_md(ExecutionSpace space, unsigned int mask, Atom *atom_)
{
  AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
  if (space == Device) {
    if (mask & X_MASK) atomKK_->k_x.modify<LMPDeviceType>();
    if (mask & V_MASK) atomKK_->k_v.modify<LMPDeviceType>();
    if (mask & F_MASK) atomKK_->k_f.modify<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK_->k_tag.modify<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK_->k_type.modify<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK_->k_mask.modify<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK_->k_image.modify<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK_->k_x.modify<LMPHostType>();
    if (mask & V_MASK) atomKK_->k_v.modify<LMPHostType>();
    if (mask & F_MASK) atomKK_->k_f.modify<LMPHostType>();
    if (mask & TAG_MASK) atomKK_->k_tag.modify<LMPHostType>();
    if (mask & TYPE_MASK) atomKK_->k_type.modify<LMPHostType>();
    if (mask & MASK_MASK) atomKK_->k_mask.modify<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK_->k_image.modify<LMPHostType>();
  }
}
