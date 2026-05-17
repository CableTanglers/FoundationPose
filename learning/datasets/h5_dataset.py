# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import os,sys,h5py,bisect,io,json
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
from learning.datasets.pose_dataset import *


# AIC PATCH (2026-05-17 OOM mitigation, per GPT review of refine_network OOM):
# kornia.warp_perspective on the full 252-pose rotation grid into the
# original 480x640 resolution allocates ~3.3 GiB transient. On a 20 GiB
# GPU also hosting Gazebo (2.7 GiB), the cumulative ~17 GiB FP baseline
# + this transient OOMs by ~300-500 MiB. Chunked replacement processes
# `chunk` poses at a time, so peak transient drops to ~3.3 GiB / (252/chunk).
# Output is concatenated to match unchunked behavior bit-exactly.
def _warp_perspective_chunked(x, M, dsize, chunk=32, **kwargs):
    """Chunked equivalent of kornia.geometry.transform.warp_perspective for
    large batch dimensions. Handles both x.shape[0] == M.shape[0] (per-pose
    input) and x.shape[0] < M.shape[0] (broadcast input via .expand)."""
    N = M.shape[0]
    if N <= chunk:
        return kornia.geometry.transform.warp_perspective(x, M, dsize=dsize, **kwargs)
    outs = []
    same_shape = (x.shape[0] == N)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        if same_shape:
            xi = x[i:end]
        else:
            # x was .expand(N, ...) — slice the expanded shape per chunk.
            xi = x[:1].expand(end - i, *x.shape[1:])
        outs.append(kornia.geometry.transform.warp_perspective(
            xi, M[i:end], dsize=dsize, **kwargs
        ))
    return torch.cat(outs, dim=0)


def _depth2xyzmap_batch_chunked(depths, Ks, zfar, chunk=32):
    """Chunked depth2xyzmap_batch — Utils.py's version allocates
    Ks.expand(bs, N, 3, 3) which for bs=252 and N=480*640 is 2.79 GiB and
    triggers OOM on the second invocation (after xyz_mapAs is held in
    memory). Chunking cuts peak transient to ~350 MiB."""
    # Import here to avoid circular module load — Utils.py imports this module.
    from Utils import depth2xyzmap_batch
    N = depths.shape[0]
    if N <= chunk:
        return depth2xyzmap_batch(depths, Ks, zfar)
    outs = []
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        outs.append(depth2xyzmap_batch(depths[i:end], Ks[i:end], zfar))
    return torch.cat(outs, dim=0)




class PairH5Dataset(torch.utils.data.Dataset):
  def __init__(self, cfg, h5_file, mode='train', max_num_key=None, cache_data=None):
    self.cfg = cfg
    self.h5_file = h5_file
    self.mode = mode

    logging.info(f"self.h5_file:{self.h5_file}")
    self.n_perturb = None
    self.H_ori = None
    self.W_ori = None
    self.cache_data = cache_data

    if self.mode=='test':
      pass
    else:
      self.object_keys = []
      key_file = h5_file.replace('.h5','_keys.pkl')
      if os.path.exists(key_file):
        with open(key_file, 'rb') as ff:
          self.object_keys = pickle.load(ff)
        logging.info(f'object_keys loaded#:{len(self.object_keys)} from {key_file}')
        if max_num_key is not None:
          self.object_keys = self.object_keys[:max_num_key]
      else:
        with h5py.File(h5_file, 'r', libver='latest') as hf:
          for k in hf:
            self.object_keys.append(k)
            if max_num_key is not None and len(self.object_keys)>=max_num_key:
              logging.info("break due to max_num_key")
              break

      logging.info(f'self.object_keys#:{len(self.object_keys)}, max_num_key:{max_num_key}')

      with h5py.File(h5_file, 'r', libver='latest') as hf:
        group = hf[self.object_keys[0]]
        cnt = 0
        for k_perturb in group:
          if 'i_perturb' in k_perturb:
            cnt += 1
          if 'crop_ratio' in group[k_perturb]:
            self.cfg['crop_ratio'] = float(group[k_perturb]['crop_ratio'][()])
          if self.H_ori is None:
            if 'H_ori' in group[k_perturb]:
              self.H_ori = int(group[k_perturb]['H_ori'][()])
              self.W_ori = int(group[k_perturb]['W_ori'][()])
            else:
              self.H_ori = 540
              self.W_ori = 720
        self.n_perturb = cnt
        logging.info(f'self.n_perturb:{self.n_perturb}')


  def __len__(self):
    if self.mode=='test':
      return 1
    return len(self.object_keys)



  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = _depth2xyzmap_batch_chunked(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapAs[:,2:3]<0.001
    batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapBs = _depth2xyzmap_batch_chunked(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapBs[:,2:3]<0.001
    batch.xyz_mapBs = batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch



  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    '''Transform the batch before feeding to the network
    !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
    '''
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch




class TripletH5Dataset(PairH5Dataset):
  def __init__(self, cfg, h5_file, mode, max_num_key=None, cache_data=None):
    super().__init__(cfg, h5_file, mode, max_num_key, cache_data=cache_data)


  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      # AIC PATCH: chunked warp_perspective to bound peak transient memory.
      # The 252-pose batch into original 480x640 was the OOM site.
      depthAs_ori = _warp_perspective_chunked(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = _depth2xyzmap_batch_chunked(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      del depthAs_ori  # release before second warp
      batch.xyz_mapAs = _warp_perspective_chunked(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:,2:3]<0.1
    batch.xyz_mapAs = (batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      depthBs_ori = _warp_perspective_chunked(batch.depthBs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapBs = _depth2xyzmap_batch_chunked(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      del depthBs_ori
      batch.xyz_mapBs = _warp_perspective_chunked(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:,2:3]<0.1
    batch.xyz_mapBs = (batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch



class ScoreMultiPairH5Dataset(TripletH5Dataset):
  def __init__(self, cfg, h5_file, mode, max_num_key=None, cache_data=None):
    super().__init__(cfg, h5_file, mode, max_num_key, cache_data=cache_data)
    if mode in ['train', 'val']:
      self.cfg['train_num_pair'] = self.n_perturb


class PoseRefinePairH5Dataset(PairH5Dataset):
  def __init__(self, cfg, h5_file, mode='train', max_num_key=None, cache_data=None):
    super().__init__(cfg=cfg, h5_file=h5_file, mode=mode, max_num_key=max_num_key, cache_data=cache_data)

    if mode!='test':
      with h5py.File(h5_file, 'r', libver='latest') as hf:
        group = hf[self.object_keys[0]]
        for key_perturb in group:
          depthA = imageio.imread(group[key_perturb]['depthA'][()])
          depthB = imageio.imread(group[key_perturb]['depthB'][()])
          self.cfg['n_view'] = min(self.cfg['n_view'], depthA.shape[1]//depthB.shape[1])
          logging.info(f'n_view:{self.cfg["n_view"]}')
          self.trans_normalizer = group[key_perturb]['trans_normalizer'][()]
          if isinstance(self.trans_normalizer, np.ndarray):
            self.trans_normalizer = self.trans_normalizer.tolist()
          self.rot_normalizer = group[key_perturb]['rot_normalizer'][()]/180.0*np.pi
          logging.info(f'self.trans_normalizer:{self.trans_normalizer}, self.rot_normalizer:{self.rot_normalizer}')
          break


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    '''Transform the batch before feeding to the network
    !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
    '''
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch

