import torch
import torch.nn as nn
import random
import torchvision
import numpy as np


def get_content_extension_loss(feats_s, feats_sw, feats_w, gts, queue):

    B, C, H, W = feats_s.shape # feat feature size (B X C X H X W)

    # uniform sampling with a size of 64 x 64 for source and wild-stylized source feature maps
    H_W_resize = 64
    HW = H_W_resize * H_W_resize
    
    upsample_n = nn.Upsample(size=[H_W_resize, H_W_resize], mode='nearest')
    feats_s_flat = upsample_n(feats_s)
    feats_sw_flat = upsample_n(feats_sw)

    feats_s_flat = feats_s_flat.contiguous().view(B, C, -1) # B X C X H X W > B X C X (H X W)
    feats_sw_flat = feats_sw_flat.contiguous().view(B, C, -1) # B X C X H X W > B X C X (H X W)
    gts_flat = upsample_n(gts.unsqueeze(1).float()).squeeze(1).long().view(B, HW)

    # uniform sampling with a size of 16 x 16 for wild feature map
    H_W_resize_w = 16
    HW_w = H_W_resize_w * H_W_resize_w
 
    upsample_n_w = nn.Upsample(size=[H_W_resize_w, H_W_resize_w], mode='nearest')
    feats_w_flat = upsample_n_w(feats_w)
    feats_w_flat = torch.einsum("bchw->bhwc", feats_w_flat).contiguous().view(B*H_W_resize_w*H_W_resize_w, C) # B X C X H X W > (B X H X W) X C

    # normalize feature of each pixel
    feats_s_flat = nn.functional.normalize(feats_s_flat, p=2, dim=1)
    feats_sw_flat = nn.functional.normalize(feats_sw_flat, p=2, dim=1)
    feats_w_flat = nn.functional.normalize(feats_w_flat, p=2, dim=1).detach() # (B X H X W) X C


    # log(dot(feats_s_flat, feats_sw_flat))
    T = 0.07
    logits_sce = torch.bmm(feats_s_flat.transpose(1,2), feats_sw_flat) # dot product: B X (H X W) X (H X W)
    logits_sce = (torch.clamp(logits_sce, min=-1, max=1))/T
    
    # compute ignore mask: same-class (excluding self) + unknown-labeled pixels
    # compute positive mask (same-class)
    logits_mask_sce_ignore = torch.eq(gts_flat.unsqueeze(2), gts_flat.unsqueeze(1)) # pos:1, neg:0. B X (H X W) X (H X W)
    # include unknown-labeled pixel
    logits_mask_sce_ignore = include_unknown(logits_mask_sce_ignore, gts_flat)

    # exclude self-pixel
    logits_mask_sce_ignore *= ~torch.eye(HW,HW).type(torch.cuda.BoolTensor).unsqueeze(0).expand([B, -1, -1]) # self:1, other:0. B X (H X W) X (H X W)

    # compute positive mask for cross entropy loss: B X (H X W)
    logits_mask_sce_pos = torch.linspace(start=0, end=HW-1, steps=HW).unsqueeze(0).expand([B, -1]).type(torch.cuda.LongTensor)

    # compute unknown-labeled mask for cross entropy loss: B X (H X W)
    logits_mask_sce_unk = torch.zeros_like(logits_mask_sce_pos, dtype=torch.bool)
    logits_mask_sce_unk[gts_flat>254] = True

    # compute loss_sce
    eps = 1e-5
    logits_sce[logits_mask_sce_ignore] = -1/T
    CELoss = nn.CrossEntropyLoss(reduction='none')
    loss_sce = CELoss(logits_sce.transpose(1,2), logits_mask_sce_pos)
    loss_sce = ((loss_sce * (~logits_mask_sce_unk)).sum(1) / ((~logits_mask_sce_unk).sum(1) + eps)).mean()


    # get wild content closest to wild-stylized source content
    idx_sim_bs = 512
    index_nearest_neighbours = (torch.randn(0)).type(torch.cuda.LongTensor)
    for idx_sim in range(int(np.ceil(HW/idx_sim_bs))):
        idx_sim_start = idx_sim*idx_sim_bs
        idx_sim_end = min((idx_sim+1)*idx_sim_bs, HW)
        similarity_matrix = torch.einsum("bcn,cq->bnq", feats_sw_flat[:,:,idx_sim_start:idx_sim_end].type(torch.cuda.HalfTensor), queue['wild'].type(torch.cuda.HalfTensor)) # B X (H X W) X Q
        index_nearest_neighbours = torch.cat((index_nearest_neighbours,torch.argmax(similarity_matrix, dim=2)),dim=1) # B X (H X W)
    # similarity_matrix = torch.einsum("bcn,cq->bnq", feats_sw_flat, queue['wild']) # B X (H X W) X Q
    # index_nearest_neighbours = torch.argmax(similarity_matrix, dim=2) # B X (H X W)
    del similarity_matrix
    nearest_neighbours = torch.index_select(queue['wild'], dim=1, index=index_nearest_neighbours.view(-1)).view(C, B, HW) # C X B X (H X W)

    # compute exp(dot(feats_s_flat, nearest_neighbours))
    logits_wce_pos = torch.einsum("bcn,cbn->bn", feats_s_flat, nearest_neighbours) # dot product: B X C X (H X W) & C X B X (H X W) => B X (H X W)
    logits_wce_pos = (torch.clamp(logits_wce_pos, min=-1, max=1))/T
    exp_logits_wce_pos = torch.exp(logits_wce_pos)

    # compute negative mask of logits_sce
    logits_mask_sce_neg = ~torch.eq(gts_flat.unsqueeze(2), gts_flat.unsqueeze(1)) # pos:0, neg:1. B X (H X W) X (H X W)

    # exclude unknown-labeled pixels from negative samples
    logits_mask_sce_neg = exclude_unknown(logits_mask_sce_neg, gts_flat)

    # sum exp(neg samples)
    exp_logits_sce_neg = (torch.exp(logits_sce) * logits_mask_sce_neg).sum(2) # B X (H X W)

    # Compute log_prob
    log_prob_wce = logits_wce_pos - torch.log(exp_logits_wce_pos + exp_logits_sce_neg) # B X (H X W)

    # Compute loss_wce
    loss_wce = -((log_prob_wce * (~logits_mask_sce_unk)).sum(1) / ((~logits_mask_sce_unk).sum(1) + eps)).mean()


    # enqueue wild contents
    sup_enqueue = feats_w_flat # Q X C # (B X H X W) X C
    _dequeue_and_enqueue(queue, sup_enqueue)

    # compute content extension learning loss
    loss_cel = loss_sce + loss_wce

    return loss_cel

def exclude_unknown(mask, gts):
    '''
    mask: [B, HW, HW]
    gts: [B, HW]
    '''
    mask = mask.transpose(1,2).contiguous()
    mask[gts>254,:] = False
    mask = mask.transpose(1,2).contiguous()

    return mask

def include_unknown(mask, gts):
    '''
    mask: [B, HW, HW]
    gts: [B, HW]
    '''
    mask = mask.transpose(1,2).contiguous()
    mask[gts>254,:] = True
    mask = mask.transpose(1,2).contiguous()

    return mask

@torch.no_grad()
def _dequeue_and_enqueue(queue, keys):
    # gather keys before updating queue
    keys = concat_all_gather(keys) # (B X H X W) X C
    
    batch_size = keys.shape[0]

    ptr = int(queue['wild_ptr'])

    # replace the keys at ptr (dequeue and enqueue)
    if (ptr + batch_size) <= queue['size']:
        # wild queue
        queue['wild'][:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % queue['size']  # move pointer
    else:
        # wild queue
        last_input_num = queue['size'] - ptr  
        queue['wild'][:,ptr:] = (keys.T)[:,:last_input_num]
        ptr = (ptr + batch_size) % queue['size']  # move pointer
        queue['wild'][:,:ptr] = (keys.T)[:,last_input_num:]
    queue['wild_ptr'][0] = ptr

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = varsize_tensor_all_gather(tensor)

    output = tensors_gather
    return output

def varsize_tensor_all_gather(tensor: torch.Tensor):
    tensor = tensor.contiguous()

    cuda_device = f'cuda:{torch.distributed.get_rank()}'
    size_tens = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=cuda_device)

    size_tens2 = [torch.ones_like(size_tens)
        for _ in range(torch.distributed.get_world_size())]
    
    torch.distributed.all_gather(size_tens2, size_tens)
    size_tens2 = torch.cat(size_tens2, dim=0).cpu()
    max_size = size_tens2.max()

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=cuda_device)
    padded[:tensor.shape[0]] = tensor

    ag = [torch.ones_like(padded)
        for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(ag,padded)
    ag = torch.cat(ag, dim=0)

    slices = []
    for i, sz in enumerate(size_tens2):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)
    
    return ret.to(tensor)
