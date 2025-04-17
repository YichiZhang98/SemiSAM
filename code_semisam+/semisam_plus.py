from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2


def compute_epistemic_uncertainty(all_preds):

    predictions = torch.stack(all_preds)
    ensemble = torch.mean(predictions, dim=0)
    stacked_preds = torch.stack(all_preds)
    kl_distance = nn.KLDivLoss(reduction='none')
    uncertainty = []
    for pred in all_preds:
        exp_variance = torch.var(pred - ensemble)
        uncertainty.append(exp_variance)
    unc = torch.mean(torch.stack(uncertainty), dim=0)
    return unc


def finetune_model_predict3D_unc(img3D, gt3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    batch_size, ch, width, height, depth = img3D.shape
    pred_batch = torch.zeros((batch_size, ch, width, height, depth), device=device)
    unc_batch = torch.zeros((batch_size, ch, width, height, depth), device=device)

    for b in range(batch_size):
        click_points = []
        click_labels = []

        img3D_single = img3D[b:b + 1,...]
        gt3D_single = gt3D[b:b + 1,...]

        crop_size = 128

        if prev_masks is None:
            prev_masks = torch.zeros_like(img3D_single).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size // 4, crop_size // 4, crop_size // 4))

        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(img3D_single.to(device))  # (1, 384, 16, 16, 16)

        all_preds = []
        all_preds.append(gt3D_single)
        for num_click in range(num_clicks):
            with torch.no_grad():
                if num_click > 1:
                    click_method = get_next_click3D_torch_2
                batch_points, batch_labels = click_method(prev_masks.to(device), gt3D_single.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)
                points_la = torch.cat(batch_labels, dim=0).to(device)

                click_points.append(points_co)
                click_labels.append(points_la)

                points_input = points_co
                labels_input = points_la

                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_input, labels_input],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device),
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                prev_masks = F.interpolate(low_res_masks, size=img3D_single.shape[-3:], mode='trilinear', align_corners=False)

                medsam_seg_prob = prev_masks  # (B, 1, W, H, D)
                medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)
                all_preds.append(medsam_seg_prob)


        uncertainty = compute_epistemic_uncertainty(all_preds)
        pred_batch[b:b + 1, :, :, :, :] = medsam_seg
        unc_batch[b:b + 1, :, :, :, :] = uncertainty

    return pred_batch, uncertainty





def finetune_model_predict3D_point(img3D, gt3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    batch_size, ch, width, height, depth = img3D.shape
    pred_batch = torch.zeros((batch_size, ch, width, height, depth), device=device)

    for b in range(batch_size):
        click_points = []
        click_labels = []

        img3D_single = img3D[b:b+1,...]
        gt3D_single = gt3D[b:b+1,...]

        crop_size = 128

        if prev_masks is None:
            prev_masks = torch.zeros_like(img3D_single).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size // 4, crop_size // 4, crop_size // 4))

        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(img3D_single.to(device))  # (1, 384, 16, 16, 16)

        for num_click in range(num_clicks):
            with torch.no_grad():
                if num_click > 1:
                    click_method = get_next_click3D_torch_2
                batch_points, batch_labels = click_method(prev_masks.to(device), gt3D_single.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)
                points_la = torch.cat(batch_labels, dim=0).to(device)

                click_points.append(points_co)
                click_labels.append(points_la)

                points_input = points_co
                labels_input = points_la

                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_input, labels_input],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device),
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                prev_masks = F.interpolate(low_res_masks, size=img3D_single.shape[-3:], mode='trilinear', align_corners=False)

                medsam_seg_prob = prev_masks  # (B, 1, W, H, D)
                medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)
        pred_batch[b:b+1,:,:,:,:] = medsam_seg

    return pred_batch



def finetune_model_predict3D_mask(img3D, gt3D, sam_model_tune, device='cuda'):
    batch_size, ch, width, height, depth = img3D.shape
    pred_batch = torch.zeros((batch_size, ch, width, height, depth), device=device)

    for b in range(batch_size):
        img3D_single = img3D[b:b + 1,...]
        gt3D_single = gt3D[b:b + 1,...]

        crop_size = 128
        low_res_gt = F.interpolate(gt3D_single.float(), size=(crop_size // 4, crop_size // 4, crop_size // 4))

        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(img3D_single.to(device))
            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=None,
                boxes=None,
                masks=low_res_gt.to(device)
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            masks = F.interpolate(low_res_masks, size=img3D_single.shape[-3:], mode='trilinear', align_corners=False)
            medsam_seg_prob = masks
            medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)

        pred_batch[b:b + 1, :, :, :, :] = medsam_seg

    return pred_batch




def semisam_branch(volume_batch, mask, generalist='SAM-Med3D', prompt='mask', device='cuda'):
    if generalist == 'SAM-Med3D':
        checkpoint_path = '../ckpt/sam_med3d.pth'
    elif generalist == 'SAM-Med3D-turbo':
        checkpoint_path = '../ckpt/sam_med3d_turbo.pth'
    elif generalist == 'SegAnyPET':
        checkpoint_path = '../ckpt/sa_pet.pth' 

    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)   
    model_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = model_dict['model_state_dict']
    sam_model_tune.load_state_dict(state_dict)

    unc = []
    if prompt == 'point':
        samseg_mask = finetune_model_predict3D_point(
            volume_batch, mask, sam_model_tune, device=device,
            click_method= get_next_click3D_torch_ritm, num_clicks=10, 
            prev_masks=None) 
    elif prompt == 'mask':
        samseg_mask = finetune_model_predict3D_mask(
            volume_batch, mask, sam_model_tune, device=device) 
    elif prompt == 'unc':
        samseg_mask, unc = finetune_model_predict3D_unc(
            volume_batch, mask, sam_model_tune, device=device,
            click_method= get_next_click3D_torch_ritm, num_clicks=10, 
            prev_masks=None) 
    
    return samseg_mask, unc