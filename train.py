import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from dataset import PolarDataset
from loss import normal_loss
from networks.PolarPoseNet import build_model
from options import PolarPoseOptions
from utils import readlines, compute_rotation_matrix_from_ortho6d, \
    transform_pts_batch, pose_from_predictions, \
    get_closest_rot_batch, normalize_image


def train_model(opts, obj_id, save_model=True, iter_print=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device ", device)

    with open(opts.object_id2name_path, 'r') as load_model_id:
        object_metainfo = json.load(load_model_id)
    assert str(obj_id) in object_metainfo

    obj_name = object_metainfo[str(obj_id)]["name"]
    print("Start training the object estimator of {}".format(obj_name))

    refractive_index = object_metainfo[str(obj_id)]["refractive_index"]
    print("with refractive index of {}".format(refractive_index))

    # Load object model meta-information of the bbox and symmetry!!
    with open(opts.object_metainfo_path, 'r') as load_model_info:
        objects_info = json.load(load_model_info)
    obj_info = objects_info[obj_name]

    train_files = opts.train_split + str('_') + obj_name + '.txt'
    val_files = opts.val_split + str('_') + obj_name + '.txt'

    train_filenames = readlines(train_files)
    val_filenames = readlines(val_files)

    train_dataset = PolarDataset(data_path=opts.data_path, filename=train_filenames,
                                 obj_id=obj_id, obj_name=obj_name, obj_info=obj_info,
                                 refractive_index=refractive_index, object_ply_path=opts.object_ply_path, istrain=True)

    print('Size of trainset ', len(train_dataset))

    val_dataset = PolarDataset(data_path=opts.data_path, filename=val_filenames,
                                 obj_id=obj_id, obj_name=obj_name, obj_info=obj_info,
                                 refractive_index=refractive_index, object_ply_path=opts.object_ply_path, istrain=False)

    print('Size of valset ', len(val_dataset))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False,
                                             num_workers=opts.num_workers, pin_memory=True, drop_last=True)

    net = build_model(opts)
    net = net.to(device)

    # custom loss function and optimizer
    criterion_center = nn.L1Loss(reduction='mean')
    criterion_tz = nn.L1Loss(reduction='mean')
    criterion_mask = nn.L1Loss(reduction='mean')
    criterion_point_matching = nn.L1Loss(reduction='mean')
    criterion_nocs = nn.L1Loss(reduction='mean')

    # specify optimizer
    optimizer = optim.Adam(net.parameters(), lr=opts.learning_rate, weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=0.5)

    # training loop
    val_loss_min = np.Inf
    log_path_train = os.path.join(opts.log_dir, 'train')
    log_path_val = os.path.join(opts.log_dir, 'val')

    writer_train = SummaryWriter(log_path_train)
    writer_val = SummaryWriter(log_path_val)

    for epoch in range(opts.num_epochs):
        t0 = time.time()
        train_loss = 0
        val_loss = 0
        print("------ Epoch ", epoch, " ---------")
        net.train()
        print("training")
        for iter, inputs in enumerate(train_loader):

            for key in [
                "roi_img_0",
                "roi_img_45",
                "roi_img_90",
                "roi_img_135",
                "roi_N_diff",
                "roi_N_spec1",
                "roi_N_spec2",
                "roi_dolp",
                "roi_aolp",
                "roi_cams",
                "model_points",
                "roi_extents",
                "roi_coord_2d",
                "gt_ratio_translation",
                "resize_ratio",
                "roi_centers",
                "roi_whs",
                "roi_mask",
                "roi_gt_normals",
                "roi_nocs",
                "gt_post_mat",
             ]:
                inputs[key] = inputs[key].to(device=device, dtype=torch.float32)

            mask_train, normals_train, nocs_train, rotation_train, translation_train = net(inputs)

            pred_rotation_mat = compute_rotation_matrix_from_ortho6d(rotation_train).to(device)

            pred_ego_rot, pred_trans = pose_from_predictions(
                pred_rotation_mat,
                pred_centroids=translation_train[:, :2],
                pred_z_vals=translation_train[:, 2:3],
                roi_cams=inputs["roi_cams"],
                roi_centers=inputs["roi_centers"],
                resize_ratios=inputs["resize_ratio"],
                roi_whs=inputs["roi_whs"],
                eps=1e-4,
                is_allo="allo",
                z_type="REL"
            )
            pred_ego_rot = pred_ego_rot.to(torch.float32).to(device)

            # point matching loss
            if "sym_info" in inputs:
                gt_rots = get_closest_rot_batch(pred_ego_rot, inputs["gt_post_mat"][:, :3, :3],
                                                sym_infos=inputs["sym_info"])
                points_tgt = transform_pts_batch(inputs["model_points"], gt_rots, t=None)
            else:
                points_tgt = transform_pts_batch(inputs["model_points"], inputs["gt_post_mat"][:, :3, :3], t=None)

            points_est = transform_pts_batch(inputs["model_points"], pred_ego_rot, t=None)

            weights = 1.0 / inputs["roi_extents"].max(1, keepdim=True)[0]
            weights = weights.view(-1, 1, 1).to(torch.float32).to(device)
            loss_point_matching_train = criterion_point_matching(weights * points_est, weights * points_tgt.detach())

            loss_centroid_train = criterion_center(translation_train[:, :2], inputs["gt_ratio_translation"][:, :2])
            loss_tz_train = criterion_tz(translation_train[:, -1], inputs["gt_ratio_translation"][:, -1])
            loss_mask_train = criterion_mask(mask_train[:, 0, :, :], inputs["roi_mask"][:, 0, :, :])
            loss_normals_train = normal_loss.loss_fn_cosine(normals_train, inputs["roi_gt_normals"].detach(),
                                                            reduction='elementwise_mean')

            loss_x = criterion_nocs(nocs_train[:, 0, :, :] * inputs["roi_mask"][:, 0, :, :],
                                    inputs["roi_nocs"][:, 0, :, :] * inputs["roi_mask"][:, 0, :, :])
            loss_y = criterion_nocs(nocs_train[:, 1, :, :] * inputs["roi_mask"][:, 0, :, :],
                                    inputs["roi_nocs"][:, 1, :, :] * inputs["roi_mask"][:, 0, :, :])
            loss_z = criterion_nocs(nocs_train[:, 2, :, :] * inputs["roi_mask"][:, 0, :, :],
                                    inputs["roi_nocs"][:, 2, :, :] * inputs["roi_mask"][:, 0, :, :])

            loss_train = loss_point_matching_train + loss_centroid_train + loss_tz_train + loss_mask_train + loss_x + loss_y + loss_z + loss_normals_train

            optimizer.zero_grad()

            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item()

            if iter % iter_print == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\tPMLoss: {:.6f},\tCentroidLoss: {:.6f},\tTzLoss: {:.6f},\tMaskLoss: {:.6f},\tNormalLoss: {:.6f}'
                    '\tXLoss: {:.6f},\tYLoss: {:.6f},\tZLoss: {:.6f}'.
                    format(epoch, iter * len(inputs["roi_img_0"]), len(train_loader.dataset),
                           100. * iter / len(train_loader), loss_train.item(),
                           loss_point_matching_train.item(), loss_centroid_train.item(),
                           loss_tz_train.item(),
                           loss_mask_train.item(), loss_normals_train.item(),
                           loss_x.item(), loss_y.item(), loss_z.item()))

        net.eval()
        print("validating")
        for iter, inputs in enumerate(val_loader):

            for key in [
                "roi_img_0",
                "roi_img_45",
                "roi_img_90",
                "roi_img_135",
                "roi_N_diff",
                "roi_N_spec1",
                "roi_N_spec2",
                "roi_dolp",
                "roi_aolp",
                "roi_cams",
                "model_points",
                "roi_extents",
                "roi_coord_2d",
                "gt_ratio_translation",
                "resize_ratio",
                "roi_centers",
                "roi_whs",
                "roi_mask",
                "roi_gt_normals",
                "roi_nocs",
                "gt_post_mat",
            ]:
                inputs[key] = inputs[key].to(device=device, dtype=torch.float32)

            mask_val, normals_val, nocs_val, rotation_val, translation_val = net(inputs)

            pred_rotation_mat = compute_rotation_matrix_from_ortho6d(rotation_val).to(device)

            pred_ego_rot, pred_trans = pose_from_predictions(
                pred_rotation_mat,
                pred_centroids=translation_val[:, :2],
                pred_z_vals=translation_val[:, 2:3],
                roi_cams=inputs["roi_cams"],
                roi_centers=inputs["roi_centers"],
                resize_ratios=inputs["resize_ratio"],
                roi_whs=inputs["roi_whs"],
                eps=1e-4,
                is_allo="allo",
                z_type="REL"
            )
            pred_ego_rot = pred_ego_rot.to(torch.float32).to(device)

            # point matching loss
            if "sym_info" in inputs:
                gt_rots = get_closest_rot_batch(pred_ego_rot, inputs["gt_post_mat"][:, :3, :3],
                                                sym_infos=inputs["sym_info"])
                points_tgt = transform_pts_batch(inputs["model_points"], gt_rots, t=None)
            else:
                points_tgt = transform_pts_batch(inputs["model_points"], inputs["gt_post_mat"][:, :3, :3], t=None)

            points_est = transform_pts_batch(inputs["model_points"], pred_ego_rot, t=None)

            weights = 1.0 / inputs["roi_extents"].max(1, keepdim=True)[0]
            weights = weights.view(-1, 1, 1).to(torch.float32).to(device)
            loss_point_matching_val = criterion_point_matching(weights * points_est, weights * points_tgt.detach())

            loss_centroid_val = criterion_center(translation_val[:, :2], inputs["gt_ratio_translation"][:, :2])
            loss_tz_val = criterion_tz(translation_val[:, -1], inputs["gt_ratio_translation"][:, -1])
            loss_mask_val = criterion_mask(mask_val[:, 0, :, :], inputs["roi_mask"][:, 0, :, :])
            loss_normals_val = normal_loss.loss_fn_cosine(normals_val, inputs["roi_gt_normals"].detach(),
                                                            reduction='elementwise_mean')
            loss_x = criterion_nocs(nocs_val[:, 0, :, :] * inputs["roi_mask"][:, 0, :, :],
                                    inputs["roi_nocs"][:, 0, :, :] * inputs["roi_mask"][:, 0, :, :])
            loss_y = criterion_nocs(nocs_val[:, 1, :, :] * inputs["roi_mask"][:, 0, :, :],
                                    inputs["roi_nocs"][:, 1, :, :] * inputs["roi_mask"][:, 0, :, :])
            loss_z = criterion_nocs(nocs_val[:, 2, :, :] * inputs["roi_mask"][:, 0, :, :],
                                    inputs["roi_nocs"][:, 2, :, :] * inputs["roi_mask"][:, 0, :, :])

            loss_val = loss_point_matching_val + loss_centroid_val + loss_tz_val + loss_mask_val + loss_x + loss_y + loss_z + loss_normals_val

            val_loss += loss_val.item()

        scheduler.step()

        # calculate average losses
        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        t_end = time.time()
        print(f'{t_end - t0} seconds')
        writer_train.add_scalar('train loss', train_loss, epoch)
        writer_val.add_scalar('val loss', val_loss, epoch)
        writer_train.add_scalar('epoch time', t_end - t0, epoch)
        for j in range(min(4, opts.batch_size)):
            writer_train.add_image("0_img/{}".format(j),
                                   inputs["roi_img_0"][j], epoch)
            writer_train.add_image("1_dolp/{}".format(j),
                                   inputs["roi_dolp"][j], epoch)
            writer_train.add_image("2_aolp/{}".format(j),
                                   normalize_image(inputs["roi_aolp"][j]), epoch)
            writer_train.add_image("3_N_diff/{}".format(j),
                                   (inputs["roi_N_diff"][j]*0.5 + 0.5), epoch)
            writer_train.add_image("4_N_spec1/{}".format(j),
                                   (inputs["roi_N_spec1"][j]*0.5 + 0.5), epoch)
            writer_train.add_image("5_N_spec2/{}".format(j),
                                   (inputs["roi_N_spec2"][j]*0.5 + 0.5), epoch)
            writer_train.add_image("6_mask_gt/{}".format(j),
                                   inputs["roi_mask"][j], epoch)
            writer_train.add_image("7_mask_pred/{}".format(j),
                                   normalize_image(mask_train[j]), epoch)
            writer_train.add_image("8_normals_gt/{}".format(j),
                                   (inputs["roi_gt_normals"][j]*0.5 + 0.5).data, epoch)
            writer_train.add_image("9_normals_pred_raw/{}".format(j),
                                   ((normals_train[j]*0.5 + 0.5)).data, epoch)
            writer_train.add_image("10_gt_nocs/{}".format(j),
                                   (inputs["roi_nocs"][j]).data, epoch)
            writer_train.add_image("11_nocs_pred_raw/{}".format(j),
                                   ((nocs_train[j])).data, epoch)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, val_loss))
        print('Epoch: {} \t lr: {}'.format(epoch, scheduler.get_lr()))

        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min,val_loss))
            if save_model:
                    torch.save(net.state_dict(), 'log/{}.pth'.format(str(int(epoch))))
            val_loss_min = val_loss

    writer_train.close()
    writer_val.close()


if __name__ == '__main__':
    options = PolarPoseOptions()
    opts = options.parse()

    train_model(opts, obj_id=1)
