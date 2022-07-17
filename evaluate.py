import os
import time
import numpy as np
import torch
from scipy import spatial
from torch.utils.data import Dataset
from dataset import PolarDataset
from utils import readlines, compute_rotation_matrix_from_ortho6d, pose_from_predictions
from networks.PolarPoseNet import build_model
from options import PolarPoseOptions
from lib.customized_bop_utils import load_ply
import json
from loss import metric_calculator


def evaluate(opts, obj_id, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device ", device)

    with open(opts.object_id2name_path, 'r') as load_model_id:
        object_name_n = json.load(load_model_id)
    assert str(obj_id) in object_name_n

    obj_name = object_name_n[str(obj_id)]["name"]
    print("Start evaluating the object estimator of {}".format(obj_name))

    refractive_index = object_name_n[str(obj_id)]["refractive_index"]
    print("with refractive index of {}".format(refractive_index))

    object_plymodel_path = os.path.join(opts.object_ply_path, obj_name + '.ply')
    model = load_ply(object_plymodel_path)
    # Load object model meta-information of the bbox
    with open(opts.object_metainfo_path, 'r') as load_model_info:
        objects_info = json.load(load_model_info)

    obj_info = objects_info[str(obj_name)]
    diameter = obj_info["diameter"] * 1000
    print("Load information of the {}, which has the diameter of {} millimeters".format(obj_name, diameter))

    net = build_model(opts)
    print("loading model from ", weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net = net.to(device)
    net.eval()

    test_filenames = opts.test_split + str('_') + obj_name + '.txt'
    test_filenames = readlines(test_filenames)

    test_dataset = PolarDataset(data_path=opts.data_path, filename=test_filenames,
                                obj_id=obj_id, obj_name=obj_name, obj_info=obj_info,
                                refractive_index=refractive_index, object_ply_path=opts.object_ply_path, istrain=False)

    print('Size of test set ', len(test_dataset))

    ad_10s = []

    running_mean = []
    running_median = []
    running_percentage1 = []
    running_percentage2 = []
    running_percentage3 = []
    t0 = time.time()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=opts.num_workers,
                                              pin_memory=True, shuffle=False)
    with torch.no_grad():
        for idx, inputs in enumerate(test_loader):
            try:

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
                    "gt_post_mat",
                    "roi_mask",
                    "roi_gt_normals",
                ]:
                    inputs[key] = inputs[key].to(device=device, dtype=torch.float32)

                mask_train, normals_train, nocs_train, rotation_train, translation_train = net(inputs)

                # transform the [B, 6] rotation to [B, 3, 3] rotation matrix
                pred_rotation_mat = compute_rotation_matrix_from_ortho6d(rotation_train).to(device)  # [B, 3, 3]

                pred_ego_rot, pred_trans = pose_from_predictions(
                    pred_rotation_mat,
                    pred_centroids=translation_train[:, :2],
                    pred_z_vals=translation_train[:, 2:3],  # must be [B, 1]
                    roi_cams=inputs["roi_cams"],
                    roi_centers=inputs["roi_centers"],
                    resize_ratios=inputs["resize_ratio"],
                    roi_whs=inputs["roi_whs"],
                    eps=1e-4,
                    is_allo="allo",
                    z_type="REL"
                )
                pred_ego_rot = pred_ego_rot.to(torch.float32).to(device)

                t_pred = pred_trans.to('cpu').detach().numpy().T

                R_pred = pred_ego_rot.to('cpu').detach().squeeze(0).numpy()  # 3, 3

                R_gt = inputs["gt_post_mat"][0, :3, :3].to('cpu').detach().numpy()
                t_gt = inputs["gt_post_mat"][:, :3, 3].to('cpu').detach().numpy().T * 1000  # evaluate in millimeters!

                if "symmetries_discrete" in obj_info or "symmetries_continuous" in obj_info:
                    adi_10 = evaluate_adi_metric(R_pred, t_pred, R_gt, t_gt, model['pts'], diameter, percentage=0.1)
                    error = adi_10
                else:
                    add_10 = evaluate_add_metric(R_pred, t_pred, R_gt, t_gt, model['pts'], diameter, percentage=0.1)
                    error = add_10

                ad_10s.append(error)

                # evaluate normal prediction
                loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels = metric_calculator(
                    normals_train.squeeze(0), inputs["roi_gt_normals"].squeeze(0),
                    mask=inputs["roi_mask"].squeeze(0).squeeze(0))

                running_mean.append(loss_deg_mean.item())
                running_median.append(loss_deg_median.item())
                running_percentage1.append(percentage_1.item())
                running_percentage2.append(percentage_2.item())
                running_percentage3.append(percentage_3.item())

            except Exception as e:
                print(e)
                continue

    ADD_10 = np.mean(ad_10s)
    t_end = time.time()
    print(f'{t_end - t0} seconds')
    print('ADD-10 metric: {}'.format(ADD_10 * 100))

    mean = sum(running_mean) / len(test_loader)
    median = sum(running_median) / len(test_loader)
    percentage_1 = sum(running_percentage1) / len(test_loader)
    percentage_2 = sum(running_percentage2) / len(test_loader)
    percentage_3 = sum(running_percentage3) / len(test_loader)
    print('mean metric: {} degree'.format(mean))
    print('median metric: {} degree'.format(median))
    print('percentage_1 metric: {} percent'.format(percentage_1))
    print('percentage_2 metric: {} percent'.format(percentage_2))
    print('percentage_3 metric: {} percent'.format(percentage_3))

    return


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def transform_pts_Rt_2d(pts, R, t, K):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :param K: 3x3 intrinsic matrix
    :return: nx2 ndarray with transformed 2D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))  # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    return pts_2d


def add(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable.

    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable
    views.

    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def evaluate_add_metric(R_est, t_est, R_gt, t_gt, pts, diameter, percentage=0.1):
    add_err = add(R_est, t_est, R_gt, t_gt, pts)
    add_metric = float(add_err < percentage * diameter)
    return add_metric


def evaluate_adi_metric(R_est, t_est, R_gt, t_gt, pts, diameter, percentage=0.1):
    adi_err = adi(R_est, t_est, R_gt, t_gt, pts)
    adi_metric = float(adi_err < percentage * diameter)
    return adi_metric


if __name__ == '__main__':
    options = PolarPoseOptions()
    opts = options.parse()
    opts.pretrained = False

    evaluate(opts, obj_id=1, weights_path="")
