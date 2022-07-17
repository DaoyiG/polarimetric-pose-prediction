import json
import math
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def transform_pts_batch(pts, R, t=None):
    """
    Args:
        pts: (B,P,3)
        R: (B,3,3)
        t: (B,3,1)

    Returns:

    """
    bs = R.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bs, n_pts, 3)
    if t is not None:
        assert t.shape[0] == bs

    pts_transformed = R.view(bs, 1, 3, 3) @ pts.view(bs, n_pts, 3, 1)
    if t is not None:
        pts_transformed += t.view(bs, 1, 3, 1)
    return pts_transformed.squeeze(-1)  # (B, P, 3)


def angular_distance(r1, r2, reduction="mean"):
    """https://math.stackexchange.com/questions/90081/quaternion-distance
    https.

    ://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tool
    s.py.

    1 - <q1, q2>^2  <==> (1-cos(theta)) / 2
    """
    assert r1.shape == r2.shape
    if r1.shape[-1] == 4:
        return angular_distance_quat(r1, r2, reduction=reduction)
    else:
        return angular_distance_rot(r1, r2, reduction=reduction)


def angular_distance_quat(pred_q, gt_q, reduction="mean"):
    dist = 1 - torch.pow(torch.bmm(pred_q.view(-1, 1, 4), gt_q.view(-1, 4, 1)), 2)
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


def angular_distance_rot(m1, m2, reduction="mean"):
    m = torch.bmm(m1, m2.transpose(1, 2))  # b*3*3
    m_trace = torch.einsum("bii->b", m)  # batch trace
    cos = (m_trace - 1) / 2  # [-1, 1]
    # eps = 1e-6
    # cos = torch.clamp(cos, -1+eps, 1-eps)  # avoid nan
    # theta = torch.acos(cos)
    dist = (1 - cos) / 2  # [0, 1]
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


def transform_translation_parametrization(translation, intrinsics):
    """
    transform from (tx, ty, tz) parametrization to (cx, cy, tz)
    Args:
        translation: (3, )
        intrinsics: (3, 3)
    Returns:
        reparametrized translation: (3, )
    """
    tx, ty, tz = translation

    cx = intrinsics[0, 0] * tx / tz + intrinsics[0, 2]
    cy = intrinsics[1, 1] * ty / tz + intrinsics[1, 2]

    reparametrized_translation = np.array([cx, cy, tz])
    return reparametrized_translation


def transform_centroid_tz_to_translation(centroid_tz, intrinsics):
    """
    transform from (tx, ty, tz) parametrization to (cx, cy, tz)
    Args:
        centroid_tz: (3, )
        intrinsics: (3, 3)
    Returns:
         translation: (3, )
    """
    cx, cy, tz = centroid_tz

    tx = (cx - intrinsics[0, 2]) * tz / intrinsics[0, 0]
    ty = (cy - intrinsics[1, 2]) * tz / intrinsics[1, 1]

    translation = np.array([tx, ty, tz])
    return translation


def transform_centroid_tz_to_translation_batch(centroid_tz, intrinsics):
    """
    transform from (tx, ty, tz) parametrization to (cx, cy, tz)
    Args:
        centroid_tz: (B, 3)
        intrinsics: (3, 3)
    Returns:
         translation: (B, 3)
    """
    cx = centroid_tz[:, 0]
    cy = centroid_tz[:, 1]
    tz = centroid_tz[:, 2]

    tx = (cx - intrinsics[0, 2]) * tz / intrinsics[0, 0]
    ty = (cy - intrinsics[1, 2]) * tz / intrinsics[1, 1]

    translation = torch.zeros_like(centroid_tz)
    translation[:, 0] = tx
    translation[:, 1] = ty
    translation[:, 2] = tz

    return translation


def read_image_cv2(file_name, format=None):
    """# NOTE modified from detectron2, use cv2 instead of PIL Read an image
    into the given format.

    Args:
        file_name (str): image file path
        format (str): "BGR" | "RGB"
    Returns:
        image (np.ndarray): an HWC image
    """
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)  # BGR
    if format == "RGB":
        # flip channels if needed
        image = image[:, :, [2, 1, 0]]
    if format == "L":
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # BGR
        # image = np.expand_dims(image, -1)  # TODO: debug this when needed
    return image


def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    if mode is 'rgb':
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    elif mode is 'grayscale':
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')

    else:
        print("mast specify a mode to conver the image!")


def _make_square(im, fill_color=(0, 0, 0, 0)):
    """
    fill in black borders to images that are non-square
    Args:
        im (PIL image): input image
        fill_color: filling color

    Returns:
        new_im (PIL image): output image
    """

    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def mksquare_reverse(h, w):
    """
    helper function to compute the offset of height and width while making a rectangular image square
    """
    size = max(h, w)
    return int((size - h) / 2), int((size - w) / 2)


def mk_square(minx, miny, maxx, maxy, height, width):
    centerx = (minx + maxx) / 2
    centery = (miny + maxy) / 2
    l = max(maxx - minx, maxy - miny) / 2

    minx = int(max(min(centerx - l, height), 0))
    maxx = int(max(min(centerx + l, height), 0))
    miny = int(max(min(centery - l, width), 0))
    maxy = int(max(min(centery + l, width), 0))

    return minx, miny, maxx, maxy


# batch*n
def normalize_vector(v):
    v = F.normalize(v, p=2, dim=1)
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def mat_to_ortho6d_batch(rots):
    """
    bx3x3
    ---
    bx6
    """
    x = rots[:, :, 0]  # col x
    y = rots[:, :, 1]  # col y
    ortho6d = torch.cat([x, y], 1)  # bx6
    return ortho6d


def mat_to_ortho6d_np(rot):
    """
    3x3
    ---
    (6,)
    """
    x = rot[:3, 0]  # col x
    y = rot[:3, 1]  # col y
    ortho6d = np.concatenate([x, y])  # (6,)
    return ortho6d


def symmetric_rot(gt_rotation, gt_translation):
    """
    Remap the rotation for symmetric objects
    Args:
        gt_rotation: original ground truth rotation
        gt_translation: original ground truth translation

    Returns:
        remapped rotation matrix
    """
    camera_location = -np.matmul(np.linalg.inv(gt_rotation), gt_translation)
    camera_location_zx = camera_location[[2, 0]]
    camera_location_zx = camera_location_zx / np.linalg.norm(camera_location_zx)

    symmetry_direction_zx = np.array((1, 0))

    cos = np.dot(camera_location_zx, symmetry_direction_zx)
    angle = np.arccos(cos)

    direction = 1. if np.cross(camera_location_zx, symmetry_direction_zx) > 0 else -1

    rotation_y = Rotation.from_euler('y', direction * angle).as_matrix()

    gt_rotation = np.matmul(gt_rotation, rotation_y.T)
    return gt_rotation


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose


def egocentric_to_allocentric(ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    return allo_pose


def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[:, 0]  # Remove added Corner dimension here.
    return q_ego


def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.

    Note, output dims: NxMx4 with N being the batchsize and N the number
    of quaternions or 3D points to be transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = a[:, :, 1] * b[:, :, 0] + a[:, :, 2] * b[:, :, 3] - a[:, :, 3] * b[:, :, 2] + a[:, :, 0] * b[:, :, 1]
    y = -a[:, :, 1] * b[:, :, 3] + a[:, :, 2] * b[:, :, 0] + a[:, :, 3] * b[:, :, 1] + a[:, :, 0] * b[:, :, 2]
    z = a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 1] + a[:, :, 3] * b[:, :, 0] + a[:, :, 0] * b[:, :, 3]
    w = -a[:, :, 1] * b[:, :, 1] - a[:, :, 2] * b[:, :, 2] - a[:, :, 3] * b[:, :, 3] + a[:, :, 0] * b[:, :, 0]

    return torch.stack((w, x, y, z), dim=2)


def pose_from_predictions(
        pred_rots,
        pred_centroids,
        pred_z_vals,
        roi_cams,
        roi_centers,
        resize_ratios,
        roi_whs,
        eps=1e-4,
        is_allo=True,
        z_type="REL",
):
    """for train
    Args:
        pred_rots:
        pred_centroids:
        pred_z_vals: [B, 1]
        roi_cams: absolute cams
        roi_centers:
        roi_scales:
        roi_whs: (bw,bh) for bboxes
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG

    Returns:

    """
    if roi_cams.dim() == 2:
        roi_cams.unsqueeze_(0)
    assert roi_cams.dim() == 3, roi_cams.dim()
    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs[:, 0]) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs[:, 1]) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "ABS":
        z = pred_z_vals
    elif z_type == "REL":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    # NOTE: z must be [B,1]
    translation = torch.cat(
        [z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0], z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1], z],
        dim=1,
    )

    if pred_rots.ndim == 2 and pred_rots.shape[-1] == 4:
        pred_quats = pred_rots
        quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
        if is_allo:
            quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
        else:
            quat_ego = quat_allo
        rot_ego = quat2mat_torch(quat_ego)
    if pred_rots.ndim == 3 and pred_rots.shape[-1] == 3:  # Nx3x3
        if is_allo:
            rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
        else:
            rot_ego = pred_rots
    return rot_ego, translation


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    # translation: Nx3
    # rot_allo: Nx3x3
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    # print('quat', quat) # Bx4
    # print('norm_quat: ', norm_quat)  # Bx1
    norm_quat = quat / (norm_quat + eps)
    # print('normed quat: ', norm_quat)
    qw, qx, qy, qz = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [1.0 - (yY + zZ), xY - wZ, xZ + wY, xY + wZ, 1.0 - (xX + zZ), yZ - wX, xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=1
    ).reshape(B, 3, 3)

    # rotMat = torch.stack([
    #     qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    #     2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx),
    #     2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
    #     dim=1).reshape(B, 3, 3)

    # w2, x2, y2, z2 = qw*qw, qx*qx, qy*qy, qz*qz
    # wx, wy, wz = qw*qx, qw*qy, qw*qz
    # xy, xz, yz = qx*qy, qx*qz, qy*qz

    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.

    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt


def get_closest_rot_batch(pred_rots, gt_rots, sym_infos):
    """
    get closest gt_rots according to current predicted poses_est and sym_infos
    --------------------
    pred_rots: [B, 4] or [B, 3, 3]
    gt_rots: [B, 4] or [B, 3, 3]
    sym_infos: list [Kx3x3 or None],
        stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_gt_rots: [B, 3, 3]
    """
    batch_size = pred_rots.shape[0]
    device = pred_rots.device
    if pred_rots.shape[-1] == 4:
        pred_rots = quat2mat_torch(pred_rots[:, :4])
    if gt_rots.shape[-1] == 4:
        gt_rots = quat2mat_torch(gt_rots[:, :4])

    closest_gt_rots = gt_rots.clone().cpu().numpy()  # B,3,3

    for i in range(batch_size):
        closest_rot = get_closest_rot(pred_rots[i].detach().cpu().numpy(), gt_rots[i].cpu().numpy(), sym_infos[i])
        # TODO: automatically detect rot_gt's format in PM_Loss to avoid converting multiple times
        closest_gt_rots[i] = closest_rot
    closest_gt_rots = torch.tensor(closest_gt_rots, device=device, dtype=gt_rots.dtype)
    return closest_gt_rots


def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(low, high, width, dtype=np.float32)
    y = np.linspace(low, high, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy


def crop_resize_by_warp_affine(img, center, scale, output_w, output_h, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_w, int) and isinstance(output_h, int):
        output_size = (output_w, output_h)
    else:
        raise ValueError("the output size should be integer!")
    trans = get_affine_transform(center, scale, rot, output_w, output_h)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img


def get_affine_transform(center, scale, rot, output_w, output_h, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_w, (int, float)) and isinstance(output_h, (int, float)):
        output_size = (output_w, output_h)
    else:
        raise ValueError("the output size should be integer!")

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class NestedDefaultDict(defaultdict):
    """
    A helper class, defines a dictionary that could insert keys at any level
    """

    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def write_dict_to_json(dic, json_path):
    with open(json_path, 'w') as f:
        json.dump(dic, f)


def load_mesh(mesh_path, is_save=False, is_normalized=False, is_flipped=False):
    with open(mesh_path, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for l in lines:
        l = l.strip()
        words = l.split(' ')
        if words[0] == 'v':
            vertices.append([float(words[1]), float(words[2]), float(words[3])])
        if words[0] == 'f':
            face_words = [x.split('/')[0] for x in words]
            faces.append([int(face_words[1]) - 1, int(face_words[2]) - 1, int(face_words[3]) - 1])

    vertices = np.array(vertices, dtype=np.float64)
    # flip mesh to unity rendering
    if is_flipped:
        vertices[:, 2] = -vertices[:, 2]
    faces = np.array(faces, dtype=np.int32)

    if is_normalized:
        maxs = np.amax(vertices, axis=0)
        mins = np.amin(vertices, axis=0)
        diffs = maxs - mins
        assert diffs.shape[0] == 3
        vertices = vertices / np.linalg.norm(diffs)

    if is_save:
        np.savetxt(mesh_path.replace('.obj', '_vertices.txt'), X=vertices)

    return vertices, faces


class AddGaussianNoise(object):
    def __init__(self, std=5 / 255, mean=0.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def list_duplicates_of(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def align_pred_gt(pred_data, inst_id):
    """
    Align the ground truth pose and the predicted pose in the output of NOCS implementation.
    Since the predicted pose idx are not same as the ground truth pose in the output of NOCS, alignment has to be done to read in the ground truth pose of the corresponding predicted pose
    Args:
        pred_data: the dictionary which stores the output information of NOCS
        inst_id: the instance id of the object in the whole scene

    Returns:
        the aligned id of the pose, i.e. the id of the predicted pose of the instance
    """

    cls = pred_data['gt_class_ids'][inst_id - 1]
    gt_bbx = pred_data['gt_bboxes'][inst_id - 1]
    center_x = (gt_bbx[0] + gt_bbx[2]) / 2
    center_y = (gt_bbx[1] + gt_bbx[3]) / 2
    pred_id_candidates = list_duplicates_of(pred_data['pred_class_ids'].tolist(), cls)
    if len(pred_id_candidates) == 1:
        aligned_id = pred_id_candidates[0]
    elif len(pred_id_candidates) == 0:
        dist = []
        for i in range(len(pred_data['pred_class_ids'])):
            pred_bbx = pred_data['pred_bboxes'][i]
            center_x_pred = (pred_bbx[0] + pred_bbx[2]) / 2
            center_y_pred = (pred_bbx[1] + pred_bbx[3]) / 2
            dist.append(np.sqrt((center_x_pred - center_x) ** 2 + (center_y_pred - center_y) ** 2))
        dist_min = min(dist)
        if dist_min > 16 and len(pred_data['pred_class_ids']) < len(pred_data['gt_class_ids']):
            return False
        else:
            aligned_id = dist.index(min(dist))
    else:
        dist = []
        for id_can in pred_id_candidates:
            pred_bbx = pred_data['pred_bboxes'][id_can]
            center_x_pred = (pred_bbx[0] + pred_bbx[2]) / 2
            center_y_pred = (pred_bbx[1] + pred_bbx[3]) / 2
            dist.append(np.sqrt((center_x_pred - center_x) ** 2 + (center_y_pred - center_y) ** 2))
        aligned_id = pred_id_candidates[dist.index(min(dist))]

    center_x_pred = (pred_data['pred_bboxes'][aligned_id][0] + pred_data['pred_bboxes'][aligned_id][2]) / 2
    center_y_pred = (pred_data['pred_bboxes'][aligned_id][1] + pred_data['pred_bboxes'][aligned_id][3]) / 2
    if np.sqrt((center_x_pred - center_x) ** 2 + (center_y_pred - center_y) ** 2) >= 20:
        return False
    else:
        return aligned_id


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


# T_poses num*3
# r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch = r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch, 1, 3, 3).expand(batch, joint_num, 3, 3).contiguous().view(batch * joint_num, 3, 3)
    src_poses = T_pose.view(1, joint_num, 3, 1).expand(batch, joint_num, 3, 1).contiguous().view(batch * joint_num, 3,
                                                                                                 1)

    out_poses = torch.matmul(r_matrices, src_poses)  # (batch*joint_num)*3*1

    return out_poses.view(batch, joint_num, 3)


# in batch*6
# out batch*5
def stereographic_project(a):
    dim = a.shape[1]
    a = normalize_vector(a)
    out = a[:, 0:dim - 1] / (1 - a[:, dim - 1])
    return out


# in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)  # batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1] + 1).cuda())  # batch*6
    unproj = 2 * a / (s2 + 1).view(batch, 1).repeat(1, a.shape[1])  # batch*5
    if (axis > 0):
        ans[:, :axis] = unproj[:, :axis]  # batch*(axis-0)
    ans[:, axis] = (s2 - 1) / (s2 + 1)  # batch
    ans[:, axis + 1:] = unproj[:,
                        axis:]  # batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans


# a batch*5
# out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  # 3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1, 3).repeat(batch, 1)  # batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  # batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  # batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  # batch*4
    b = torch.cat((a[:, 0:2], u), 1)  # batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


# quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle(axisAngle):
    batch = axisAngle.shape[0]

    theta = torch.tanh(axisAngle[:, 0]) * np.pi  # [-180, 180]
    sin = torch.sin(theta * 0.5)
    axis = normalize_vector(axisAngle[:, 1:4])  # batch*3
    qw = torch.cos(theta * 0.5)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*3 (x,y,z)*theta
def compute_rotation_matrix_from_Rodriguez(rod):
    batch = rod.shape[0]

    axis, theta = normalize_vector(rod, return_mag=True)

    sin = torch.sin(theta)

    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf(hopf):
    batch = hopf.shape[0]

    theta = (torch.tanh(hopf[:, 0]) + 1.0) * np.pi / 2.0  # [0, pi]
    phi = (torch.tanh(hopf[:, 1]) + 1.0) * np.pi  # [0,2pi)
    tao = (torch.tanh(hopf[:, 2]) + 1.0) * np.pi  # [0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# euler_sin_cos batch*6
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch = euler_sin_cos.shape[0]

    s1 = euler_sin_cos[:, 0].view(batch, 1)
    c1 = euler_sin_cos[:, 1].view(batch, 1)
    s2 = euler_sin_cos[:, 2].view(batch, 1)
    c2 = euler_sin_cos[:, 3].view(batch, 1)
    s3 = euler_sin_cos[:, 4].view(batch, 1)
    c3 = euler_sin_cos[:, 5].view(batch, 1)

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta


def get_sampled_rotation_matrices_by_quat(batch):
    # quat = torch.autograd.Variable(torch.rand(batch,4).cuda())
    quat = torch.autograd.Variable(torch.randn(batch, 4).cuda())
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix


def get_sampled_rotation_matrices_by_hpof(batch):
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0, 1, batch) * np.pi).cuda())  # [0, pi]
    phi = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).cuda())  # [0,2pi)
    tao = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).cuda())  # [0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*4 angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle(batch, return_quaternion=False):
    theta = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(-1, 1, batch) * np.pi).cuda())  # [0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    axis = normalize_vector(axis)  # batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    quaternion = torch.cat((qw.view(batch, 1), qx.view(batch, 1), qy.view(batch, 1), qz.view(batch, 1)), 1)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    if (return_quaternion == True):
        return matrix, quaternion
    else:
        return matrix
