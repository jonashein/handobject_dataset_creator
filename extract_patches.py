from configargparse import ArgumentParser
import os
import pickle
import numpy as np
import trimesh
import math
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from random import random
import shutil
from tqdm import tqdm

def main(args):
    os.makedirs(os.path.join(args.output, "meta"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "segm"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "overlay"), exist_ok=True)
    obj_model_path = shutil.copy(args.obj_model, args.output)
    obj_model_path = os.path.relpath(obj_model_path, start=args.output)

    patch_size = np.array([args.patch_size, args.patch_size]).astype(np.int)
    target_focal_length = 480.0
    target_intr = np.array([[target_focal_length, 0.0, args.patch_size / 2.0], [0.0, target_focal_length, args.patch_size / 2.0], [0.0, 0.0, 1.0]])
    target_intr_inv = np.linalg.inv(target_intr)

    # load calibration values
    intrinsics = np.loadtxt(args.intrinsics) # (3,3)
    scaling_factor = target_focal_length / intrinsics[0, 0]
    intrinsics[:2, :] = intrinsics[:2, :] * scaling_factor

    # load object labels file
    obj_labels = load_object_labels(args.obj_labels)
    # load hand labels
    hand_labels = load_hand_labels(args.hand_labels)

    # load object 3d center
    obj_verts_center3d = get_model_center(args.obj_model)
    obj_verts_center3d = np.hstack([obj_verts_center3d, np.array([1])])
    obj_mesh = trimesh.load(args.obj_model)
    with open("assets/mano/mano_faces.pkl", "rb") as f:
        mano_faces = pickle.load(f)

    # prepare hand vertices
    hand_verts = hand_labels['mano_verts3d']
    hand_joints = hand_labels['mano_joints3d']

    # iterate over labels
    last_timestamp = -args.min_time_delta
    for timestamp in tqdm(sorted(obj_labels.keys())):
        # check minimal time delta between consecutive frames
        if abs(timestamp - last_timestamp) < args.min_time_delta:
            continue

        obj_pose = obj_labels[timestamp]
        rgb_img = None
        meta = {}
        meta['obj_path'] = obj_model_path
        meta['cam_extr'] = np.eye(4)[:3, :] # (3,4)
        meta['cam_calib'] = target_intr # (3,3)

        color_path = os.path.join(args.frames, "rgb_{}.jpg".format(timestamp))
        if os.path.isfile(color_path):
            rgb_img = Image.open(color_path)
            rgb_img = rgb_img.resize((int(rgb_img.width * scaling_factor), int(rgb_img.height * scaling_factor)))

        # compute 2d vertices and center
        obj_verts_3d = transform(obj_mesh.vertices, obj_pose)
        obj_verts_2d = transform(obj_verts_3d, intrinsics, hom_input=True)
        #obj_model_center2d = np.mean(verts_2d, axis=0)
        obj_model_center2d = (np.min(obj_verts_2d, axis=0) + np.max(obj_verts_2d, axis=0)) / 2.0

        # Add random noise
        rand_radius = args.center_jitter * math.sqrt(random())
        rand_angle = random() * 2.0 * np.pi
        center_noise = np.array([rand_radius * math.cos(rand_angle), rand_radius * math.sin(rand_angle)])
        obj_model_center2d += center_noise

        # Ensure patch is within image boundaries
        w, h = rgb_img.size
        patch_center2d = np.maximum(obj_model_center2d, patch_size / 2.0)
        patch_center2d = np.minimum(patch_center2d, np.array([w, h]) - patch_size / 2.0)
        meta['capture_patch_center2d'] = patch_center2d # Store 2d center of cropped patch in case we want to show results in the full-hd image later
        meta['capture_intrinsics'] = intrinsics # Store original camera intrinsics for the same reason

        # compute affine transformation to compensate for image patch
        # Adjust rotation&translation for delta in intrinsics (namely the different principal point)
        patch_intrinsics = np.array(intrinsics)
        patch_intrinsics[:2, 2] = intrinsics[:2, 2] - patch_center2d + patch_size / 2.0
        Rt = target_intr_inv @ patch_intrinsics @ obj_pose[:3, :]
        meta['affine_transform'] = np.concatenate([Rt, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

        tl_corner = (patch_center2d - patch_size / 2.0).astype(np.int)
        br_corner = tl_corner + patch_size

        visible = np.logical_and(np.all(obj_verts_2d >= tl_corner, axis=0), np.all(obj_verts_2d < br_corner, axis=0))
        visible_factor = np.sum(visible) / visible.shape[0]

        # Skip frame if drill is too truncated
        if visible_factor < (1 - args.max_truncation):
            continue

        # Transform hand (which is relative to the object pose) into current frame
        meta['side'] = hand_labels['mano_side']
        meta['shape'] = hand_labels['mano_shape']

        hand_verts_3d = transform(hand_verts, meta['affine_transform'])
        meta['verts_3d'] = hand_verts_3d
        hand_joints_3d = transform(hand_joints, meta['affine_transform'])
        tips_3d = get_finger_tips(hand_verts_3d, side=meta['side'])
        hand_joints_3d = np.concatenate([hand_joints_3d, tips_3d])
        # Matches to idx in order Root, Thumb, Index, Middle, Ring, Pinky
        # With numbering increasing for each finger from base to tips
        idxs = [
            0,
            13, 14, 15, 16,
            1, 2, 3, 17,
            4, 5, 6, 18,
            10, 11, 12, 19,
            7, 8, 9, 20
        ]
        hand_joints_3d = hand_joints_3d[idxs]
        meta['coords_3d'] = hand_joints_3d

        # crop and store rgb patch
        if rgb_img is not None:
            rgb_img_mat = np.array(rgb_img)
            rgb_patch_mat = rgb_img_mat[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0], :]
            rgb_patch = Image.fromarray(rgb_patch_mat)
            rgb_patch.save(os.path.join(args.output, args.split, "rgb", "{}{}.jpg".format(args.out_prefix, timestamp)))

        # recompute 2d vertices using the stored values
        obj_verts_3d = transform(obj_mesh.vertices, meta['affine_transform'])
        obj_verts_2d = transform(obj_verts_3d, meta['cam_calib'], hom_input=True)
        hand_verts_3d = transform(hand_verts, meta['affine_transform'])
        hand_verts_2d = transform(hand_verts_3d, meta['cam_calib'], hom_input=True)
        #hand_joints_3d = transform(hand_joints, meta['affine_transform'])
        #hand_joints_2d = transform(hand_joints_3d, meta['cam_calib'], hom_input=True)

        # generate object segmentation mask
        obj_mask = Image.new("L", rgb_patch.size)
        obj_maskd = ImageDraw.Draw(obj_mask)
        for f in obj_mesh.faces:
            poly = [tuple(obj_verts_2d[i, :]) for i in f]
            obj_maskd.polygon(poly, fill=255)
        # generate hand segmentation mask
        hand_mask = Image.new("L", rgb_patch.size)
        hand_maskd = ImageDraw.Draw(hand_mask)
        for f in mano_faces:
            poly = [tuple(hand_verts_2d[i, :]) for i in f]
            hand_maskd.polygon(poly, fill=255)
        # combine hand and object mask
        merged_mask = Image.merge("RGB", (obj_mask, hand_mask, Image.new("L", rgb_patch.size)))
        merged_mask.save(os.path.join(args.output, args.split, "segm", "{}{}.png".format(args.out_prefix, timestamp)))

        meta_path = os.path.join(args.output, args.split, "meta", "{}{}.pkl".format(args.out_prefix, timestamp))
        pickle.dump(meta, open(meta_path, "wb"))

        last_timestamp = timestamp

        if args.save_overlay or args.show_patch or args.show_full:
            obj_mask_mat = np.array(obj_mask)
            hand_mask_mat = np.array(hand_mask)

            rgb_patch_mat[:, :, 0] = np.maximum(0.5 * rgb_patch_mat[:, :, 0], obj_mask_mat)
            rgb_patch_mat[:, :, 1] = 0.5 * rgb_patch_mat[:, :, 1]
            rgb_patch_mat[:, :, 2] = np.maximum(0.5 * rgb_patch_mat[:, :, 2], hand_mask_mat)
            overlay_patch = Image.fromarray(rgb_patch_mat)
            if args.save_overlay:
                overlay_patch.save(os.path.join(args.output, args.split, "overlay", "{}{}.png".format(args.out_prefix, timestamp)))

            # show drill overlay on patch
            if args.show_patch:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.subplots()
                ax.axis("off")
                ax.imshow(rgb_patch_mat)
                ax.scatter(hand_verts_2d[:, 0], hand_verts_2d[:, 1], c="b", s=1, alpha=0.2)
                ax.scatter(obj_verts_2d[:, 0], obj_verts_2d[:, 1], c="r", s=1, alpha=0.02)
                plt.show()

            # show drill overlay on full image
            if args.show_full:
                obj_verts_3d = transform(obj_mesh.vertices, obj_pose)
                obj_verts_2d = transform(obj_verts_3d, intrinsics, hom_input=True)
                hand_verts_3d = transform(hand_verts, obj_pose)
                hand_verts_2d = transform(hand_verts_3d, intrinsics, hom_input=True)

                rgb_img_mat[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0], :] = rgb_patch_mat
                fig = plt.figure(figsize=(10, 10))
                ax = fig.subplots()
                ax.axis("off")
                ax.imshow(rgb_img_mat)
                ax.scatter(hand_verts_2d[:, 0], hand_verts_2d[:, 1], c="b", s=1, alpha=0.2)
                ax.scatter(obj_verts_2d[:, 0], obj_verts_2d[:, 1], c="r", s=1, alpha=0.02)
                plt.show()


def transform(points, transformation, hom_input=False):
    # if hom_input is True, we assume that the input points are in homogeneous format
    if hom_input:
        points_hom = points.transpose() # (4, N)
    else:
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))]).transpose()  # (4, N)
    assert transformation.shape[1] == points_hom.shape[0], "Invalid transform inputs: {} @ {}".format(transformation.shape, points_hom.shape)
    points_dim = points_hom.shape[0] - 1
    points_trans = transformation @ points_hom  # (4, N)
    points_trans = points_trans[:points_dim, :] / points_trans[points_dim:, :] # (3, N)
    return points_trans.transpose() # (N, 3)


def load_object_labels(label_path):
    labels = {}
    # Read files
    with open(label_path, "r") as f:
        lines = f.readlines()
    # Parse lines
    for i in range(0, len(lines), 5):
        timestamp = int(lines[i])
        pose = np.eye(4)
        for j in range(0, 3):
            pose[j, :] = np.array(lines[i+j+1].split()).astype(np.float32)
        labels[timestamp] = pose
    return labels


def load_hand_labels(label_path):
    try:
        with open(label_path, "rb") as f:
            meta = pickle.load(f)
    except:
        print("Error: Could not load hand labels from file {}".format(label_path))
    return meta


def get_model_center(model_path):
    mesh = trimesh.load(model_path)
    verts = mesh.vertices
    #min = np.amin(verts, axis=0)
    #max = np.amax(verts, axis=0)
    #center3d = (min + max) / 2.0
    center3d = np.mean(verts, axis=0)
    return center3d

def get_finger_tips(verts, side='right'):
    # In order Thumb, Index, Middle, Ring, Pinky
    right_tip_idxs = [745, 317, 444, 556, 673]
    left_tip_idxs = [745, 317, 445, 556, 673]
    if side == 'right':
        tip_idxs = right_tip_idxs
    elif side == 'left':
        tip_idxs = left_tip_idxs
    else:
        raise ValueError('Side sould be in [right, left], got {}'.format(side))

    tips = verts[np.array(tip_idxs)]
    return tips


if __name__ == "__main__":
    parser = ArgumentParser(description="Patch extraction tool")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--frames", required=True, help="Path to rgb frames")
    parser.add_argument("--obj_model", required=True, help="Path to object 3d model")
    parser.add_argument("--obj_labels", required=True, help="Path to ground truth per-frame object labels")
    parser.add_argument("--hand_labels", required=True, help="Path to relative hand pose parameters")
    parser.add_argument("--extrinsics", required=True, help="Path to color sensor extrinsics")
    parser.add_argument("--intrinsics", required=True, help="Path to color sensor intrinsics")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the extracted (square) patches")
    parser.add_argument("--center_jitter", type=int, default=128, help="Center jittering radius in pixels")
    parser.add_argument("--min_time_delta", type=int, default=100000, help="Minimal time delta between two frames in us")
    parser.add_argument("--max_truncation", type=float, default=0.6, help="Maximum fraction of object vertices that are allowed to be truncated")
    parser.add_argument("--out_prefix", default="", help="Prefix for the generated sample files")
    parser.add_argument("--show_patch", action="store_true", help="Display the patches with ground truth overlay")
    parser.add_argument("--show_full", action="store_true", help="Display the full-hd image with ground truth overlay")
    parser.add_argument("--save_overlay", action="store_true", help="Save the patch overlay for further inspection")
    parser.add_argument("--output", required=True, help="Output dataset root directory")

    args = parser.parse_args()
    print("Parameters:")
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print("{}: {}".format(key, val))

    main(args)
