import os
import sys
from os.path import join as pjoin
from matplotlib.colors import rgb_to_hsv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import open3d as o3d


def save_image(img_array, save_path, filename):
    img = Image.fromarray(img_array)
    img.save(pjoin(save_path, '{}.png'.format(filename)))
    print('{} saved!'.format(filename))


def visu_depth_map(depth_map):
    # 最近的点置为0 最远的点置为255 后applyColorMap
    object_mask = (abs(depth_map) >= 1e-6)
    empty_mask = (abs(depth_map) < 1e-6)
    new_map = depth_map - depth_map[object_mask].min()
    new_map = new_map / new_map.max()
    new_map = np.clip(new_map * 255, 0, 255).astype('uint8')
    colored_depth_map = cv2.applyColorMap(new_map, cv2.COLORMAP_JET)
    colored_depth_map[empty_mask] = np.array([0, 0, 0])
    return colored_depth_map


def visu_2D_seg_map(seg_map):
    H, W = seg_map.shape
    seg_image = np.zeros((H, W, 3)).astype("uint8")

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")

    for y in range(0, H):
        for x in range(0, W):
            if seg_map[y, x] == -1:
                continue
            if seg_map[y, x] == 0:
                seg_image[y, x] = cmap[14]
            else:
                seg_image[y, x] = cmap[int(seg_map[y, x])]

    return seg_image


def visu_2D_bbox(rgb_image, bboxes_with_pose_list):
    image = np.copy(rgb_image)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")

    lines = [[0, 1], [1, 2], [2, 3], [0, 3]]

    for part_dict in bboxes_with_pose_list:
        category_id = part_dict['category_id']
        bbox_2d_coord = part_dict['bbox_2d']
        y_min = bbox_2d_coord[0, 0]
        x_min = bbox_2d_coord[0, 1]
        y_max = bbox_2d_coord[1, 0]
        x_max = bbox_2d_coord[1, 1]
        corners = [(y_min, x_min), (y_max, x_min), (y_max, x_max), (y_min, x_max)]
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = corners[line[0]][1]
            y_start = corners[line[0]][0]
            x_end = corners[line[1]][1]
            y_end = corners[line[1]][0]
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 2
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)

    return image


def visu_3D_bbox_semantic(rgb_image, bboxes_with_pose_list, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for part_dict in bboxes_with_pose_list:
        category_id = part_dict['category_id']
        bbox = part_dict['bbox_3d']
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 2
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_3D_bbox_pose_in_color(rgb_image, bboxes_with_pose_list, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]

    for part_dict in bboxes_with_pose_list:
        bbox = part_dict['bbox_3d']
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        for i, line in enumerate(lines):
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 2
            linetype = 4
            color = tuple(int(x) for x in colors[i])
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_3D_bbox_pose_in_frame(rgb_image, bboxes_with_pose_list, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for part_dict in bboxes_with_pose_list:
        category_id = part_dict['category_id']
        bbox = part_dict['bbox_3d']
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 2
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)

        R, T, S = part_dict['NPCS_RTS']
        center_camera = ((T - Rtilt_trl) @ Rtilt_rot).reshape(3, )
        center_x = int(center_camera[0] * K[0][0] / center_camera[2] + K[0][2])
        center_y = int(center_camera[1] * K[1][1] / center_camera[2] + K[1][2])
        center = (center_x, center_y)

        pose_x, pose_y, pose_z = part_dict['pose_frame']
        pose_frame = np.vstack((pose_x, pose_y, pose_z))
        delta = 0.1
        frame_len = min(S) * (1 + delta)
        pose_frame = pose_frame * frame_len
        frame_end = (pose_frame + T - Rtilt_trl) @ Rtilt_rot
        frame_end_2D = []
        for i in range(3):
            x_end = int(frame_end[i, 0] * K[0][0] / frame_end[i, 2] + K[0][2])
            y_end = int(frame_end[i, 1] * K[1][1] / frame_end[i, 2] + K[1][2])
            end = (x_end, y_end)
            frame_end_2D.append(end)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(3):
            end = frame_end_2D[i]
            thickness = 2
            linetype = 4
            cv2.line(image, center, end, color[i], thickness, linetype)

    return image


def visu_NPCS_map(npcs_map, depth_map):
    height = npcs_map.shape[0]
    width = npcs_map.shape[1]

    npcs_image = npcs_map + np.array([0.5, 0.5, 0.5])
    assert (npcs_image > 0).all(), 'NPCS map error!'
    assert (npcs_image < 1).all(), 'NPCS map error!'
    empty_mask = (abs(depth_map) < 1e-6)
    npcs_image[empty_mask] = np.array([0, 0, 0])
    npcs_image = (np.clip(npcs_image, 0, 1) * 255).astype('uint8')

    return npcs_image


def get_recovery_whole_point_cloud_camera(rgb_image, depth_map, meta):
    height, width = depth_map.shape
    K = meta['camera_intrinsic']
    K = np.array(K).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []

    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < 1e-6:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])

    point_cloud = np.array(point_cloud)
    per_point_rgb = np.array(per_point_rgb)

    return point_cloud, per_point_rgb


def get_recovery_part_point_cloud_camera(rgb_image, depth_map, mask, meta):
    height, width = depth_map.shape
    K = meta['camera_intrinsic']
    K = np.array(K).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []

    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < 1e-6:
                continue
            if not mask[y_][x_]:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])

    point_cloud = np.array(point_cloud)
    per_point_rgb = np.array(per_point_rgb)

    return point_cloud, per_point_rgb


def draw_bbox_in_3D_semantic(bbox, category_id):
    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]

    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [cmap[category_id] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_bbox_in_3D_pose_color(bbox):
    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]

    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visu_point_cloud_with_bbox_semantic(rgb_image, depth_map, bboxes_with_pose_list, meta):

    point_cloud, per_point_rgb = get_recovery_whole_point_cloud_camera(rgb_image, depth_map, meta)
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    point_cloud_world = point_cloud @ Rtilt_rot.T + Rtilt_trl

    vis_list = []
    for part_dict in bboxes_with_pose_list:
        category_id = part_dict['category_id']
        bbox = part_dict['bbox_3d']
        bbox_t = draw_bbox_in_3D_semantic(bbox, category_id)
        vis_list.append(bbox_t)

    pcd = o3d.geometry.PointCloud()
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcd.points = o3d.utility.Vector3dVector(point_cloud_world)
    pcd.colors = o3d.utility.Vector3dVector(per_point_rgb)
    vis_list.append(pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis_list.append(coord_frame)
    o3d.visualization.draw_geometries(vis_list)


def visu_point_cloud_with_bbox_pose_color(rgb_image, depth_map, bboxes_with_pose_list, meta):

    point_cloud, per_point_rgb = get_recovery_whole_point_cloud_camera(rgb_image, depth_map, meta)
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    point_cloud_world = point_cloud @ Rtilt_rot.T + Rtilt_trl

    vis_list = []
    for part_dict in bboxes_with_pose_list:
        category_id = part_dict['category_id']
        bbox = part_dict['bbox_3d']
        bbox_t = draw_bbox_in_3D_pose_color(bbox)
        vis_list.append(bbox_t)

    pcd = o3d.geometry.PointCloud()
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcd.points = o3d.utility.Vector3dVector(point_cloud_world)
    pcd.colors = o3d.utility.Vector3dVector(per_point_rgb)
    vis_list.append(pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis_list.append(coord_frame)
    o3d.visualization.draw_geometries(vis_list)


def visu_NPCS_in_3D_with_bbox_pose_color(rgb_image, depth_map, anno_dict, meta):
    ins_seg_map = anno_dict['instance_segmentation']
    bboxes_with_pose_list = anno_dict['bboxes_with_pose']
    npcs_map = anno_dict['npcs_map']
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)

    for part_dict in bboxes_with_pose_list:
        category_id = part_dict['category_id']
        instance_id = part_dict['instance_id']
        bbox_world = part_dict['bbox_3d']
        mask = ins_seg_map == instance_id
        point_cloud, per_point_rgb = get_recovery_part_point_cloud_camera(rgb_image, depth_map, mask, meta)
        point_cloud_world = point_cloud @ Rtilt_rot.T + Rtilt_trl
        R, T, S = part_dict['NPCS_RTS']
        scaler = math.sqrt(S[0]**2 + S[1]**2 + S[2]**2)
        point_cloud_canon = npcs_map[mask]
        bbox_canon = (bbox_world - T) @ R.T / scaler

        vis_list = []
        vis_list.append(draw_bbox_in_3D_pose_color(bbox_world))
        vis_list.append(draw_bbox_in_3D_pose_color(bbox_canon))

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(point_cloud_world)
        pcd_1.colors = o3d.utility.Vector3dVector(per_point_rgb)
        vis_list.append(pcd_1)
        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(point_cloud_canon)
        pcd_2.colors = o3d.utility.Vector3dVector(per_point_rgb)
        vis_list.append(pcd_2)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        vis_list.append(coord_frame)
        o3d.visualization.draw_geometries(vis_list)
