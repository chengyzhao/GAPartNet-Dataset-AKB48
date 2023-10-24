import os
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import transforms3d.axangles as t
import math

# TODO 超参，用来调整tight bbox在每个方向的缩放倍数，以容纳point cloud的微小错位（原因未知）
EPSILON_L = 0.001
EPSILON_W = 0.001
EPSILON_H = 0.001


def draw_line(x, y, z, vector):
    points = []
    list = [(0 + i * 0.001) for i in range(3000)]
    for i in list:
        point = [x + vector[0] * i, y + vector[1] * i, z + vector[2] * i]
        points.append(point)
    return np.array(points)


def draw_bbox(bbox, category_id):
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


def draw_bbox_with_pose(bbox, category_id):
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


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def load_model_split(inpath):
    vsplit = []
    fsplit = []
    dict_mesh = {}
    list_group = []
    list_xyz = []
    list_face = []
    with open(inpath, "r", errors='replace') as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if len(line) < 2:
                line = fp.readline()
                cnt += 1
                continue
            xyz = []
            face = []
            if line[0] == 'g':
                list_group.append(line[2:])
            if line[0:2] == 'v ':
                vcount = 0
                while line[0:2] == 'v ':
                    xyz.append([float(coord) for coord in line[2:].strip().split()])
                    vcount += 1
                    line = fp.readline()
                    cnt += 1
                vsplit.append(vcount)
                list_xyz.append(xyz)

            if line[0] == 'f':
                fcount = 0
                while line[0] == 'f':
                    face.append([num for num in line[2:].strip().split()])
                    fcount += 1
                    line = fp.readline()
                    cnt += 1
                    if not line:
                        break
                fsplit.append(fcount)
                list_face.append(face)
            line = fp.readline()
            cnt += 1
    dict_mesh['v'] = list_xyz
    dict_mesh['f'] = list_face

    return dict_mesh, list_group, vsplit, fsplit


def get_all_objs(obj_file_list):
    pts_list = []  # for each link, a list of vertices
    name_list = []  # for each link, the .obj filenames
    offset = 0

    def read_obj_file(obj_file):
        try:
            tm = trimesh.load(obj_file)
            vertices_obj = np.array(tm.vertices)
        except:
            dict_mesh, _, _, _ = load_model_split(obj_file)
            vertices_obj = np.concatenate(dict_mesh['v'], axis=0)
        return vertices_obj

    for k, obj_files in enumerate(obj_file_list):  # each corresponds to a link
        cur_list = None
        if isinstance(obj_files, list):
            cur_list = obj_files
        elif obj_files is not None:
            cur_list = [obj_files]
        # collect all names & vertices
        part_pts = []
        name_objs = []
        for obj_file in cur_list:
            if obj_file is not None and not isinstance(obj_file, list):
                vertices_obj = read_obj_file(obj_file)
                part_pts.append(vertices_obj)
                name_obj = obj_file.split('.')[0].split('/')[-1]
                name_objs.append(name_obj)

        part_pts_all = np.concatenate(part_pts, axis=0)
        pts_list.append(part_pts_all + offset)
        name_list.append(name_objs)  # which should follow the right

    # vertices: a list of sublists,
    # sublists contain vertices in the whole shape (0) and in each part (1, 2, ..)
    vertices = [pts_list]
    for part in pts_list:
        vertices.append([part])

    norm_factors = []  # for each link, a float
    corner_pts = []
    # calculate bbox & corners for the whole object
    # as well as each part
    for j in range(len(vertices)):

        part_verts = np.concatenate(vertices[j], axis=0)  # merge sublists
        pmax, pmin = np.amax(part_verts, axis=0), np.amin(part_verts, axis=0)
        corner_pts.append([pmin, pmax])  # [index][left/right][x, y, z], numpy array
        norm_factor = np.sqrt(1) / np.sqrt(np.sum((pmax - pmin)**2))
        norm_factors.append(norm_factor)

    return vertices[1:], norm_factors, corner_pts


def get_model_pts(obj_file_list, is_debug=False):
    """
    For item obj_category/item,
    get_urdf(_mobility) returns obj_file_list:
        [[objs of 0th link], [objs of 1st link], ...]
    This function reads these obj files,
    and calculates the following for each link (part) --- except the first link (base link)
    - model parts,
    - norm_factors, 1 / diagonal of bbox
    - corner_pts, the corners of the bbox

    """
    if obj_file_list is not None and obj_file_list[0] == []:
        if is_debug:
            print('removing the first obj list, which corresponds to base')
        obj_file_list = obj_file_list[1:]

    model_pts, norm_factors, corner_pts = get_all_objs(obj_file_list=obj_file_list)
    return model_pts, norm_factors, corner_pts


def get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s):
    l = l_s / 2
    w = w_s / 2
    h = h_s / 2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] += x_center
    corners_3d[1, :] += y_center
    corners_3d[2, :] += z_center
    return np.transpose(corners_3d)


def get_hinge_handle_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, linkname_list, joint_pose_dict, base_rot_mat, meta):
    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    link_id = linkname_list.index(link_name)

    joint_id_list = []
    last_link_name = link_name
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links']):
            if joint_info['child'][i] == last_link_name:
                joint_id_list.append(i)
                if joint_info['parent'][i] == 'root':
                    end_flag = True
                    break
                else:
                    last_link_name = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        # joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1
        joint_qpos = joint_qpos_list[0]

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (base_rot_mat.T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)
        obj2world = t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos).T
        part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point

        # 以axis方向向量为 canon space +x，提起方向为 canon space +z，叉乘得 canon space +y
        canon_in_world_x = axis_direction_vector
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
        rotation_theta = joint_qpos - 0
        canon_in_world_z = np.cross(canon_in_world_x, np.array([0, 0, 1])) * math.sin(rotation_theta) + np.array(
            [0, 0, 1]) * math.cos(rotation_theta)
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
        canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t
        tight_bbox_world_moved = tight_bbox_world
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        # pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        # if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        #     print('part points: ', pcs_world.shape[0])
        #     print('points in box: ', pc_in_bbox.shape[0])
        #     print('Error! tight bbox failed: not all points are in the tight bbox!')
        #     exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    else:
        print('Error! for hinge handle, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)
        
        
def get_slider_drawer_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, linkname_list, joint_pose_dict, base_rot_mat, meta):
    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    link_id = linkname_list.index(link_name)

    joint_id_list = []
    last_link_name = link_name
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links']):
            if joint_info['child'][i] == last_link_name:
                joint_id_list.append(i)
                if joint_info['parent'][i] == 'root':
                    end_flag = True
                    break
                else:
                    last_link_name = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        # joint_qpos = joint_qpos_list[joint_id]
        joint_qpos = joint_qpos_list[int(link_name[-1])-1]

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (base_rot_mat.T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)

        # 以axis方向向量为 canon space +x，world space +z为 canon space +z，叉乘得 canon space +y
        canon_in_world_x = axis_direction_vector
        if canon_in_world_x[0][0] > 0:
            canon_in_world_x = canon_in_world_x * (-1)
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
        canon_in_world_y = np.cross(np.array([0, 0, 1]), canon_in_world_x)
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
        canon_in_world_z = np.cross(canon_in_world_x, canon_in_world_y)
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t + (axis_direction_vector * joint_qpos)
        tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        if pc_in_bbox.shape[0] != pcs_world.shape[0]:
            print('part points: ', pcs_world.shape[0])
            print('points in box: ', pc_in_bbox.shape[0])
            print('Error! tight bbox failed: not all points are in the tight bbox!')
            exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    else:
        print('Error! for slider drawer, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


def get_hinge_lid_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, linkname_list, joint_pose_dict, base_rot_mat, pre_qpos, pre_bbox, meta):
    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    link_id = linkname_list.index(link_name)

    joint_id_list = []
    last_link_name = link_name
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links']):
            if joint_info['child'][i] == last_link_name:
                joint_id_list.append(i)
                if joint_info['parent'][i] == 'root':
                    end_flag = True
                    break
                else:
                    last_link_name = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        # joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1
        joint_qpos = joint_qpos_list[0]

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (base_rot_mat.T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)
        obj2world = t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos).T
        part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point

        # 以axis方向向量为 canon space +x，lid的正面朝向为 canon space +z，叉乘得从转轴指向lid对边的方向为 canon space +y
        # ! 假设转动方向符合右手定则，lid在static状态下正面朝向world space的-x方向
        canon_in_world_x = axis_direction_vector
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
        if pre_qpos is None:
            rotation_theta = joint_qpos - 0
        else:
            rotation_theta = joint_qpos - pre_qpos
        canon_in_world_z = np.cross(canon_in_world_x, np.array([0, 0, 1])) * math.sin(rotation_theta) + np.array(
            [0, 0, 1]) * math.cos(rotation_theta)
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
        canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
        if pre_bbox is not None:
            given_direction = pre_bbox[2] - pre_bbox[3]
            given_direction = given_direction / np.linalg.norm(given_direction)
            if np.sum(axis_direction_vector*given_direction) < 0:
                canon_in_world_x = -canon_in_world_x
                canon_in_world_y = -canon_in_world_y
                canon_in_world_z = np.cross(canon_in_world_x, canon_in_world_y)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t
        tight_bbox_world_moved = tight_bbox_world
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        # pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        # if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        #     print('part points: ', pcs_world.shape[0])
        #     print('points in box: ', pc_in_bbox.shape[0])
        #     print('Error! tight bbox failed: not all points are in the tight bbox!')
        #     print(_.shape[0])
        #     print(pcs_world.shape[0])
        #     error_point = 0
        #     for i in range(_.shape[0]):
        #         if not _[i]:
        #             error_point = i
        #     print(pcs_world[error_point])
        #     print(pcs_original_canon[error_point] @ canon2world)
        #     print(world2canon @ canon2world)
        #     exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    else:
        print('Error! for hinge lid, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)

