import os
import sys
from os.path import join as pjoin
import json
import numpy as np
from argparse import ArgumentParser
import time

sys.path.append('./utils')
from utils.render_utils import set_all_scene, render_rgb_image, render_depth_map, get_semantic_info, get_link_pose_dict, get_joint_pose_dict, get_mapping_dict, get_recovery_point_cloud_per_part, get_pose_bbox, add_background_color_for_image, get_2D_segmentation, get_2D_3D_bbox, get_NPCS_map, get_final_anno_bbox_pose, get_cam_pos, get_camera_rotate, get_light_decay, check_object_in_image
from utils.urdf_utils import get_urdf_mobility
from utils.config_utils import CAMERA_ROTATE, get_fixed_handle_configs, get_id_label, save_rgb_image, save_depth_map, save_anno_dict, save_meta, DATASET_PATH, FIXED_HANDLE_STRATEGY, TARGET_PARTS_FIRST_STAGE, TARGET_PARTS_SECOND_STAGE, BACKGROUND_RGB, RERENDER_MAX, SAVE_PATH
from utils.visu_utils import visu_3D_bbox_semantic, visu_3D_bbox_pose_in_color, save_image

if __name__ == "__main__":
    # 加载配置参数
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=int)
    parser.add_argument('--category', type=str)
    parser.add_argument('--render_index', type=int)
    parser.add_argument('--camera_position_index', type=int)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--dis_min', type=float, default=2.8)
    parser.add_argument('--dis_max', type=float, default=4.1)
    parser.add_argument('--theta_min', type=float, default=30.0)
    parser.add_argument('--theta_max', type=float, default=90.0)
    parser.add_argument('--phi_min', type=float, default=120.0)
    parser.add_argument('--phi_max', type=float, default=240.0)

    CONFS = parser.parse_args()

    MODEL_ID = CONFS.model_id
    CATEGORY = CONFS.category
    RENDER_INDEX = CONFS.render_index
    CAMERA_POSITION_INDEX = CONFS.camera_position_index
    WIDTH = CONFS.width
    HEIGHT = CONFS.height
    DIS_MIN = CONFS.dis_min
    DIS_MAX = CONFS.dis_max
    THETA_MIN = CONFS.theta_min
    THETA_MAX = CONFS.theta_max
    PHI_MIN = CONFS.phi_min
    PHI_MAX = CONFS.phi_max

    # 配置路径
    anno_data_path = pjoin(DATASET_PATH, CATEGORY.lower(), str(MODEL_ID))

    save_path = SAVE_PATH
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    rgb_save_path = pjoin(save_path, 'image_vulkan')
    if not os.path.exists(rgb_save_path):
        os.mkdir(rgb_save_path)
    depth_save_path = pjoin(save_path, 'depth')
    if not os.path.exists(depth_save_path):
        os.mkdir(depth_save_path)
    anno_save_path = pjoin(save_path, 'annotation')
    if not os.path.exists(anno_save_path):
        os.mkdir(anno_save_path)
    meta_save_path = pjoin(save_path, 'metafile')
    if not os.path.exists(meta_save_path):
        os.mkdir(meta_save_path)
    meta_raw_save_path = pjoin(save_path, 'metafile_raw')
    if not os.path.exists(meta_raw_save_path):
        os.mkdir(meta_raw_save_path)

    seed = int(time.time())
    np.random.seed(seed)
    
    init_dict = None
    if "init_pose.json" in os.listdir(anno_data_path):
        with open(pjoin(anno_data_path, "init_pose.json"), 'r') as fd:
            init_dict = json.load(fd)

    engine = None

    rerender_cnt = 0
    print(f'------> Current Render Stage 1: {CATEGORY} : {MODEL_ID} : {CAMERA_POSITION_INDEX} : {RENDER_INDEX}')

    # render RGB image and depth map
    valid_flag = False  # check if object instance is all in image
    while not valid_flag:
        cam_pos = get_cam_pos(theta_min=THETA_MIN,
                              theta_max=THETA_MAX,
                              phi_min=PHI_MIN,
                              phi_max=PHI_MAX,
                              dis_min=DIS_MIN,
                              dis_max=DIS_MAX)

        # debug
        # cam_pos=np.array([-2.5,2.5,2.5])

        light_decay = get_light_decay()
        cam_rot = get_camera_rotate()
        
        scene, camera, joint_qpos, metafile_raw, engine = set_all_scene(data_path=anno_data_path,
                                                                    cam_pos=cam_pos,
                                                                    width=WIDTH,
                                                                    height=HEIGHT,
                                                                    model_id=MODEL_ID,
                                                                    category=CATEGORY,
                                                                    cam_pos_index=CAMERA_POSITION_INDEX,
                                                                    light_decay=light_decay,
                                                                    camera_rotate=cam_rot,
                                                                    render_index=RENDER_INDEX,
                                                                    engine=engine)
        rgb_image = render_rgb_image(camera=camera)
        depth_map = render_depth_map(camera=camera)

        # 检查object是否超出画面，如果超出，则重新render
        valid_flag = check_object_in_image(depth_map, metafile_raw)
        if not valid_flag:
            if rerender_cnt >= RERENDER_MAX:
                print(
                    f'Error! Fail in first stage in rendering {CATEGORY} : {MODEL_ID} : {CAMERA_POSITION_INDEX} : {RENDER_INDEX}'
                )
                exit(-1)
            rerender_cnt += 1
            print(f'------>Rerender Stage 1: {CATEGORY} : {MODEL_ID} : {CAMERA_POSITION_INDEX} : {RENDER_INDEX}')
            continue
        else:
            final_rgb_image = rgb_image
            final_depth_map = depth_map
            break

    # render annotation
    scene, camera, joint_qpos, metafile_raw, engine = set_all_scene(data_path=anno_data_path,
                                                                cam_pos=cam_pos,
                                                                width=WIDTH,
                                                                height=HEIGHT,
                                                                model_id=MODEL_ID,
                                                                category=CATEGORY,
                                                                render_index=RENDER_INDEX,
                                                                cam_pos_index=CAMERA_POSITION_INDEX,
                                                                light_decay=light_decay,
                                                                camera_rotate=cam_rot,
                                                                joint_qpos=joint_qpos,
                                                                engine=engine)
    rgb_image = render_rgb_image(camera=camera)
    depth_map = render_depth_map(camera=camera)

    # 二次进行边界检查，如果超出，直接exit
    valid_flag = check_object_in_image(depth_map, metafile_raw)
    if not valid_flag:
        print(
            'Error! Fail in second stage in rendering {CATEGORY} : {MODEL_ID} : {CAMERA_POSITION_INDEX} : {RENDER_INDEX}'
        )
        exit(-1)
        
    link_name_to_semantics_dict, link_name_list = get_semantic_info(anno_data_path)
    urdf_ins, base_rot_mat = get_urdf_mobility(anno_data_path, link_name_list)
    metafile = metafile_raw
    link_pose_dict = get_link_pose_dict(scene=scene)
    joint_pose_dict = get_joint_pose_dict(link_pose_dict=link_pose_dict, urdf_ins=urdf_ins)
    visId2instName, instName2catId = get_mapping_dict(scene=scene,
                                                    linkname2catName=link_name_to_semantics_dict,
                                                    target_parts_list=TARGET_PARTS_SECOND_STAGE)
    part_list, point_cloud, per_point_rgb = get_recovery_point_cloud_per_part(camera=camera,
                                                                            rgb_image=rgb_image,
                                                                            depth_map=depth_map,
                                                                            meta=metafile,
                                                                            visId2instName=visId2instName,
                                                                            instName2catId=instName2catId)
    part_bbox_list = get_pose_bbox(part_list=part_list,
                                    target_parts_list=TARGET_PARTS_SECOND_STAGE,
                                    final_target_parts_list=TARGET_PARTS_SECOND_STAGE,
                                    scene=scene,
                                    urdf_ins=urdf_ins,
                                    link_pose_dict=link_pose_dict,
                                    linkname_list=link_name_list,
                                    joint_pose_dict=joint_pose_dict,
                                    fixed_handle_config=None,
                                    base_rot_mat = base_rot_mat,
                                    init_dict=init_dict,
                                    meta=metafile)

    final_rgb_image = add_background_color_for_image(final_rgb_image, final_depth_map, metafile, BACKGROUND_RGB)
    sem_seg_map, ins_seg_map, anno_mapping = get_2D_segmentation(camera=camera,
                                                                depth_map=final_depth_map,
                                                                part_bbox_list=part_bbox_list,
                                                                fixed_handle_anno_flag=None,
                                                                meta=metafile,
                                                                visId2instName=visId2instName,
                                                                instName2catId=instName2catId)
    bbox_2D_3D_list = get_2D_3D_bbox(sem_seg_map, ins_seg_map, anno_mapping)
    npcs_map = get_NPCS_map(final_depth_map, ins_seg_map, anno_mapping, metafile)
    anno_bbox_pose_list = get_final_anno_bbox_pose(bbox_2D_3D_list, TARGET_PARTS_SECOND_STAGE)

    anno_dict = {
        'semantic_segmentation': sem_seg_map,
        'instance_segmentation': ins_seg_map,
        'npcs_map': npcs_map,
        'bboxes_with_pose': anno_bbox_pose_list
    }

    # 保存RGB、depth、annotation、metafile
    save_name = '{}_{}_{:02d}_{:03d}'.format(CATEGORY, MODEL_ID, CAMERA_POSITION_INDEX, RENDER_INDEX)
    save_rgb_image(final_rgb_image, rgb_save_path, save_name)
    save_depth_map(final_depth_map, depth_save_path, save_name)
    save_anno_dict(anno_dict, anno_save_path, save_name)
    save_meta(metafile, meta_save_path, save_name)
    save_meta(metafile_raw, meta_raw_save_path, save_name)

    # 结束
    print(f'------> Render Over Stage 1: {CATEGORY} : {MODEL_ID} : {CAMERA_POSITION_INDEX} : {RENDER_INDEX}\n')
