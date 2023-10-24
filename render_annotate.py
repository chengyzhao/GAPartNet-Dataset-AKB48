import os
import sys
from os.path import join as pjoin
import numpy as np
from argparse import ArgumentParser
import time
import shutil
import copy
import json


sys.path.append('./utils')
from utils.render_utils import get_cam_pos, set_all_scene, render_rgb_image, render_depth_map, get_semantic_info, get_mapping_dict, get_recovery_point_cloud_per_part, get_pose_bbox, get_link_pose_dict, get_joint_pose_dict, add_background_color_for_image, check_object_in_image, get_2D_segmentation, get_2D_3D_bbox, get_NPCS_map, get_final_anno_bbox_pose, \
    recover_pose_bbox_to_static_pose, merge_bbox_into_annotation, merge_link_annotation, check_annotation_all_filled
from utils.urdf_utils import get_urdf_mobility, create_link_annos, modify_urdf_file_add_remove, modify_semantic_info, create_link_annos
from utils.config_utils import CAMERA_ROTATE, DATASET_PATH, SAVE_PATH, TARGET_PARTS_SECOND_STAGE, BACKGROUND_RGB, RENDERED_DATA_PATH_1, RENDERED_DATA_PATH_2, \
    load_meta, load_anno_dict, NUM_RENDER, CAMERA_POSITION_RANGE, save_link_annos


if __name__ == "__main__":
    # 加载配置参数
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=int)
    parser.add_argument('--category', type=str)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    CONFS = parser.parse_args()

    MODEL_ID = CONFS.model_id
    CATEGORY = CONFS.category
    WIDTH = CONFS.width
    HEIGHT = CONFS.height
    
    print(f'------> Current Run: {CATEGORY} : {MODEL_ID}')
    
    # 配置路径
    anno_data_path = pjoin(DATASET_PATH, CATEGORY.lower(), str(MODEL_ID))
    
    # copy data to save path
    save_path = SAVE_PATH
    anno_new_path = pjoin(save_path, CATEGORY, str(MODEL_ID))
    if os.path.exists(anno_new_path):
        shutil.rmtree(anno_new_path)
    shutil.copytree(anno_data_path, anno_new_path)
    print('Copy data to save path done!')
    
    anno_data_path = anno_new_path
    
    # load rendered metas and annos
    num_camera_position = len(CAMERA_POSITION_RANGE[CATEGORY])
    num_render = NUM_RENDER
    # anno_history = {}
    meta_history = {}
    for i_cam in range(num_camera_position):
        # anno_history[i_cam] = {}
        meta_history[i_cam] = {}
        for i_render in range(num_render):
            filename = '{}_{}_{:02d}_{:03d}'.format(CATEGORY, MODEL_ID, i_cam, i_render)
            rendered_data_path = RENDERED_DATA_PATH_2 if CATEGORY.lower() == 'drawer' else RENDERED_DATA_PATH_1
            metafile = load_meta(pjoin(rendered_data_path, 'metafile_raw'), filename)
            meta_history[i_cam][i_render] = metafile
            # anno_dict = load_anno_dict(pjoin(RENDERED_DATA_PATH, 'annotation'), filename)
    print('Load history metafile and annotation done!')
    
    engine = None
    all_link_annotations = None
    finished = False
    p_new_urdf_file = None
    p_new_semantic_file = None
    
    init_dict = None
    if "init_pose.json" in os.listdir(anno_data_path):
        with open(pjoin(anno_data_path, "init_pose.json"), 'r') as fd:
            init_dict = json.load(fd)
    
    for i_render in range(num_render):
        for i_cam in range(num_camera_position):
            print(f'------> Current Run: {CATEGORY} : {MODEL_ID} : {i_cam} : {i_render}')
            
            meta_his = meta_history[i_cam][i_render]
            cam_pos_his = np.array(meta_his['camera_position'], dtype=np.float32).reshape(-1)
            joint_qpos_his = np.array(meta_his['joint_qpos'], dtype=np.float32).reshape(-1)
            camera_rotate = None
            # to render some corner case where without camera_rotate, the object is not completely in the image
            # camera_rotate = float(meta_his['camera_rotate'])
            
            # use random camera pose to render corner case: gaparts at back
            # camera_rotate = None
            # camera_range = CAMERA_POSITION_RANGE[CATEGORY][i_cam]
            # cam_pos_his = get_cam_pos(theta_min=camera_range['theta_min'],
            #                   theta_max=camera_range['theta_max'],
            #                   phi_min=camera_range['phi_min'],
            #                   phi_max=camera_range['phi_max'],
            #                   dis_min=camera_range['distance_min'],
            #                   dis_max=camera_range['distance_max'])
            # if np.random.rand() < 0.5:
            #     cam_pos_his[0] = -cam_pos_his[0] # flip x
            
            scene, camera, joint_qpos, metafile_raw, engine, robot = set_all_scene(data_path=anno_data_path,
                                                                    cam_pos=cam_pos_his,
                                                                    width=WIDTH,
                                                                    height=HEIGHT,
                                                                    model_id=MODEL_ID,
                                                                    category=CATEGORY,
                                                                    joint_qpos=joint_qpos_his,
                                                                    camera_rotate=camera_rotate,
                                                                    engine=engine)
            rgb_image = render_rgb_image(camera=camera)
            depth_map = render_depth_map(camera=camera)

            final_rgb_image = rgb_image
            final_depth_map = depth_map

            valid_flag = check_object_in_image(depth_map, metafile_raw)
            if not valid_flag:
                print(
                    f'Error! Fail in boundary check stage in rendering {CATEGORY} : {MODEL_ID} : {i_cam} : {i_render}')
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
            
            # 1. modify urdf file
            p_new_urdf_file = modify_urdf_file_add_remove(anno_data_path)
            print("Save new urdf file done: ", p_new_urdf_file)
            
            # 2. create semantic label
            p_new_semantic_file, all_link_names = modify_semantic_info(anno_data_path)
            print("Save new semantic file done: ", p_new_semantic_file)
            
            # 3. create gapart semantic and pose annotation
            # 3.1 init gapart semantic annotation w/o pose
            link_annotations, instname2linkname = create_link_annos(instName2catId, TARGET_PARTS_SECOND_STAGE, all_link_names)
            # Here, all links are annotated, but line/round fixed handle are not specified, hinge_handle on Door is not fixed (still in hinge_handleline or hinge_handleround)
            # print(f'link_annotations: {link_annotations}')
            
            # 3.2 add pose to gapart annotation
            static_part_pose_bbox_list = recover_pose_bbox_to_static_pose(part_bbox_list, urdf_ins, joint_pose_dict, TARGET_PARTS_SECOND_STAGE, metafile, robot) # (final_cat_id in SECOND_STAGE, bbox at all qpos=0, inst_name)
            # print(f'static_part_pose_bbox_list: {static_part_pose_bbox_list}')
            
            # 3.3 merge to link annotations
            link_annotations = merge_bbox_into_annotation(link_annotations, static_part_pose_bbox_list, instname2linkname)
            # Here, add bbox anno to gaparts, specify line/round fixed handle, hinge_handle on Door is fixed (kinematic joint is not fixed yet)
            # print(f'link_annotations: {link_annotations}')
            
            # 3.4 merge with previous frame
            if all_link_annotations is None:
                all_link_annotations = copy.deepcopy(link_annotations)
            else:
                all_link_annotations = merge_link_annotation(all_link_annotations, copy.deepcopy(link_annotations))
            
            # 3.5 check if all parts are annotated
            finished = check_annotation_all_filled(all_link_annotations, TARGET_PARTS_SECOND_STAGE)
            
            if finished:
                break
            
        if finished:
            break
    
    assert finished, "Error! Not all parts are annotated!"
    assert p_new_urdf_file is not None, "Error! p_new_urdf_file is None!"
    assert p_new_semantic_file is not None, "Error! p_new_semantic_file is None!"
    
    save_link_annos(all_link_annotations, anno_data_path)
    print('Save link annotation done!')
    
    print(f'------> Current Run: {CATEGORY} : {MODEL_ID} done!')

