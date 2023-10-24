import os
from os.path import join as pjoin
import xml.etree.ElementTree as ET
import transforms3d.euler as t


# get all info from urdf
def get_urdf_mobility(inpath, linkname_list, verbose=False):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/motion_sapien_v2.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])

    urdf_ins = {}
    tree_urdf = ET.parse(urdf_name)
    num_real_links = len(tree_urdf.findall('link'))
    assert num_real_links == len(linkname_list)
    root_urdf = tree_urdf.getroot()
    rpy_xyz = {}
    list_xyz = [None] * num_real_links
    list_rpy = [None] * num_real_links
    list_obj = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    for link in root_urdf.iter('link'):
        index_link = linkname_list.index(link.attrib['name'])
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[index_link].append([float(x) for x in origin.attrib['xyz'].split()])
                else:
                    list_xyz[index_link].append([0, 0, 0])
                if 'rpy' in origin.attrib:
                    list_rpy[index_link].append([float(x) for x in origin.attrib['rpy'].split()])
                else:
                    list_rpy[index_link].append([0, 0, 0])
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                        list_obj[index_link].append(mesh.attrib['filename'])
                    else:
                        list_obj[index_link].append(inpath + '/' + mesh.attrib['filename'].split('//')[-1].split('.')[0]+'.obj')

    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy  # here it is empty list
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj
    
    base_rotation_mat = None

    rpy_xyz = {}
    list_type = [None] * num_real_links
    list_parent = [None] * num_real_links
    list_child = [None] * num_real_links
    list_xyz = [None] * num_real_links
    list_rpy = [None] * num_real_links
    list_axis = [None] * num_real_links
    list_limit = [[0, 0]] * num_real_links
    list_name = [None] * num_real_links
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        for child in joint.iter('child'):
            link_name = child.attrib['link']
            link_index = linkname_list.index(link_name)
            joint_index = link_index
            list_child[joint_index] = link_name

        list_type[joint_index] = joint.attrib['type']
        list_name[joint_index] = joint.attrib['name']

        for parent in joint.iter('parent'):
            link_name = parent.attrib['link']
            link_index = linkname_list.index(link_name)
            list_parent[joint_index] = link_name

        has_origin = False
        for origin in joint.iter('origin'):
            has_origin = True
            if 'xyz' in origin.attrib:
                list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                list_xyz[joint_index] = [0, 0, 0]
            if 'rpy' in origin.attrib:
                list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                list_rpy[joint_index] = [0, 0, 0]
        if not has_origin:
            list_xyz[joint_index] = [0, 0, 0]
            list_rpy[joint_index] = [0, 0, 0]
            
        for axis in joint.iter('axis'):  # we must have
            list_axis[joint_index] = [float(x) for x in axis.attrib['xyz'].split()]
        for limit in joint.iter('limit'):
            list_limit[joint_index] = [float(limit.attrib['lower']), float(limit.attrib['upper'])]
        # 特殊处理continuous的上下限，和render_utils的处理保持一致
        if joint.attrib['type'] == 'continuous':
            list_limit[joint_index] = [-10000.0, 10000.0]
            
        if list_parent[joint_index] == "root":
            base_rotation_mat = t.euler2mat(list_rpy[joint_index][0], list_rpy[joint_index][1], list_rpy[joint_index][2])
    
    assert base_rotation_mat is not None

    rpy_xyz['name'] = list_name
    rpy_xyz['type'] = list_type
    rpy_xyz['parent'] = list_parent
    rpy_xyz['child'] = list_child
    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy
    rpy_xyz['axis'] = list_axis
    rpy_xyz['limit'] = list_limit

    urdf_ins['joint'] = rpy_xyz
    urdf_ins['num_links'] = num_real_links
    if verbose:
        for j, pos in enumerate(urdf_ins['link']['xyz']):
            if len(pos) > 3:
                print('link {} xyz: '.format(linkname_list[j]), pos[0])
            else:
                print('link {} xyz: '.format(linkname_list[j]), pos)
        for j, orient in enumerate(urdf_ins['link']['rpy']):
            if len(orient) > 3:
                print('link {} rpy: '.format(linkname_list[j]), orient[0])
            else:
                print('link {} rpy: '.format(linkname_list[j]), orient)
        # for joint
        for j, pos in enumerate(urdf_ins['joint']['xyz']):
            print('joint {} xyz: '.format(urdf_ins['joint']['name'][j]), pos)
        for j, orient in enumerate(urdf_ins['joint']['rpy']):
            print('joint {} rpy: '.format(urdf_ins['joint']['name'][j]), orient)
        for j, orient in enumerate(urdf_ins['joint']['axis']):
            print('joint {} axis: '.format(urdf_ins['joint']['name'][j]), orient)
        for j, child in enumerate(urdf_ins['joint']['child']):
            print('joint {} has child link: '.format(urdf_ins['joint']['name'][j]), child)
        for j, parent in enumerate(urdf_ins['joint']['parent']):
            print('joint {} has parent link: '.format(urdf_ins['joint']['name'][j]), parent)

    return urdf_ins, base_rotation_mat


def modify_urdf_file_add_remove(inpath):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/motion_sapien_v2.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])
    
    with open(urdf_name, 'r') as f:
        urdf_lines = f.readlines()
    
    # write new urdf file
    new_urdf_name = pjoin(inpath, 'mobility_relabel_gapartnet.urdf')
    if os.path.exists(new_urdf_name):
        os.remove(new_urdf_name)
    with open(new_urdf_name, 'w') as f:
        f.writelines(urdf_lines)
    
    return new_urdf_name


def modify_semantic_info(in_path):
    semantics_path = os.path.join(in_path, 'semantics.txt')
    semantics = []
    with open(semantics_path, 'r') as fd:
        for line in fd:
            ls = line.strip().split(' ')
            if ls[0] == 'root':
                assert ls[1] == 'fixed', 'root link should be fixed'
                assert ls[2] == 'root', 'root link should be root'
                continue
            if ls[1] == 'revolute' and ls[2] == 'handle':
                ls[1] = 'hinge'
            semantics.append(ls)
    
    new_semantic_name = os.path.join(in_path, 'semantics_relabel_gapartnet.txt')
    with open(new_semantic_name, 'w') as fd:
        for line in semantics:
            fd.write(' '.join(line) + '\n')
    
    return new_semantic_name, [x[0] for x in semantics]


def create_link_annos(instname2catid, mapping, all_link_names):
    annos = {x: {} for x in all_link_names}
    instname2linkname = {}
    
    for instname in instname2catid.keys():
        if instname == "others":
            continue
        
        assert '/' not in instname
        link_name = instname.split(':')[0]
        cat_name = instname.split(':')[1]
        assert not cat_name.endswith('_null')
        assert mapping[instname2catid[instname] - 1] == cat_name
        annos[link_name]['is_gapart'] = True
        annos[link_name]['category'] = cat_name
        annos[link_name]['bbox'] = None
        instname2linkname[instname] = link_name
    
    for link_name in all_link_names:
        if 'is_gapart' not in annos[link_name].keys():
            annos[link_name]['is_gapart'] = False
    
    return annos, instname2linkname


