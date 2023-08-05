from tqdm import tqdm
from referit3d.in_out.scannet_scan import ScannetScan, ScannetDataset
import pandas as pd
import string
#from extract_objs_from_description import ExtractObjsFromDescription
import glob
import numpy as np
from benchmark_auto_obj_extraction_module_sr3d import read_referring_data_scv
import multiprocessing as mp
import math
import json
from extraction_utils import *
from extract_anchors_utils import get_relationship_between_2_objs, extract_target_loc_from_pred_objs_from_description,get3d_box_center_from_pcs,get3d_box_from_pcs
import re

def scannet_loader(scan_id):
    """Helper function to load the scans in memory.
    :param scan_id:
    :return: the loaded scan.
    """
    global scannet

    # print("scan_id = ", scan_id)
    scan_i = ScannetScan(scan_id=scan_id, scannet_dataset=scannet, apply_global_alignment=False, load_dense=load_dense)
    if load_dense:
        scan_i.load_point_clouds_of_all_objects_dense()
    else:
        scan_i.load_point_clouds_of_all_objects()

    return scan_i


if __name__ == '__main__':
    df = read_referring_data_scv(file_path="new_nr3d_cot_ref_paraphrases_anchors_3.csv")
    scan_ids = df.scan_id
    # Configurations:
    # ---------------
    load_dense = False
    scannet_dataset_path = ".SCANNET_PATH/"

    # Read the scan related information
    top_scan_dir = scannet_dataset_path + "scannet/scans"
    idx_to_semantic_class_file = '../automatic_loc_module/referit3d/data/mappings/scannet_idx_to_semantic_class.json'
    instance_class_to_semantic_class_file = '../automatic_loc_module/referit3d/data/mappings/scannet_instance_class_to_semantic_class.json'
    axis_alignment_info_file = '../automatic_loc_module/referit3d/data/scannet/scans_axis_alignment_matrices.json'
    scannet = ScannetDataset(top_scan_dir,
                             idx_to_semantic_class_file,
                             instance_class_to_semantic_class_file,
                             axis_alignment_info_file)
    # Loop on the whole scenes and load them once:
    all_scenes_paths = glob.glob(top_scan_dir+"/*")
 
    all_scenes_paths = list(np.unique(np.array(scan_ids)))
    
    scenes_dict = {}
    all_scan_ids = all_scenes_paths
    n_items = len(all_scan_ids)
    n_processes = min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)

    for i, data in enumerate(pool.imap(scannet_loader, all_scan_ids, chunksize=chunks)):
        scenes_dict[all_scan_ids[i]] = data
    pool.close()
    pool.join()

    
    # Create our obj retrieval module:
    # obj_extractor = ExtractObjsFromDescription("./data/scannet_instance_class_to_semantic_class.json")
    
    with open('mapped_relation.json') as f:
        mapped_relation = json.load(f)
    relation_fn_dict = {
        'near': nearest_3dobject,
        'far': farthest_3dobject,
        'right': right_3dobject,
        'left': left_3dobject,
        'on': on_3dobject,
        'down': under_3dobject,
        'front': front_3dobject,
        'back': back_3dobject
    }
    reverse_relation_fn_dict = {
        'near': nearest_3dobject,
        'far': farthest_3dobject,
        'right': left_3dobject,
        'left': right_3dobject,
        'on': under_3dobject,
        'down': on_3dobject,
        'front': back_3dobject,
        'back': front_3dobject
    }

    relations_df = pd.read_csv("relations_parsed.csv")
    relations_df['used_obj'] = [False] * len(relations_df)
    relations_df['used_sub'] = [False] * len(relations_df)
    relations = relations_df[relations_df['id'] == i].drop_duplicates(subset=['object', 'subject', 'relation']).copy()
    #print("The whole relations in the sentence are: ", relations)
    
    gt_objs_name_all_scenes = []
    gt_utternaces_all_scenes = []
    pred_objs_name_all_scenes = []
    counter = 0
    unique_anchor_counter = 0
    empty_relation_counter = 0
    empty_anchor_counter = 0
    no_box_counter = 0
    target_mismatch_target = 0
    correct_counter = 0
    target_counter = 0
    unique_anchor_counter2 = 0
    empty_relation_counter2 = 0
    empty_anchor_counter2 = 0
    hard_counter = 0
    empty_anchor_flag = False
    empty_relation_flag = False
    unique_anchor_flag = False


    all_scene_anchors = {}
    changed_anchors = {}
    all_scenes_refined_path = []
    all_scenes_num_anchors = []
    for i in tqdm(range(len(df))):
    #if True:
        scan_id = scan_ids[i]
        
        refined_path = df['path_edited'][i][1:-1].split(', ')
        #remove single quote from each element in refined_path 
        refined_path = [x[1:-1] for x in refined_path]
        
        # Refine the detected path:
        if len(refined_path) == 0:
            refined_path.append(df['instance_type'][i])
        if df['instance_type'][i] != refined_path[-1]:
            # The target should be the last object in the path if not force it
            if df['instance_type'][i] in refined_path:
                refined_path.remove(df['instance_type'][i])
                refined_path.append(df['instance_type'][i])
            else:  # if not exist at all, add it to the end
                refined_path.append(df['instance_type'][i])
        
        all_scenes_refined_path.append(refined_path)

        empty_anchor_flag = False
        empty_relation_flag = False
        unique_anchor_flag = False
        # Run our obj retrieval module:
        pred_objs_name = extract_objs_from_description(df.utterance[i], df.instance_type[i], refined_path)
        all_scenes_num_anchors.append(len(pred_objs_name)-1)
        # if len(pred_objs_name[-1]) >0 and  df.instance_type[i] != pred_objs_name[-1]:
        #     target_mismatch_target += 1
        trgt_idx = 9999
        # possible_anchors_dict = {}  # clear the dictionary for each scene
        all_scene_anchors[i] = [-1] * (len(pred_objs_name))

        # if len(pred_objs_name) == 1 and pred_objs_name[0] == df.instance_type[i]:
        #     # all_scene_anchors[i].append(-1)
        #     no_anchor_counter_per_scene += 1

        pred_objs_name_all_scenes.append(pred_objs_name)
        # Extract easy objs; prior knowledge (the target and the unique objects):
        pred_anchor = [None] * len(pred_objs_name)
        possible_anchors_dict = {}  # clear the dictionary for each scene
        for pred_obj_idx, pred_obj_name in enumerate(pred_objs_name):  # Loop on the predicted objects in the utterance
            counter += 1
            if '*' == pred_obj_name[0]:
                no_box_counter += 1
                continue
            # Get all the possible objects that exist in the scene and match the predicted obj class from the text:
            if not (pred_obj_name in possible_anchors_dict.keys()):
                possible_anchors_dict[pred_obj_name] = []  # initialize the list once
                for obj_3d in scenes_dict[scan_id].three_d_objects:
                    if (pred_obj_name == obj_3d.instance_label): # or (pred_obj_name in obj_3d.instance_label.split()) or (obj_3d.instance_label in pred_obj_name.split()):
                        possible_anchors_dict[pred_obj_name].append(obj_3d)

            # Exclude the target:
            if pred_obj_name == df.instance_type[i]: # or pred_obj_name in df.instance_type[i] or df.instance_type[i] in pred_obj_name:
                # Detect the target location from the predicted objects from the utterance:
                obj_name = df.instance_type[i]
                if pred_obj_name != df.instance_type[i]:
                    obj_name = pred_obj_name
                target_idx = extract_target_loc_from_pred_objs_from_description(pred_objs_list=pred_objs_name,
                                                                                target_class=obj_name)
                if pred_obj_idx == target_idx:  # make sure it is the target not text-distractor
                    for anchor_id, anchor in enumerate(possible_anchors_dict[pred_obj_name]):
                        if anchor.object_id == df.target_id[i]:
                            target_counter += 1
                            trgt_idx = target_idx
                            target_anchor = possible_anchors_dict[pred_obj_name][anchor_id]
                            pred_anchor[pred_obj_idx] = target_anchor
                            del possible_anchors_dict[pred_obj_name][anchor_id]
                            del all_scene_anchors[i][pred_obj_idx]
                            break
                    continue
            # import pdb; pdb.set_trace()
            if len(possible_anchors_dict[pred_obj_name]) == 0:
                print("XXX  Error the obj not found",pred_obj_name, "!!!")
                empty_anchor_counter += 1
                empty_anchor_flag = True
                continue
            elif len(possible_anchors_dict[pred_obj_name]) == 1:  # Unique object
                pred_anchor[pred_obj_idx] = possible_anchors_dict[pred_obj_name][0]
                all_scene_anchors[i][pred_obj_idx] = possible_anchors_dict[pred_obj_name][0].object_id
                unique_anchor_flag = True
                unique_anchor_counter += 1

        # Assign the hard objs (Several objects) using the geometry info:
        remaining_indices = [c for c, x in enumerate(pred_anchor) if x is None]  # find indices of hard objs
        objs_center = [None] * len(pred_objs_name)
        # Loop on the remaining objects:
        for idx in remaining_indices:
            hard_counter += 1
            pred_obj_name = pred_objs_name[idx]
            if pred_obj_name[0] == '*':  # Skip object that don't have a bbox
                continue

            # check unique obj may be after the target removal the obj become unique.
            if len(possible_anchors_dict[pred_obj_name]) == 1:
                pred_anchor[idx] = possible_anchors_dict[pred_obj_name][0]
                all_scene_anchors[i][idx] = possible_anchors_dict[pred_obj_name][0].object_id
                unique_anchor_flag = True
                unique_anchor_counter += 1
                continue

            # 1- Get the center of each object:
            # 1.1-unassigned objs of same class center
            unassigned_anchors_center = []
            for anchor in possible_anchors_dict[pred_obj_name]:  # unassigned objs
                obj_pc = scenes_dict[scan_id].pc[anchor.points]
                unassigned_anchors_center.append(np.mean(obj_pc, axis=0))
                # w, l, h = get3d_box_from_pcs(obj_pc)
            # 1.2-assigned objs center
            known_indices = [c for c, x in enumerate(pred_anchor) if x is not None]  # find indices of hard objs
            known_centers = []
            for known_idx in known_indices:
                obj_pc = scenes_dict[scan_id].pc[pred_anchor[known_idx].points]
                known_centers.append((get3d_box_center_from_pcs(obj_pc), pred_anchor[known_idx].instance_label))

            # 2- Get the relationship between objects:
            # get relationships containing the current obj:
            relations = get_relationship_conatining_obj(i, relations_df, pred_obj_name, target=df.instance_type[i])
            #print("relations = ", relations)
            # 1st case for random:
            if len(relations) == 0:
                # all_scene_anchors[i].append(-1)
                empty_relation_counter += 1
                empty_relation_flag = True
                rand_idx = np.random.randint(0,len(possible_anchors_dict[pred_obj_name]))
                pred_anchor[idx] = possible_anchors_dict[pred_obj_name][rand_idx]
                # del unassigned_anchors_center[rand_idx]
                all_scene_anchors[i][idx] = pred_anchor[idx].object_id
            deleted_objs = 0
            found = False
            # loop over the relations and get which one of the relations has a known object in the second value of the tuple
            for relation_tuple in relations:
                if found:
                    break
                relation, obj2, flag = relation_tuple
                # Get the closest relation mapping from mapped_relation using get_closest_relation_mapping
                closest_relation = get_relationship_between_2_objs(relation, mapped_relation.keys())
                obj2_centers = [c[0] for c in known_centers if c[1] == obj2]
                if closest_relation == '' or len(obj2_centers) == 0:
                    continue
                
                relation_mapped = mapped_relation[closest_relation]
                for known_center in obj2_centers: 
                    if len(unassigned_anchors_center) == 0:
                        break
                    if flag:
                        approx_anchor = relation_fn_dict[relation_mapped](known_center, unassigned_anchors_center)
                    else:
                        approx_anchor = reverse_relation_fn_dict[relation_mapped](known_center, unassigned_anchors_center)
                    # get the index of the chosen center with regards to unassigned_anchors_center:
                    correct_counter += 1
                    if approx_anchor is not None:
                        chosen_center_idx = np.argmin(np.linalg.norm(np.array(unassigned_anchors_center) - np.array(approx_anchor), axis=1))
                        del unassigned_anchors_center[chosen_center_idx]
                        chosen_center_idx = chosen_center_idx + deleted_objs
                        deleted_objs += 1
                        found = True
                        pred_anchor[idx] = possible_anchors_dict[pred_obj_name][chosen_center_idx]
                        all_scene_anchors[i][idx] = possible_anchors_dict[pred_obj_name][chosen_center_idx].object_id
                    else:
                        # the relation is not valid (e.g., no object is following the relation)
                        # So get the nearest:
                        approx_anchor = relation_fn_dict['near'](known_center, unassigned_anchors_center)
                        chosen_center_idx = np.argmin(np.linalg.norm(np.array(unassigned_anchors_center) - np.array(approx_anchor), axis=1))
                        del unassigned_anchors_center[chosen_center_idx]
                        found = True
                        chosen_center_idx = chosen_center_idx + deleted_objs
                        deleted_objs += 1
                        pred_anchor[idx] = possible_anchors_dict[pred_obj_name][chosen_center_idx]
                        all_scene_anchors[i][idx] = possible_anchors_dict[pred_obj_name][chosen_center_idx].object_id
                    break
        
        
            # 2nd case for random:
            # If u can not find object in all relations so take the nearest one (Random guess)
            if pred_anchor[idx] == None:
                empty_relation_counter += 1
                rand_idx = np.random.randint(0,len(possible_anchors_dict[pred_obj_name]))
                pred_anchor[idx] = possible_anchors_dict[pred_obj_name][rand_idx]
                all_scene_anchors[i][idx] = pred_anchor[idx].object_id
                empty_relation_flag = True
                
                
                
                
        print("---------------------")
        print("num of all objects: ", counter)
        print("num of Target objects: ", (target_counter/counter)*100)
        print("num of Unique objects: ", (unique_anchor_counter/counter)*100)
        print("num of Geometry objects: ", (correct_counter/counter)*100)
        print("num of Random objects: ", (empty_relation_counter/counter)*100)
        print("num of No box objects: ", (no_box_counter/counter)*100)
        print("num of No anchors objects: ", (empty_anchor_counter/counter)*100)
        print("---------------------")
        print("num of all objects: ", counter)
        print("num of Target objects: ", target_counter)
        print("num of Unique objects: ", unique_anchor_counter)
        print("num of Geometry objects: ", correct_counter)
        print("num of Random objects: ", empty_relation_counter)
        print("num of No box objects: ", no_box_counter)
        print("num of No anchors objects: ", empty_anchor_counter)
        print("---------------------")
        print("Check the total number: ", (target_counter + unique_anchor_counter + correct_counter + empty_relation_counter + no_box_counter + empty_anchor_counter)==counter)
        print("---------------------")
        print("The Assigned anchors are: ", len(pred_anchor))

        df['path'] = all_scenes_refined_path
        df['num_anchors'] = all_scenes_num_anchors

        for obj_count, obj in enumerate(pred_anchor):
            print("obj = ", obj)
            if obj:  # as sometimes it is none
                print("Obj ", obj_count, ": ", obj.instance_label, obj.object_id)
                
    #Given all_scene_anchors with lists, add it to the new column in pd dataframe 
    df['all_scene_anchors'] = all_scene_anchors.values()
    df['anchor_ids'] = df['all_scene_anchors']
    df.to_csv("Final_new_nr3d_cot_ref_paraphrases_anchors.csv", index=False)