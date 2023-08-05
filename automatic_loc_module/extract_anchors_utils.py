import numpy as np
import re

def get3d_box_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h


def get3d_box_center_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D center of the 3D box that contains the PCs.
    """
    w, l, h = get3d_box_from_pcs(pc)
    return np.array([pc[:, 0].max() - w / 2, pc[:, 1].max() - l / 2, pc[:, 2].max() - h / 2])


def extract_target_loc_from_pred_objs_from_description(pred_objs_list, target_class):
    indices = [c for c, x in enumerate(pred_objs_list) if x == target_class]  # find indices of the target class
    if len(indices) == 1:
        return indices[0]
    else:  # multiple targets have been found.
        # TODO: Eslam: think about a way to find which one is the target.
        # print("XXX for now will return the first occurrence")
        return indices[-1]  # for now will return the first occurrence


def get_relationship_between_2_objs(target_string, RELATIONS):
        target_string = target_string.lower()
        words_pred = []
        for rel_word in RELATIONS:
            # if rel_word in sub_phrase.split(" "):
            if re.search(r'\b%s\b' % (re.escape(rel_word.lower())), target_string) is not None:
                words_pred.append(rel_word)
        if len(words_pred) == 0:
             # search the other way around:
            for rel_word in RELATIONS:
                if re.search(r'\b%s\b' % (re.escape(target_string)), rel_word.lower()) is not None:
                    words_pred.append(rel_word)
        max_str = ''
        max_len = 0
        if len(words_pred) == 0:
            # get the closest match by finding if the target is a substring of any of the relations:
            for rel_word in RELATIONS:
                if target_string in rel_word.lower() or rel_word.lower() in target_string:
                    words_pred.append(rel_word)
        for word in words_pred:
            if len(word) > len(max_str):
                 max_str = word
        return max_str