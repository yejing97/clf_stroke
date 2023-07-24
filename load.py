from torch import nn
import torch
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import intervals as I
PI = math.pi


from LG.lg import Lg
from vocab.vocab import vocab
import normalization

doc_namespace = "{http://www.w3.org/2003/InkML}"

def load_lg(lg_path, dic):
    lg = Lg(lg_path).segmentGraph()
    edge = lg[3]
    node = lg[0]
    strokes = lg[1]
    obj_dic = {}
    strokes_keys = sorted(strokes, key = lambda x:int(x))
    for key in strokes_keys:
        obj_dic[int(key)] = list(strokes[key].values())[0]
    for i in [k for k,v in dic.items() if v == 'None']:
        obj_dic[int(i)] = 'None'
    dic_index = list(sorted(obj_dic.keys()))

    new_matrix = torch.zeros(len(dic), len(dic))
    for key in edge:
        s0 = key[0]
        s1 = key[1]
        relation = list(edge[key].keys())[0]
        index0 = [k for k,v in obj_dic.items() if v == s0]
        index1 = [k for k,v in obj_dic.items() if v == s1]
        for i in index0:
            for j in index1:
                new_matrix[dic_index.index(int(i)), dic_index.index(int(j))] = vocab.rel2indices([relation])[0]
    for key in node:
        for i in node[key][0]:
            for j in node[key][0] - set(i):
                new_matrix[dic_index.index(int(i)), dic_index.index(int(j))] = vocab.rel2indices(['INNER'])[0]
    return new_matrix
        

def load_inkml(file_path):
    strokes = []
    labels = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    last_stroke = []
    dic = {}
    for trace_tag in root.findall(doc_namespace + 'traceGroup'):
        for trace_tag in trace_tag.findall(doc_namespace + 'traceGroup'):
            for annotation in trace_tag.findall(doc_namespace + 'annotation'):
                label = annotation.text
            for traceview in trace_tag.findall(doc_namespace + 'traceView'):
                s_id = traceview.get('traceDataRef')
                dic[s_id] = label
    for trace_tag in root.findall(doc_namespace + 'trace'):
        points = []
        last_point = (0, 0)
        for coord in (trace_tag.text).replace('\n', '').split(','):
            this_point = (float(coord.strip().split(' ')[0]), float(coord.strip().split(' ')[1]))
            if this_point != last_point:
                points.append(this_point)
                last_point = this_point
        for coord in trace_tag.items():
            id = coord[1]
        if last_stroke != points:
            strokes.append(points)
            points_np = np.array(points)
            if np.isnan(points_np).any() == True:
                print(file_path)
            if dic.__contains__(id):
                labels.append(dic[id])
            else:
                labels.append('None')
                dic[id] = 'None'
        last_stroke = points
    return strokes, vocab.words2indices(labels), dic

def combine_inkml_lg(inkml_path, lg_path, norm_nb, norm_type = 'stroke_keep_shape', speed_norm = False, edge_combination = 'diff'):
    strokes, labels, dic = load_inkml(inkml_path)
    LOS_edge = LOS(strokes)
    # strokes_foredge = normalization.norm_eq(strokes)
    # strokes_for_edge = normalization.padding_fixed_length(strokes, norm_nb, type = 'eq_keep_shape', speed_norm = speed_norm)
    # strokes = normalization.padding_fixed_length(strokes, norm_nb, norm_type, speed_norm)
    stroke_level, eq_level = normalization.normalization(strokes, norm_nb, norm_type, speed_norm)
    # eq_level = normalization.normalization(strokes, norm_nb, 'eq_keep_shape', speed_norm)
    edges = np.array([]).reshape(0, norm_nb * 2, 2)
    edge_l = load_lg(lg_path, dic)
    edge_labels = []
    # LOS_edge = LOS(strokes)
    if LOS_edge.shape[0] == edge_l.shape[0]:
        gt_edge = torch.zeros(LOS_edge.shape)
        gt_edge[edge_l > 0] = 1
        strcture = LOS_edge.bool() + gt_edge.bool()
        for i in range(strcture.shape[0]):
            for j in range(strcture.shape[1]):
                if strcture[i][j] == True:
                    if edge_combination == 'concate':
                        edge = edge_combination_concate(eq_level[i], eq_level[j])
                    else :
                        edge = edge_combination_diff(eq_level[i], eq_level[j])
                    # print(edge.shape)
                    # print(edges.shape)
                    edges = np.vstack((edges, edge))
                    edge_labels.append(edge_l[i][j])
        return torch.tensor(stroke_level), torch.tensor(labels),torch.tensor(edges), torch.tensor(edge_labels)

def edge_combination_concate(stroke1, stroke2):
    return np.concatenate((stroke1, stroke2), axis=0).reshape(1, -1, 2)

def edge_combination_diff(stroke1, stroke2):
    return (stroke1 - stroke2).reshape(1, -1, 2)

def calculate_center(points) :
    left = math.inf
    right = -math.inf 
    top = -math.inf 
    bottom = math.inf 
    for p in points :
        left = min(left, p[0])
        right = max(right, p[0])
        top = max(top, p[1])
        bottom = min(bottom, p[1])
    return np.array([(left + right) / 2, (top + bottom) / 2])

def distance(s, center) :
    return math.sqrt(pow(center[0] - calculate_center(s)[0], 2) + pow(center[1] - calculate_center(s)[1], 2))


def LOS(strokes) :
    size = len(strokes)
    edges = np.zeros((size, size))
    for i in range(len(strokes)) : 
        sc = calculate_center(strokes[i])
        U = I.FloatInterval.closed(0, 2*PI) # intervals
        # s = sorted(strokes, key = lambda x:distance(x,sc))
        index = sorted(range(len(strokes)), key = lambda x:distance(strokes[x],sc))
        for j in index :
            if i != j: 
                min_theta = math.inf
                max_theta = -math.inf
                for n in strokes[j] :
                    n = np.array(n)
                    w = n - sc
                    h = np.array([1, 0])
                    if w[1] >= 0 :
                        theta = math.acos(np.dot(w, h) / np.linalg.norm(w,ord=1) * np.linalg.norm(h,ord=1))
                    else :
                        theta = 2 * PI - math.acos(np.dot(w, h) / np.linalg.norm(w,ord=1) * np.linalg.norm(h,ord=1))
                    min_theta = min(min_theta, theta)
                    max_theta = max(max_theta, theta)
                h = I.FloatInterval.closed(min_theta, max_theta)
                # print(h)
                V = U.intersection(h)
                if not V.is_empty() :
                    edges[i][j] = 1
                    edges[j][i] = 1
                # updates intervals
                    U = U - h

    return torch.tensor(edges)