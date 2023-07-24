import load
import normalization
import os
import torch
import numpy as np
import tqdm 

# def norm_type():
#     norm_type = ['stroke_keep_shape', 'stroke_tension']


def make_node_pt(tgt_path, inkml_path, norm_nb, norm_type = normalization.stroke_keep_shape, speed_type = normalization.Speed_norm_stroke):
    print('start make node pt:' + str(norm_nb) + ' ' + norm_type.__name__ + ' ' + speed_type.__name__)
    pt_name = str(norm_nb) + '_' + norm_type.__name__ + '_' + speed_type.__name__
    all_nodes = np.zeros((0, norm_nb, 2))
    # all_node_labels = np.zeros((0))
    all_node_labels = []
    for _, _, files in os.walk(inkml_path):
        with tqdm.tqdm(total=len(files)) as pbar:
            for file in files:
                pbar.update(1)
                if file.endswith('.inkml'):
                    inkml = os.path.join(inkml_path, file)
                    try:
                        strokes, labels, dic = load.load_inkml(inkml)
                        normed_strokes = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)
                        # gt = np.array(labels)

                        max_len = max(normalization.stroke_length(stroke)[-1] for stroke in strokes)
                        alpha = max_len/norm_nb
                        for i in range(len(strokes)):
                            # speed normalization or no-speed normalization
                            new_stroke = speed_type(strokes[i], norm_nb, alpha)
                            # [0,1] normalization
                            new_stroke = norm_type(new_stroke)
                            if np.isnan(new_stroke).any() == True:
                                print('norm error')
                            else:
                                n = np.zeros((norm_nb, 2))
                                # padding
                                n[:new_stroke.shape[0], :] = new_stroke
                                normed_strokes[i, :, :] = n
                        all_nodes = np.vstack((all_nodes, normed_strokes))
                        all_node_labels = all_node_labels + labels
                    except:
                        print(inkml)
                        continue
    pbar.close()
    os.makedirs(os.path.join(tgt_path), exist_ok=True)
    torch.save(torch.tensor(all_nodes), os.path.join(tgt_path, 'X_node.pt'))
    torch.save(torch.tensor(all_node_labels), os.path.join(tgt_path, 'y_node.pt'))

def make_edge_pt(tgt_path, inkml_path, lg_path, norm_nb, norm_type = normalization.stroke_keep_shape, speed_type = normalization.Speed_norm_stroke, edge_combination = load.edge_combination_concate):
    pt_name = str(norm_nb) + '_' + norm_type.__name__ + '_' + speed_type.__name__ 
    print('start make edge pt:' + pt_name + ' ' + edge_combination.__name__)
    if edge_combination == load.edge_combination_diff:
        all_edges = np.zeros((0, norm_nb, 2))
    else:
        all_edges = np.zeros((0, norm_nb * 2, 2))
    all_edges_labels = []
    for _, _, files in os.walk(inkml_path):
        for _, _, files in os.walk(inkml_path):
            with tqdm.tqdm(total=len(files)) as pbar:
                for file in files:
                    pbar.update(1)
                    if file.endswith('.inkml'):
                        inkml = os.path.join(inkml_path, file)
                        lg = os.path.join(lg_path, file.replace('inkml', 'lg'))
                        if os.path.exists(lg):
                            # try:
                                strokes, labels, dic = load.load_inkml(inkml)
                                edge_l = load.load_lg(lg, dic)
                                LOS_edge = load.LOS(strokes)
                                new_strokes = []
                                edge_labels = []
                                if edge_combination == load.edge_combination_diff:
                                    edges = np.array([]).reshape(0, norm_nb, 2)
                                else:
                                    edges = np.array([]).reshape(0, norm_nb * 2, 2)

                                max_len = max(normalization.stroke_length(stroke)[-1] for stroke in strokes)
                                alpha = max_len/norm_nb
                                for i in range(len(strokes)):
                                    # speed normalization or no-speed normalization
                                    new_stroke = speed_type(strokes[i], norm_nb, alpha)
                                    # new_stroke = norm_type(new_stroke)
                                    new_strokes.append(new_stroke)
                                # [0,1] normalization + padding
                                new_strokes = normalization.eq_keep_shape(new_strokes, norm_nb)

                                if LOS_edge.shape[0] == edge_l.shape[0]:
                                    gt_edge = torch.zeros(LOS_edge.shape)
                                    gt_edge[edge_l > 0] = 1
                                    strcture = LOS_edge.bool() + gt_edge.bool()
                                    for i in range(strcture.shape[0]):
                                        for j in range(strcture.shape[1]):
                                            if strcture[i][j] == True:
                                                edge = edge_combination(new_strokes[i], new_strokes[j])
                                                # print(edge.shape)
                                                # print(edges.shape)
                                                edges = np.vstack((edges, edge))
                                                edge_labels.append(edge_l[i][j])
                                all_edges = np.vstack((all_edges, edges))
                                all_edges_labels = all_edges_labels + edge_labels
                            # except:
                            #     print(inkml)
                            #     continue
    pbar.close()
    os.makedirs(os.path.join(tgt_path), exist_ok=True)
    torch.save(torch.tensor(all_edges), os.path.join(tgt_path, 'X_'+ edge_combination.__name__ + '.pt'))
    torch.save(torch.tensor(all_edges_labels), os.path.join(tgt_path, 'y_'+ edge_combination.__name__ +'.pt'))



# make_node_pt(
#     tgt_path = '/home/e19b516g/yejing/code/g2g/data/pt/',
#     inkml_path='/home/e19b516g/yejing/data/for_strokes_inkml/',
#     norm_nb=50
# )
# make_edge_pt(
#     tgt_path = '/home/e19b516g/yejing/code/g2g/data/pt/',
#     inkml_path='/home/e19b516g/yejing/data/for_strokes_inkml/',
#     lg_path='/home/e19b516g/yejing/data/for_strokes_lg/',
#     norm_nb=50

# )