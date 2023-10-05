import torch
import os
import numpy as np

# from LG.lg import Lg
# from vocab.vocab import vocab

class CROHMEDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, args) -> None:
        super().__init__()
        self.doc_namespace = "{http://www.w3.org/2003/InkML}"
        self.root_path = "/home/e19b516g/yejing/data/WebData_CROHME23/"
        # self.root_path = "/WebData_CROHME23/"
        self.csv_path = "./csv/"
        self.data_path = os.path.join(self.root_path, data_type)
        self.lg_path = os.path.join(self.data_path, '/CROHME/LG')
        self.data_list, self.lg_list = self.get_data_list(data_type)
        self.min_points = 3
        self.max_strokes = 60
        print('successfully init CROHME dataset')
# from vocab.vocab import vocab

class FuzzyEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, args) -> None:
        # self.doc_namespace = "{http://www.w3.org/2003/InkML}"
        self.doc_namespage = args['doc_namespace']
        self.root_path = args['root_path']
        # self.root_path = '/home/e19b516g/yejing/data/EXP/'
        self.data_type = data_type
        self.filter_type = args['filter_type']
        self.feature_type = args['feature_type']
        self.data_list = self.get_data_list()
        print('successfully init Fuzzy Embedding dataset')
    
    def get_data_list(self):
        data_list = []
        for root, _, files in os.walk(os.path.join(self.root_path, self.feature_type, self.data_type)):
            for file in files:
                if file.endswith('.npy') and os.path.exists(os.path.join(root.replace(self.feature_type, 'GT'), file)):
                    data_list.append(os.path.join(root, file))
        return data_list
    
    def keep_los(self, fg_emb, gt):
        a = torch.ones_like(gt)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if torch.all(fg_emb[i, j, :, :] == 0):
                    a[i, j] = 0
        indices = torch.nonzero(a.reshape(-1)).squeeze()
        fg_emb = fg_emb.view(-1, fg_emb.shape[2], fg_emb.shape[3])[indices]
        gt = gt.reshape(-1)[indices]
        return fg_emb, gt
    
    def keep_rel(self, fg_emb, gt):
        a = torch.ones_like(gt)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if torch.all(gt[i,j] == 0):
                    a[i, j] = 0
        indices = torch.nonzero(a.reshape(-1)).squeeze()
        fg_emb = fg_emb.view(-1, fg_emb.shape[2], fg_emb.shape[3])[indices]
        gt = gt.reshape(-1)[indices]
        return fg_emb, gt

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        try:
            data_path = self.data_list[index]
            file_name = data_path.split('/')[-1]
            fg_emb = torch.from_numpy(np.load(data_path))
            gt_path = data_path.replace(self.feature_type, 'GT')
            gt = torch.from_numpy(np.load(gt_path))
            if self.filter_type == 'los':
                fg_emb, gt = self.keep_los(fg_emb, gt)
            elif self.filter_type == 'rel':
                fg_emb, gt = self.keep_rel(fg_emb, gt)
            return fg_emb, gt, file_name
        except:
            if index < len(self.data_list):
                print(self.data_list[index])
                return self.__getitem__(index+1)
            else:
                return self.__getitem__(-1)