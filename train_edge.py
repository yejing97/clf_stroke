import mlp as mlp
import Dataset as Dataset
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
data_args = {
    'doc_namespace': "{http://www.w3.org/2003/InkML}",
    'root_path': "/home/xie-y/data/EXP/EXP/"
}
model_args = {
    'input_size': 400,
    'hidden_size': 128,
    'output_size': 14
}
train_data = Dataset.FuzzyEmbeddingDataset('train', args=data_args)
val_data = Dataset.FuzzyEmbeddingDataset('val', args=data_args)
test_data = Dataset.FuzzyEmbeddingDataset('test', args=data_args)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

model = mlp.MLP(**model_args).to(torch.device("cuda"))

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    losses_train = []
    losses_val = []
    accs = []
    for epoch in range(100):
        # loop = tqdm(, total=100, desc='train')

        loss_sum = 0
        loss_val = 0
        nb_batch = 0
        nb_val = 0
        acc_sum = 0
        model.train()
        for fg_emb, gt, id in tqdm(train_loader):
            if fg_emb.size(1) != 0:
                # print(gt.reshape(-1).shape)
                try:
                    optimizer.zero_grad()
                    output = model(fg_emb.to(torch.device("cuda")))
                    # print(output.shape)
                    # print(gt.reshape(-1).shape)
                    loss = criterion(output, gt.reshape(-1).to(torch.long).to(torch.device("cuda")))
                    loss_sum += loss.item()
                    nb_batch += 1
                    loss.backward()
                    optimizer.step()
                except:
                    print(id[0] + ' error!')
                    print(fg_emb.shape)
        losses_train.append(loss_sum / nb_batch)
        print('epoch: {}, loss: {}'.format(epoch, loss_sum / nb_batch))
        model.eval()
        for fg_emb, gt, id in tqdm(val_loader):
            if fg_emb.size(1) != 0:
                try:
                    output = model(fg_emb.to(torch.device("cuda")))
                    _, pred = output.max(dim=1)
                    acc = accuracy_score(gt.reshape(-1), pred.reshape(-1).cpu())
                    loss = criterion(output, gt.reshape(-1).to(torch.long).to(torch.device("cuda")))
                    loss_val += loss.item()
                    acc_sum += acc
                    nb_val += 1
                except:
                    print(id)
                    print(fg_emb.shape)
        accs.append(acc_sum / nb_val)
        losses_val.append(loss_val / nb_val)
        print('epoch: {}, acc: {}, loss: {}'.format(epoch, acc_sum / nb_val, loss_val / nb_val))


            # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        # torch.save(model.state_dict(), '/home/e19b516g/yejing/data/EXP/model/mlp.pth')

train()