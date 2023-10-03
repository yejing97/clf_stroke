import mlp as mlp
import Dataset as Dataset
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import time, os

parser = argparse.ArgumentParser()
parser.add_argument('--doc_namespace', type=str, default="{http://www.w3.org/2003/InkML}")
parser.add_argument('--root_path', type=str, default="/home/xie-y/data/EXP/EXP/")
parser.add_argument('--input_size', type=int, default=400)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=14)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_name', type=str, default="mlp")
parser.add_argument('--result_path', type=str, default="/home/xie-y/data/EXP/results/" + time.strftime('%Y_%m_%d_%H_%M_%S'))
args = parser.parse_args()

data_args = {
    'doc_namespace': args.doc_namespace,
    'root_path': args.root_path,
}
model_args = {
    'input_size': args.input_size,
    'hidden_size': args.hidden_size,
    'output_size': args.output_size,
}
train_data = Dataset.FuzzyEmbeddingDataset('train', args=data_args)
val_data = Dataset.FuzzyEmbeddingDataset('val', args=data_args)
test_data = Dataset.FuzzyEmbeddingDataset('test', args=data_args)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

if args.model_name == 'mlp':
    model = mlp.MLP(**model_args).to(torch.device("cuda"))
else:
    model = svm.SVC(kernel='linear', C=1).to(torch.device("cuda"))

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
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
                try:
                    optimizer.zero_grad()
                    output = model(fg_emb.to(torch.device("cuda")))
                    loss = criterion(output, gt.reshape(-1).to(torch.long).to(torch.device("cuda")))
                    print(loss.item())
                    loss_sum += loss.item()
                    nb_batch += 1
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
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

                    if not os.path.exists(args.result_path):
                        os.makedirs(args.result_path)
                    torch.save(pred, os.path.join(args.result_path, id[0].split('.')[0] + '.pt'))
                    # print(os.path.join(args.result_path, id[0].split('.')[0] + '.pt'))
                except:
                    print(id)
                    print(fg_emb.shape)
        accs.append(acc_sum / nb_val)
        losses_val.append(loss_val / nb_val)
        print('epoch: {}, acc: {}, loss: {}'.format(epoch, acc_sum / nb_val, loss_val / nb_val))


            # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        # torch.save(model.state_dict(), '/home/e19b516g/yejing/data/EXP/model/mlp.pth')

train()