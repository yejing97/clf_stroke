import mlp as mlp
import Dataset as Dataset
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import time, os
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--doc_namespace', type=str, default="{http://www.w3.org/2003/InkML}")
parser.add_argument('--root_path', type=str, default="/home/xie-y/data/EXP/")
parser.add_argument('--feature_nb', type=int, default=5)
parser.add_argument('--input_size', type=int, default=20)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--filter_type', type=str, default="los")
parser.add_argument('--output_size', type=int, default=14)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--model_name', type=str, default="mlp")
parser.add_argument('--result_path', type=str, default="/home/xie-y/data/EXP/results/")
parser.add_argument('--log', type=str, default="/home/xie-y/data/EXP/results/logs/")
# parser.add_argument('--result_path', type=str, default="/home/xie-y/data/EXP/results/" + time.strftime('%m_%d_%H_%M_%S'))
args = parser.parse_args()

data_args = {
    'doc_namespace': args.doc_namespace,
    'root_path': os.path.join(args.root_path, str(args.feature_nb)),
    'filter_type': args.filter_type,
}
# print(data_args)
model_args = {
    'input_size': args.input_size,
    'hidden_size': args.hidden_size,
    'output_size': args.output_size,
    # 'result_path': args.result_path,
}

def get_logger(log_path):
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

log_path = os.path.join(args.log, 'f_nb_' + str(args.feature_nb) + '_lr_'+ str(args.lr) + '_filter_' + str(args.filter_type) + '.log')
logger = get_logger(log_path)

train_data = Dataset.FuzzyEmbeddingDataset('train', args=data_args)
val_data = Dataset.FuzzyEmbeddingDataset('val', args=data_args)
test_data = Dataset.FuzzyEmbeddingDataset('test', args=data_args)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

if args.model_name == 'mlp':
    model = mlp.MLP(**model_args).to(torch.device("cuda"))
else:
    model = svm.SVC(kernel='linear', C=1).to(torch.device("cuda"))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
criterion = torch.nn.CrossEntropyLoss()
losses_train = []
losses_val = []
accs = []
logger.info('----------{}---------------'.format(time.strftime('%m_%d_%H_%M_%S')))
logger.info('----------start training---------------')
logger.info('feature number: {}'.format(args.feature_nb))
logger.info('filter type: {}'.format(args.filter_type))
logger.info('epoches number: {}'.format(args.epoches))
logger.info('initial learning rate: {}'.format(args.lr))

for epoch in range(args.epoches):
    logger.info('epoch: {} start, learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    loss_sum = 0
    loss_val = 0
    nb_batch = 0
    nb_val = 0
    acc_sum = 0
    model.train()
    for fg_emb, gt, id in tqdm(train_loader):
        # print(fg_emb.shape)
        if fg_emb.size(1) != 0:
            # try:
                output = model(fg_emb.to(torch.device("cuda")))
                loss = criterion(output, gt.reshape(-1).to(torch.long).to(torch.device("cuda")))
                loss_sum += loss.item()
                nb_batch += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            # except:
            #     logger.debug(id)
            #     logger.debug(fg_emb.shape)
    # losses_train.append(loss_sum / nb_batch)
    logger.info('epoch: {} end, loss: {}'.format(epoch, loss_sum / nb_batch))
    # print('epoch: {}, loss: {}'.format(epoch, loss_sum / nb_batch))
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
                pred_path = os.path.join(args.result_path, 'prediction' , 'f_nb_' + str(args.feature_nb) + '_lr_'+ str(args.lr) + '_filter_' + str(args.filter_type))
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                # pred_path = os.path.join(args.result_path, 'f_nb_' + str(args.feature_nb) + '_lr_'+ str(args.lr) +'.pt')
                torch.save(pred, os.path.join(pred_path, id[0].split('.')[0] + '.pt'))
                # print(os.path.join(args.result_path, id[0].split('.')[0] + '.pt'))
            except:
                logger.debug(id)
                logger.debug(fg_emb.shape)
                # print(id)
                # print(fg_emb.shape)
    accs.append(acc_sum / nb_val)
    losses_val.append(loss_val / nb_val)
    logger.info('epoch: {} validation, acc: {}, loss: {}'.format(epoch, acc_sum / nb_val, loss_val / nb_val))
    model_path = os.path.join(args.result_path, 'model', 'f_nb_' + str(args.feature_nb) + '_lr_'+ str(args.lr) + '_filter_' + str(args.filter_type))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, 'epoch_' + str(epoch) + '.pt'))
    logger.info('epoch: {} model saved'.format(epoch))
    # print('epoch: {}, acc: {}, loss: {}'.format(epoch, acc_sum / nb_val, loss_val / nb_val))


            # print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        # torch.save(model.state_dict(), '/home/e19b516g/yejing/data/EXP/model/mlp.pth')