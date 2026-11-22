import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import dgl
import dgl.function as fn
import pandas as pd
import pickle as pkl
from dgl.nn.pytorch import GraphConv


# torch.cuda.set_device(3)


class GCN_encoder(nn.Module):

    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        super(GCN_encoder, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, h2_feats)
        self.clsf = GraphConv(h2_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        embed = self.conv2(g, h)
        h = F.relu(embed)
        h = self.clsf(g, h)
        return embed, h


class DotPredictor(nn.Module):
    def forward(self, g, embed):
        with g.local_scope():
            g.ndata['embed'] = embed
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('embed', 'embed', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))

        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
            torch.from_numpy(np.array(idx))


# def compute_loss(clsf_pred, clsf_label, pos_score, neg_score, alpha):
#     clsf_loss = nn.BCEWithLogitsLoss()
#     loss_1 = clsf_loss(clsf_pred, clsf_label)

#     scores = torch.cat([pos_score, neg_score]).cuda()
#     labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cuda()
#     loss_2 = F.binary_cross_entropy_with_logits(scores, labels)
#     return alpha*loss_1 + (1-alpha)*loss_2 #+ beta*rec_loss

def compute_loss_2(tf_gcn_embed, tf_ae_embed, x_bar, x, alpha):
    loss_1 = F.mse_loss(tf_gcn_embed, tf_ae_embed)
    loss_2 = F.mse_loss(x_bar, x)

    return alpha * loss_1 + loss_2


def grpah_filtering(grpah):
    weight = grpah.edata['weight'].cpu().numpy()
    src = grpah.edges()[0].cpu().numpy()
    dst = grpah.edges()[1].cpu().numpy()
    neg_src = []
    neg_dst = []
    pos_src = []
    pos_dst = []
    for i in range(len(weight)):
        if (weight[i] < 0):
            neg_src.append(src[i])
            neg_dst.append(dst[i])
        else:
            pos_src.append(src[i])
            pos_dst.append(dst[i])

    return (pos_src, pos_dst, neg_src, neg_dst)


def pretrain_ae(model, GCN, dataset, g,epoch_n=1000,alpha_n=0.01):
    # train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = Adam([{'params': model.parameters()}, {'params': GCN.parameters()}], lr=1e-3)

    for epoch in range(epoch_n):

        embed, clsf = GCN(g, g.ndata['all'])  # .to(torch.float))
        x_bar, z = model(g.ndata['all'])  # .to(torch.float))
        #         labels = g.ndata['labels']
        #         labeled = g.ndata['with_label']

        #         pos_score = pred(pos_g, embed)
        #         neg_score = pred(neg_g, embed)
        TF = g.ndata['TF']
        alpha = alpha_n

        # x = torch.Tensor(dataset.x).cuda().float()

        x = torch.Tensor(dataset.x).cpu().double()
        x_bar, _ = model(x)
        # loss = compute_loss(clsf[labeled], labels[labeled], pos_score, neg_score, alpha) + F.mse_loss(x_bar, x)
        loss = compute_loss_2(embed[TF], z[TF], x_bar, x, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                x = torch.Tensor(dataset.x).cpu().double()
                x_bar, z = model(x)
                # loss = compute_loss(clsf[labeled], labels[labeled], pos_score, neg_score, alpha) + F.mse_loss(x_bar, x)
                loss = compute_loss_2(embed[TF], z[TF], x_bar, x, alpha)
                print('{} loss: {}'.format(epoch, loss))
                # kmeans = KMeans(n_clusters=9, n_init=20).fit(z.data.cpu().numpy())
            # eva(y, kmeans.labels_, epoch)

        # torch.save(model.state_dict(), 'model.pkl')
class gae_m():
    def __init__(self,g,epoch_n=1000,alpha_n=0.01):
        g = dgl.add_self_loop(g)
        self.epoch_n = epoch_n
        self.alpha_n = alpha_n
        self.g = g
    def gae_ex(self,n_components = 8):
        fea = self.g.ndata['all'].cpu().numpy()
        GCN = GCN_encoder(self.g.ndata['all'].shape[1], 512, 64, n_components)
        GCN = GCN.double()
        GCN.cpu()
        model = AE(
            n_enc_1=500,
            n_enc_2=500,
            n_enc_3=2000,
            n_dec_1=2000,
            n_dec_2=500,
            n_dec_3=500,
            n_input=fea.shape[1],
            n_z=64, )
        model = model.double()
        model.cpu()
        # pred = DotPredictor().cuda()

        # pos_u, pos_v, neg_u, neg_v = grpah_filtering(g)
        # pos_g = dgl.graph((pos_u, pos_v), num_nodes=g.number_of_nodes())
        # neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())

        self.g = self.g.to('cpu')
        # pos_g = pos_g.to('cuda')
        # neg_g = neg_g.to('cuda')

        x = np.float64(fea)

        dataset = LoadDataset(x)
        pretrain_ae(model, GCN, dataset,self.g,self.epoch_n,self.alpha_n)
        with torch.no_grad():
            x = torch.Tensor(dataset.x).cpu().double()
            x_bar, z = model(x)
            # loss = F.mse_loss(x_bar, x)
            z = z.data.cpu().numpy()
        self.z = z