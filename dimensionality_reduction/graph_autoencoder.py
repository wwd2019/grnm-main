import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
# torch.cuda.set_device(3)

# class GCN_encoder(nn.Module):
#
#     def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
#         super(GCN_encoder, self).__init__()
#         self.conv1 = GraphConv(in_feats, h1_feats)
#         self.conv2 = GraphConv(h1_feats, h2_feats)
#         self.clsf = GraphConv(h2_feats, num_classes)
#
#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         embed = self.conv2(g, h)
#         h = F.relu(embed)
#         h = self.clsf(g, h)
#         return embed, h
class GCN_encoder(nn.Module):
    """
    A Graph Convolutional Network (GCN) encoder that applies graph convolution to input node features
    for the purpose of node classification or embedding generation.

    The model consists of a sequence of graph convolutional layers followed by ReLU activations. The final
    layer outputs can be used for classification or further processed depending on the application.

    Attributes:
        conv1 (GraphConv): The first graph convolutional layer that transforms input features
                           to a higher dimensional hidden space.
        conv2 (GraphConv): The second graph convolutional layer that further transforms the
                           representation from the first layer to another hidden space.
        clsf (GraphConv): The final graph convolutional layer that outputs features which can
                          be used for node classification or other similar tasks.

    Parameters:
        in_feats (int): Number of features in the input node feature vector.
        h1_feats (int): Number of features (neurons) in the first hidden layer.
        h2_feats (int): Number of features (neurons) in the second hidden layer.
        num_classes (int): Number of output features (classes) in the classification layer.
    """

    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        """
        Initializes the GCN encoder module with three graph convolutional layers.

        Args:
            in_feats (int): The size of each input sample.
            h1_feats (int): The size of each output sample from the first GraphConv layer.
            h2_feats (int): The size of each output sample from the second GraphConv layer.
            num_classes (int): The size of each output sample from the final GraphConv layer,
                               typically the number of classes in a classification problem.
        """
        super(GCN_encoder, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, h2_feats)
        self.clsf = GraphConv(h2_feats, num_classes)

    def forward(self, g, in_feat):
        """
        Forward pass of the GCN encoder.

        The input feature is processed through two graph convolutional layers with ReLU activation
        followed by a final graph convolutional layer that produces the output directly.

        Args:
            g (DGLGraph): The graph.
            in_feat (torch.Tensor): The input features, typically node features.

        Returns:
            tuple: A tuple containing the embedding from the second layer and the output of the final layer.
                   The embeddings can be used for tasks such as link prediction or node classification.
        """
        h = self.conv1(g, in_feat)  # Apply first graph convolution layer
        h = F.relu(h)  # Apply ReLU activation function
        embed = self.conv2(g, h)  # Apply second graph convolution layer
        h = F.relu(embed)  # Apply ReLU activation function
        h = self.clsf(g, h)  # Apply final graph convolution layer
        return embed, h  # Return embeddings and final output

class DotPredictor(nn.Module):
    # def forward(self, g, embed):
    #     with g.local_scope():
    #         g.ndata['embed'] = embed
    #         # Compute a new edge feature named 'score' by a dot-product between the
    #         # source node feature 'h' and destination node feature 'h'.
    #         g.apply_edges(fn.u_dot_v('embed', 'embed', 'score'))
    #         # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
    #         return g.edata['score'][:, 0]
    def forward(self, g, embed):
        """
        Forward pass of the DotPredictor.

        Parameters:
            g (dgl.DGLGraph): The graph for which edge scores are to be predicted. The graph should
                              have edges between nodes whose scores are to be computed.
            embed (torch.Tensor): The embeddings of the nodes in the graph. Each row corresponds
                                  to a node's embedding.

        Returns:
            torch.Tensor: A one-dimensional tensor containing the computed edge scores for each edge in
                          the graph. Each score is the result of the dot product between the embeddings of
                          the source and destination nodes of the edge.

        Notes:
            This method uses DGL's local scope to avoid altering the original graph's node and edge data,
            ensuring that operations are localized and temporary within the scope of this method.
        """
        with g.local_scope():  # Temporary graph modifications within this block won't affect the original graph
            g.ndata['embed'] = embed  # Assign embeddings to 'embed' field in graph's node data
            # Apply the dot product between 'embed' of source and destination nodes for each edge
            g.apply_edges(fn.u_dot_v('embed', 'embed', 'score'))
            # Squeeze the result to remove the extra dimension added by 'u_dot_v' which returns a 1-element vector
            return g.edata['score'][:, 0]  # Return the scores as a 1D tensor


class AE(nn.Module):
    """
    An Autoencoder (AE) model for unsupervised learning of efficient codings.

    The model consists of an encoder that reduces the dimensionality of the input to a latent
    space representation, followed by a decoder that reconstructs the input from the latent space.
    """
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
    """A custom Dataset class designed to load and transform data for use with PyTorch's DataLoader."""
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
            torch.from_numpy(np.array(idx))

def compute_loss_2(tf_gcn_embed, tf_ae_embed, x_bar, x, alpha):
    """
          Computes a combined loss that includes both embedding similarity loss and reconstruction loss.

        Parameters:
            tf_gcn_embed (torch.Tensor): Embeddings generated by the Graph Convolutional Network (GCN).
            tf_ae_embed (torch.Tensor): Corresponding embeddings generated by the Autoencoder.
            x_bar (torch.Tensor): Reconstructed outputs from the Autoencoder.
            x (torch.Tensor): Original inputs to the Autoencoder.
            alpha (float): Weighting factor for the embedding loss relative to the reconstruction loss.

        Returns:
            torch.Tensor: The calculated combined loss, weighted by alpha.
    """
    loss_1 = F.mse_loss(tf_gcn_embed, tf_ae_embed)
    loss_2 = F.mse_loss(x_bar, x)

    return alpha * loss_1 + loss_2


def grpah_filtering(grpah):
    """
        Filters the edges of a graph into positive and negative based on edge weights.

        Parameters:
            graph (DGLGraph): The graph whose edges will be filtered.

        Returns:
            tuple: A tuple of lists containing the source and destination node indices for positive and negative edges.
    """
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
    """
    Pretrains an Autoencoder and a GCN model using a specified graph and dataset.

    Parameters:
        model (torch.nn.Module): The Autoencoder model to be trained.
        GCN (torch.nn.Module): The Graph Convolutional Network model to be trained.
        dataset (Dataset): The dataset to train the models on.
        g (DGLGraph): The graph on which the models operate.
        epoch_n (int): Number of epochs to train for.
        alpha_n (float): Weighting factor used in the loss function.

    """
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
class GraphAutoencoderTrainer():
    """
    Trains the graph autoencoder model using the initialized graph and training parameters.

    Parameters:
         n_components (int): Number of dimensions of the output node embeddings.

    Modifies:
        self.z (np.ndarray): Updates the learned embeddings with the output from the autoencoder.
    """
    def __init__(self,g,epoch_n=1000,alpha_n=0.01):
        g = dgl.add_self_loop(g)
        self.epoch_n = epoch_n
        self.alpha_n = alpha_n
        self.g = g
    def train(self,n_components = 8):
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