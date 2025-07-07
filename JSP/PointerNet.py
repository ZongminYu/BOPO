import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class CAMEncoder3(nn.Module):
    """
    Graph encoder based on two layers of GAT.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 embed_size: int = 128,
                 dropout: float = 0.,
                 leaky_slope: float = 0.15):
        """
        Constructor.

        Args:
            input_size: Number of features in each node/operation.
            hidden_size: Hidden units in the first layer.
            embed_size: Number of dimensions in the output embeddings.
            leaky_slope: Slope in the leaky ReLU.
        """
        super(CAMEncoder3, self).__init__()
        self.job_gat1 = GATv2Conv(
                            in_channels=input_size,
                            out_channels=hidden_size,
                            dropout=dropout,
                            concat=False,
                            heads=2,
                            add_self_loops=False,
                            negative_slope=leaky_slope)
        self.job_gat2 = GATv2Conv(
                            in_channels=hidden_size*2 + input_size,
                            out_channels=embed_size,
                            dropout=dropout,
                            concat=False,
                            heads=2,
                            add_self_loops=False,
                            negative_slope=leaky_slope)

        self.mac_gat1 = GATv2Conv(
                            in_channels=input_size,
                            out_channels=hidden_size,
                            dropout=dropout,
                            concat=False,
                            heads=2,
                            add_self_loops=False,
                            negative_slope=leaky_slope)
        self.mac_gat2 = GATv2Conv(
                            in_channels=hidden_size*2 + input_size,
                            out_channels=embed_size,
                            dropout=dropout,
                            concat=False,
                            heads=2,
                            add_self_loops=False,
                            negative_slope=leaky_slope)
        #
        self.out_size = input_size + embed_size

    def forward(self, x: torch.Tensor, job_edges: torch.Tensor, mac_edges: torch.Tensor):
        """
        Forward pass.

        Args:
            x: The features for each node. Shape: (num nodes, input_size).
            job_edges: edge index of the job layer. Shape: (2, num edges).
            mac_edges: edge index of the mac layer. Shape: (2, num edges).
        Return:
            The node embeddings.
        """
        #
        h1_job = torch.relu(self.job_gat1(x, job_edges))
        h1_mac = torch.relu(self.mac_gat1(x, mac_edges))
        h = torch.cat([x, h1_job, h1_mac], dim=-1)
        #
        h2_job = self.job_gat2(h, job_edges)
        h2_mac = self.mac_gat2(h, mac_edges)

        return torch.cat([x, torch.relu((h2_job+h2_mac) / 2)], dim=-1)


class LSTMDecoder2(nn.Module):

    def __init__(self,
                 encoder_size: int,
                 context_size: int,
                 hidden_size: int = 64,
                 att_size: int = 128,
                 leaky_slope: float = 0.15,
                 dropout: float=0.,):
        """
        Constructor.

        encoder_size: Number of features in the output of the encoder.
        context_size: Number of features in the state.
        hidden_size: Number of hidden dimensions in the key network.
        att_size: Number of features for the attention.
        """
        super(LSTMDecoder2, self).__init__()
        self.act = nn.LeakyReLU(leaky_slope)
        self.norm = nn.LayerNorm(att_size)
        self.dropout = nn.Dropout(dropout)
        ### Query net
        self.linear1 = nn.Linear(encoder_size, att_size)
        self.lstm = nn.LSTM(att_size, att_size, batch_first=True)
        self.linear2 = nn.Linear(att_size, att_size, bias=False)

        ### Key net
        self.linear3 = nn.Linear(context_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size+encoder_size, att_size, bias=False)

        self.init_weight()

    def forward(self, embed, context, last_embed, h=None, c=None):
        ### Query net
        # [B, 1, E] -> [B, 1, A]
        x = self.act(self.linear1(last_embed))
        # [B, 1, A] -> [B, 1, A]
        if h is None:
            s_g, (new_h, new_c) = self.lstm(x)
        else:
            s_g, (new_h, new_c) = self.lstm(x, (h, c))
        query = self.linear2(self.norm(self.dropout(s_g) + x))

        ### Key net
        # [B, N, IN] -> [B, N, D]
        s_j = self.act(self.linear3(context))
        # [B, N, E] | [B, N, D] -> [B, N, A]
        key = self.linear4(torch.cat([embed, s_j], dim=-1))

        # Attention
        # [B, N, A] @ [B, 1, A].T -> [B, N]
        z = torch.bmm(key, query.permute(0,2,1)).squeeze(-1)
        return z, (new_h, new_c)

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)
        nn.init.xavier_uniform_(self.linear1.weight.data, gain=0.25)
        nn.init.xavier_uniform_(self.linear2.weight.data, gain=0.25)
        nn.init.xavier_uniform_(self.linear3.weight.data, gain=0.25)
        nn.init.xavier_uniform_(self.linear4.weight.data, gain=0.25)
