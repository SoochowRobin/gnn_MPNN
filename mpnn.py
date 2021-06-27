import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# MessagePassing is base class


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        # define linear transformation, input channels and output channels
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # x has shape [num_nodes, num_features]
        # edge_index has shape [2, E]
    def forward(self, x, edge_index):
        # step 1: add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # step 2: linear transform node feature matrix
        x = self.lin(x)

        # step 3: compute normalization coefficients
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # step 4: start propagating message
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


x = torch.tensor([[1, 0, 1], [2, 1, -1], [-1, 3, 2], [2, 0, 1], [4, 5, -2]], dtype=torch.float)
edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 4], [1, 2, 0, 2, 0, 1, 3, 2, 4, 3]])

conv = GCNConv(3, 4)
x = conv(x, edge_index)
print(x.shape)
print(x)