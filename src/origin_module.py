import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch.nn import Sequential, Linear, ReLU

#   origin_module
#   GCN / GraphSage / GIN / GAT
class Origin_Module(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(Origin_Module, self).__init__()
        self.model_type = model_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList() 

        if self.model_type == 'gcn': # default normalize=True
            self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))   # in - hidden  1st layer    
            for _ in range(self.num_layers - 2):    # hidden - hidden  (n-2) layer
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))   
            self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))  # hidden - out  last layer
        elif self.model_type == 'sage': # default normalize=False
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(self.num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        elif self.model_type == 'gin':
            self.convs.append(GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)), train_eps=True))
            for _ in range(self.num_layers - 2):
                self.convs.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)), train_eps=True))
            self.convs.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels)), train_eps=True))
        elif self.model_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels//4, heads=4, dropout=0.6))
            for _ in range(self.num_layers - 2):
                self.convs.append(GATConv(hidden_channels, hidden_channels//4, heads=4, dropout=0.6))
            # On the Pubmed dataset, use heads=8 in conv2.
            self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False,
                                 dropout=0.6))
        else:
            assert False

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):    
            if self.model_type == 'gcn':
                x = self.convs[i](x, edge_index)         
                # x = x[:size[1]]
            else:
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                if self.model_type == 'gcn':
                    x = self.convs[i](x, edge_index)
                    # print(x.shape, size)
                    # x = x[:size[1]]
                else:
                    x_target = x[:size[1]]  
                    x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

