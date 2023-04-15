import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv

from src import ley_agg, ley_sage_conv, ley_gin_conv

from torch.nn import Sequential, Linear, ReLU


#ley_sage / ley_gin
class Ley_Module(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(Ley_Module, self).__init__()
        self.model_type = model_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()  
        
        if self.model_type == 'sage': # default normalize=False
            self.convs.append(ley_sage_conv.Ley_SAGEConv(in_channels, hidden_channels))
            for _ in range(self.num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        else: 
            #self.model_type == 'gin':
            self.convs.append(ley_gin_conv.Ley_GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)), train_eps=True))
            for _ in range(self.num_layers - 2):
                self.convs.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)), train_eps=True))
            self.convs.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels)), train_eps=True))
   
    
    def forward(self, x, adjs, ley_agg_out):
        for i, (edge_index, _, size) in enumerate(adjs):   
            if i==0:
                x_target = x[:size[1]] 
                x = self.convs[i]((x, x_target), ley_agg_out)
            else:
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)



