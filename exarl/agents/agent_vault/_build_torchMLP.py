
import torch
import torch.nn as nn

class MLPModel(nn.Module):

    def __init__(self, in_size, out_size, num_l , l_size, 
                 activ=nn.ReLU,
                 out_activ=nn.Identity,
                 squash_out = False):
        super(MLPModel,self).__init__()

        self.activation = activ()
        self.out_activation = out_activ()
        self.squash_out = squash_out
        self.state_means = self.register_buffer(name="state_means",tensor=torch.zeros(in_size, requires_grad=False))
        self.state_std = self.register_buffer(name="state_std",tensor=torch.ones(in_size, requires_grad=False))
        
        if num_l == 0:
            self.mlp_layers = []
            self.mlp_out_layer = nn.Linear(in_size, out_size,bias=False)
        else:
            self.mlp_layers = nn.ModuleList([nn.Linear(in_size, l_size, bias=True)])
            self.mlp_layers.extend([nn.Linear(l_size, l_size,bias=True) for _ in range(num_l-1)])
            self.mlp_out_layer = nn.Linear(l_size, out_size,bias=True)

    
    def forward(self,data):
        # Normalize the input data with mean and std
        data = (torch.as_tensor(data, dtype=self.state_means.dtype) - self.state_means) / self.state_std
       
        for layer in self.mlp_layers:
            data = self.activation(layer(data.float()))

        # Last Layer of the network output..
        out = self.out_activation(self.mlp_out_layer(data))

        # Pass the output via tanh activation function
        if self.squash_out:
            out = nn.Tanh()(out)
        return out
