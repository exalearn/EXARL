import torch
import torch.nn as nn

class LSTMPolicyModel(nn.Module):

    def __init__(self,in_size, out_size, num_l , l_size, 
                 activ=nn.ReLU,
                 out_activ=nn.Identity,
                 squash_out = True):
        super(LSTMPolicyModel, self).__init__()

        self.activation = activ()
        self.out_activation = out_activ()
        self.squash_out = squash_out

        self.l_size = l_size
        self.in_size = in_size

        self.state_means = self.register_buffer(name="state_means",tensor=torch.zeros(in_size, requires_grad=False))
        self.state_std = self.register_buffer(name="state_std", tensor=torch.ones(in_size, requires_grad=False))

        self.lstm = nn.LSTM(input_size=in_size, hidden_size=l_size,num_layers=num_l)
        self.FC = nn.Linear(l_size, l_size,bias=True)
        self.out_layer = nn.Linear(l_size, out_size,bias=True)
    
    def forward(self,data):
        # Normalize the input data with mean and std
        data = (torch.as_tensor(data, dtype=self.state_means.dtype) - self.state_means) / self.state_std

        data = data.view(1, 1, self.in_size) 
        out_ , (hn,cn) = self.lstm(data.float())

        hn = out_.view(-1, self.l_size) #reshaping the data for Dense layer next
        
        out = self.activation(hn)
        
        out = self.activation(self.FC(out)) #first Dense
        out = self.out_activation(self.out_layer(out)) #Final Output

        # Pass the output via tanh activation function
        if self.squash_out:
            out = nn.Tanh()(out)
        return out