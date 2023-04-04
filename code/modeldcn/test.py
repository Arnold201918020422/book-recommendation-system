import torch
import torch.nn as nn



# Layers in a cross layer
class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xi):
        interaction_out = self.weight(xi) * x0 + self.bias
        return interaction_out


# A submodule in a cross layer
class CrossLayerBlock(nn.Module):
    def __init__(self, input_dim, layer_num):
        super(CrossLayerBlock, self).__init__()
        # number of cross layer
        self.layer_num = layer_num
        # Enter the dimension of the feature
        self.input_dim = input_dim
        self.layer = nn.ModuleList(CrossLayer(self.input_dim) for _ in range(self.layer_num))

    def forward(self, x0):
        xi = x0
        for i in range(self.layer_num):
            xi = xi + self.layer[i](x0, xi)
        return xi


# implementation  of essay
class DCN(nn.Module):
    # Initialize the network layer
    def __init__(self, input_dim, embedding_dim, cross_layer_num, deep_layer_num):
        super(DCN, self).__init__()
        # Enter the dimension of the feature
        self.input_dim = input_dim
        # The dimensions of the embedding vector
        self.embedding_dim = embedding_dim
        # The number of cross layers
        self.cross_layer_num = cross_layer_num
        # The number of deep layers
        self.deep_layer_num = deep_layer_num

        # Embedding layers
        self.embedding_layer = nn.Linear(self.input_dim, self.embedding_dim, bias=False)
        # cross layer
        self.cross_layer = CrossLayerBlock(self.embedding_dim, self.cross_layer_num)
        # deep layer
        self.deep_layer = nn.ModuleList()
        for i in range(self.deep_layer_num):
            layer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
                nn.ReLU(),
            )
            self.deep_layer.append(layer)
        self.deep_layer = nn.Sequential(*self.deep_layer)
        # Output layer
        self.output_layer = nn.Linear(self.embedding_dim * 2, 2, bias=True)

    # Forward passing, build a computational graph
    def forward(self, x):
        # Perform feature embedding
        x_embedding = self.embedding_layer(x)
        # cross layer
        cross_resault = self.cross_layer(x_embedding)
        # deep layer
        deep_resault = self.deep_layer(x_embedding)
        temp_resault = torch.cat([cross_resault, deep_resault], dim=-1)
        out = self.output_layer(temp_resault)
        out = torch.sigmoid(out)#change the sentence
        return out


# data testing
if __name__ == '__main__':
    x = torch.randn(1, 10)
    dcn = DCN(10, 20, 2, 2)
    output = dcn(x)
    print(output)