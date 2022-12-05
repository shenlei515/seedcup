import torch
import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        """Naive FCModel as reference.

        Args:
            in_features (int): input size for feature dimension
            out_features (int): output size for feature dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.attn1 = nn.parameter.Parameter( torch.randn(self.in_features) ,requires_grad=True)
        self.bn0 = nn.BatchNorm1d(self.in_features)
        self.fc1 = nn.Linear(self.in_features, 100 * self.in_features)
        self.bn1 = nn.BatchNorm1d(100 * self.in_features)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(100 * self.in_features, 10*self.in_features)
        self.bn2 = nn.BatchNorm1d(10*self.in_features)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(10* self.in_features, 100*self.in_features)
        self.bn3 = nn.BatchNorm1d( 100 * self.in_features )
        self.activation3 = nn.ReLU()
        self.fc4 = nn.Linear( 100 * self.in_features,  self.in_features )
        self.bn4 = nn.BatchNorm1d( self.in_features )
        self.activation4 = nn.ReLU()
        # self.fc6 = nn.Linear(self.in_features, 100 * self.in_features)
        # self.bn6 = nn.BatchNorm1d(100 * self.in_features)
        # self.activation6 = nn.ReLU()
        # self.fc7 = nn.Linear(100 * self.in_features, 10*self.in_features)
        # self.bn7 = nn.BatchNorm1d(10*self.in_features)
        # self.activation7 = nn.ReLU()
        # self.fc8 = nn.Linear(10 * self.in_features, self.in_features)
        # self.bn8 = nn.BatchNorm1d(self.in_features)
        # self.activation8 = nn.ReLU()
        self.fc5 = nn.Linear( self.in_features, self.out_features )
        self.bn5 = nn.BatchNorm1d( self.out_features )
        self.softmax = nn.Softmax(1)
        self.model = nn.Sequential(self.bn0,self.fc1, nn.Dropout(0.2), self.bn1,self.activation1,  self.fc2, nn.Dropout(0.2), self.bn2, self.activation2,
                                   self.fc3,nn.Dropout(0.2),self.bn3,self.activation3,self.fc4,nn.Dropout(0.2),self.bn4,self.activation4,
                                #    self.fc6,nn.Dropout(0.2),self.bn6,self.activation6,self.fc7,nn.Dropout(0.2),self.bn7,self.activation7,self.fc8,nn.Dropout(0.2),self.bn8,self.activation8,
                                   self.fc5,nn.Dropout(0.2),self.bn5,
                                   self.softmax)

    def forward(self, x):
        return self.model(self.attn1*x)


class Fake1DAttention(nn.Module):

    def __init__(self, in_features, out_features):
        """Fake 1D attention model as reference.

        Args:
            in_features (int): input size for feature dimension
            out_features (int): output size for feature dimension
        """
        super().__init__()

        self.attn1 = nn.parameter.Parameter(torch.randn(in_features), requires_grad=True)
        self.fc1 = nn.Linear(in_features, in_features)
        self.activation1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(in_features)

        self.fc2 = nn.Linear(in_features, in_features)
        self.activation2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(in_features)

        self.fc3 = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(1)

        self.model = nn.Sequential(
            self.fc1, self.activation1,  # self.bn1,
            self.fc2, self.activation2,  # self.bn2,
            self.fc3, self.softmax
        )

    def forward(self, x):
        return self.model(self.attn1 * x)


if __name__ == '__main__':
    batch, in_features, out_features = 10, 10, 5
    model = FCModel(in_features, out_features)
    x = torch.randn(batch, in_features)
    print(model(x).shape)
