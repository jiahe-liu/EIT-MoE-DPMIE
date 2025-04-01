import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define capsule network layer
class CapsuleLayer(nn.Module):
    def __init__(self, num_input_capsules, num_output_capsules, input_dim, output_dim):
        super(CapsuleLayer, self).__init__()
        self.num_input_capsules = num_input_capsules
        self.num_output_capsules = num_output_capsules
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Weight matrix for output capsules
        self.W = nn.Parameter(torch.randn(num_output_capsules, num_input_capsules, input_dim, output_dim))

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm / (1 + norm**2)) * (x / (norm + 1e-8))  # add epsilon to prevent NaN

    def forward(self, x):
        # x shape: [batch_size, num_input_capsules, input_dim]
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # [batch_size, 1, num_input_capsules, input_dim]
        x = x.expand(-1, self.num_output_capsules, -1, -1)  # [batch_size, num_output_capsules, num_input_capsules, input_dim]
        W = self.W.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, num_output_capsules, num_input_capsules, input_dim, output_dim]
        u = torch.matmul(x.unsqueeze(-2), W).squeeze(-2)  # [batch_size, num_output_capsules, num_input_capsules, output_dim]
        u_hat = self.squash(u)

        # Dynamic routing
        b = torch.zeros(batch_size, self.num_output_capsules, self.num_input_capsules).to(x.device)
        for i in range(3):
            c = F.softmax(b, dim=1)  # Routing weights
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)  # Weighted sum
            v = self.squash(s)  # Output capsules
            if i < 2:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)

        return v


# Define gating network using capsule layer
class GatingNetwork(nn.Module):
    def __init__(self, input_dim=256, num_experts=4):
        super(GatingNetwork, self).__init__()
        self.input_fc = nn.Linear(input_dim, 256)
        self.input_bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.capsule_layer = CapsuleLayer(num_input_capsules=1, num_output_capsules=num_experts, input_dim=256, output_dim=16)
        self.output_fc = nn.Linear(16, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = self.relu(x)
        x = x.unsqueeze(1)  # Add capsule dimension
        x = self.capsule_layer(x)
        x = self.output_fc(x)
        x = self.softmax(x)
        return x


# Training function
def train_gating_network(gating_network, x_train, train_labels, x_valid, valid_labels):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gating_network.parameters(), lr=0.001, weight_decay=1e-6)

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    train_labels = torch.tensor(np.argmax(train_labels, axis=1), dtype=torch.long)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    valid_labels = torch.tensor(np.argmax(valid_labels, axis=1), dtype=torch.long)

    for epoch in range(2000):  # 2000 epochs is usually enough
        gating_network.train()
        optimizer.zero_grad()

        output = gating_network(x_train)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        if epoch % 200 == 0:
            gating_network.eval()
            with torch.no_grad():
                val_output = gating_network(x_valid)
                val_loss = criterion(val_output, valid_labels)
                print(f'Validation Loss: {val_loss.item():.4f}')

    return gating_network
