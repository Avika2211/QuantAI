import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
import numpy as np

# Define a Graph Neural Network (GNN) for material property prediction
class GNNMaterialPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNMaterialPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

# Simulated Graph dataset for materials (replace with real dataset)
def create_synthetic_material_data(num_samples=1000, num_features=16):
    x = torch.rand((num_samples, num_features))
    edge_index = torch.randint(0, num_samples, (2, num_samples))
    y = torch.rand((num_samples, 1))  # Material property prediction
    return torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)

# Main function
def main():
    # Create synthetic material data (GNN-friendly)
    material_data = create_synthetic_material_data()

    # Initialize the GNN model
    model = GNNMaterialPredictor(input_dim=16, hidden_dim=64, output_dim=1)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(material_data)
        loss = criterion(output, material_data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

if __name__ == "__main__":
    main()
