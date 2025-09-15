import torch
import torch.nn as nn
from torchinfo import summary

class PiczakCNN(nn.Module):
    """
    Corrected PyTorch implementation of the CNN architecture described in the paper
    "Environmental Sound Classification with Convolutional Neural Networks" by K. Piczak.
    This version includes the ReLU activations after each convolutional layer as specified in the paper's text.
    """
    def __init__(self, num_classes, input_shape=(60, 41)):
        super(PiczakCNN, self).__init__()

        # --- First Convolutional Block (Conv + ReLU) ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=80, kernel_size=(57, 6)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(0.5)

        # --- Second Convolutional Block (Conv + ReLU) ---
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        
        self.flatten = nn.Flatten()
        
        # --- Fully Connected Layers ---
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate the input size for the first fully connected layer automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, *input_shape)
            dummy_output = self._forward_conv(dummy_input)
            fc1_input_features = dummy_output.shape[1]
            
        self.fc1 = nn.Linear(fc1_input_features, 5000)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(5000, 5000)
        self.relu2 = nn.ReLU()
        
        self.output_layer = nn.Linear(5000, num_classes)

    def _forward_conv(self, x):
        """ Helper function to pass input through convolutional layers for size calculation """
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = self.conv_block2(x)
        x = self.pool2(x)
        return self.flatten(x)

    def forward(self, x):
        # Convolutional part
        x = self.conv_block1(x) # Contains Conv1 + ReLU
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv_block2(x) # Contains Conv2 + ReLU
        x = self.pool2(x)
        
        # Flatten and apply dropout before FC layers
        x = self.flatten(x)
        x = self.dropout2(x)
        
        # Fully connected part
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    NUM_CLASSES = 10 
    INPUT_SHAPE = (60, 41) 
    
    model_piczak = PiczakCNN(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    
    input_size = (1, 2, *INPUT_SHAPE) 
    
    print("--- Corrected PiczakCNN Architecture ---")
    summary(model_piczak, 
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params"],
            verbose=1)