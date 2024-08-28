import torch
import numpy as np
import torch.nn as nn
from seisbench.models import SeisBenchModel



class ConvNQ(SeisBenchModel):
    """
    ConvNQ
    
    This module defines the `ConvNQ` model, which inherits from the `SeisBenchModel` class.
    The model is designed for tasks in seismology and uses a convolutional neural network (CNN)
    architecture.
 
    Structure of the `ConvNQ` model:
    - **Convolutional Layers**: The architecture consists of several convolutional layers,
      each followed by non-linear activation functions (e.g., ReLU) and pooling layers to reduce
      the dimensionality.
    - **Normalization Layers**: Batch normalization layers are included to stabilize the learning
      process and improve generalization.
    - **Output Layer**: The final layer of the model is a dense layer that produces the required
      output, whether for classification, regression, or another specific seismological task.
    - **Optimization**: The model is optimized for working with seismic data and is trained
      through backpropagation using an adaptive optimizer (e.g., Adam).
    
    This design allows the model to capture complex patterns in seismic data,
    such as seismic wave signals, improving the accuracy of predictions.
    
    """
    def __init__(self, citation=None, num_classes=10, regularization=0.01, input_length=3001):
        # Define el dispositivo: usa GPU si está disponible, de lo contrario usa CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Mueve el modelo al dispositivo correcto
#        self.to(self.device)
        super().__init__(citation)
        self.num_classes = num_classes
        self.regularization = regularization

        # Define convolutional layers with BatchNorm for better training stability
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=2, padding=1),  # 3 canales de entrada
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Cálculo del tamaño de la salida después de las capas convolucionales
        dummy_input = torch.zeros(1, 3, input_length)  # 3 canales de entrada
        output_size = self.conv_layers(dummy_input).numel()

        # Capa completamente conectada
        self.fc1 = nn.Linear(output_size, num_classes)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv_layers(x)
        print(f"Shape after conv layers: {x.shape}")
        x = x.view(x.size(0), -1)  # Aplanado
        print(f"Shape after flattening: {x.shape}")
        logits = self.fc1(x)
        return logits

    def get_model_args(self):
        # Return model parameters
        return {"citation": self._citation, "num_classes": self.num_classes, "regularization": self.regularization}

    def loss_function(self, logits, targets):
        # Configure the loss function with regularization
        loss = nn.CrossEntropyLoss()(logits, targets)
        reg_loss = self.regularization * sum(param.norm(2) for param in self.parameters())
        return loss + reg_loss

    def accuracy(self, logits, targets):
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

    def classify(self, stream):
        # Asegúrate de que las tres trazas tengan la misma longitud
        desired_length = min(len(tr.data) for tr in stream)  # O usa la longitud que prefieras
    	# Recorta o rellena las trazas para que tengan la misma longitud
        traces_data = []
        for tr in stream:
            data = tr.data
            if len(data) > desired_length:
                data = data[:desired_length]  # Truncar si es más largo
            elif len(data) < desired_length:
                padding = np.zeros(desired_length - len(data))
                data = np.concatenate((data, padding))  # Rellenar si es más corto
            traces_data.append(data)
   
        # Combinar las tres trazas en un solo tensor de entrada
        combined_data = np.stack(traces_data, axis=0)  # Forma [3, length]
        combined_data = combined_data[np.newaxis, :]  # Añadir dimensión de batch: Forma [1, 3, length]
    
        # Procesa la traza combinada utilizando tu modelo
        classification = self.predict(combined_data)  # combined_data tiene la forma [1, 3, length]
    
        return classification

    def predict(self, data):
        self.eval()  # Cambia a modo de evaluación
        with torch.no_grad():
            data_tensor = torch.tensor(data).float().to(self.device)
            prediction = self(data_tensor)  # Usa forward en vez de self.model(data_tensor)
            prediction_list = prediction.squeeze().tolist()  # Convierte la salida a una lista
        return prediction_list
