import torch.nn as nn
from seisbench.models import SeisBenchModel

class ConvNQ(SeisBenchModel):
    def __init__(self, citation=None):
        super().__init__(citation)
        # Definición de la red convolucional
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 25, 50)  # Suponiendo una entrada de tamaño 50
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # Implementación del forward pass
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 32 * 25)  # Aplanar para la capa totalmente conectada
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_model_args(self):
        # Implementar este método para devolver los parámetros necesarios para guardar el modelo
        return {"citation": self._citation}
