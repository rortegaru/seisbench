import torch.nn as nn
from seisbench.models import SeisBenchModel

class ConvNQ(SeisBenchModel):
    def __init__(self, citation=None, num_classes=10, regularization=0.01):
        super().__init__(citation)
        self.num_classes = num_classes
        self.regularization = regularization

        # Definición de las 8 capas convolucionales
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        # Capa totalmente conectada
        self.fc1 = nn.Linear(32 * 25, num_classes)  # Ajusta el tamaño según sea necesario

    def forward(self, x):
        # Pasar por las capas convolucionales con activación ReLU
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x = nn.ReLU()(self.conv7(x))
        x = nn.ReLU()(self.conv8(x))

        # Aplanar para la capa totalmente conectada
        x = x.view(x.size(0), -1)
        
        # Logits de salida
        logits = self.fc1(x)

        return logits

    def get_model_args(self):
        # Retornar los parámetros del modelo
        return {"citation": self._citation, "num_classes": self.num_classes, "regularization": self.regularization}

    def loss_function(self, logits, targets):
        # Configuración de la función de pérdida
        loss = nn.CrossEntropyLoss()(logits, targets)
        reg_loss = self.regularization * sum(torch.sum(param ** 2) for param in self.parameters())
        return loss + reg_loss

    def accuracy(self, logits, targets):
        # Cálculo de precisión
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy
