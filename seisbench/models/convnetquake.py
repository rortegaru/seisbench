import torch
import torch.nn as nn
import torch.nn.functional as F
from seisbench.models import SeisBenchModel

class ConvNetQuake(SeisBenchModel):
    def __init__(self, inputs, config, checkpoint_dir, is_training=False):
        super(ConvNetQuake, self).__init__(citation=config.citation)
        self.is_training = is_training
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Parámetros
        c = 32  # número de canales por capa conv
        ksize = 3  # tamaño del kernel de la convolución
        depth = 8  # profundidad de la red
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Capas convolucionales
        layers = []
        in_channels = inputs.shape[1]
        for i in range(depth):
            layers.append(nn.Conv1d(in_channels, c, kernel_size=ksize, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = c
        self.conv_layers = nn.Sequential(*layers)

        # Dummy input para calcular el tamaño de la salida
        dummy_input = torch.zeros(1, inputs.shape[1], inputs.shape[2]).to(self.device)
        output_size = self.conv_layers(dummy_input).numel()

        # Capa completamente conectada
        self.fc = nn.Linear(output_size, config.n_clusters)
        
        # Inicialización
        self.to(self.device)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Aplanar
        logits = self.fc(x)
        class_prob = F.softmax(logits, dim=1)
        class_prediction = torch.argmax(class_prob, dim=1)
        return logits, class_prob, class_prediction

    def loss_function(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        reg_loss = self.config.regularization * sum(param.norm(2) for param in self.parameters())
        return loss + reg_loss

    def accuracy(self, logits, targets):
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

    def validation_metrics(self, logits, targets):
        loss = self.loss_function(logits, targets)
        det_accuracy = self.accuracy(logits, targets)
        loc_accuracy = det_accuracy  # Puede diferir si hay otras métricas de localización
        return {"loss": loss.item(), "detection_accuracy": det_accuracy.item(), "localization_accuracy": loc_accuracy.item()}

    def validation_metrics_message(self, metrics):
        return 'loss = {:.5f} | det. acc. = {:.1f}% | loc. acc. = {:.1f}%'.format(
            metrics['loss'], metrics['detection_accuracy']*100, metrics['localization_accuracy']*100)

    def _setup_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _summary_step(self, step_data):
        step = step_data['step']
        loss = step_data['loss']
        det_accuracy = step_data['detection_accuracy']
        loc_accuracy = step_data['localization_accuracy']
        duration = step_data['duration']
        avg_duration = 1000*duration/step

        if self.is_training:
            return f'Step {step} | {duration:.0f}s ({avg_duration:.0f}ms) | loss = {loss:.4f} | det. acc. = {100*det_accuracy:.1f}% | loc. acc. = {100*loc_accuracy:.1f}%'
        else:
            return f'Step {step} | {duration:.0f}s ({avg_duration:.0f}ms) | accuracy = {100*det_accuracy:.1f}% | accuracy = {100*loc_accuracy:.1f}%'

    def train_step(self, inputs, targets):
        self.train()
        logits, _, _ = self(inputs)
        loss = self.loss_function(logits, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, inputs, targets):
        self.eval()
        with torch.no_grad():
            logits, _, _ = self(inputs)
            metrics = self.validation_metrics(logits, targets)
        return metrics
