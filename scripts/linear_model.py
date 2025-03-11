from tlkit.data.datasets.icifar_dataset import get_dataloaders
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from  tqdm import tqdm

# Which tasks to complete
num_tasks = [0,1,2,3,4,5,6,7,8,9]
tasks = ['cifar0-9',   'cifar10-19', 'cifar20-29', 'cifar30-39', 'cifar40-49',
         'cifar50-59', 'cifar60-69', 'cifar70-79', 'cifar80-89', 'cifar90-99']
tasks = [tasks[i] for i in num_tasks]

# Generating the dataloaders
dataloaders = get_dataloaders(
    targets=[[t] for t in tasks],
    data_path='/tmp/icifar_demo/data',
    epochs_until_cycle=0,
    batch_size=128,
)

dl_train, dl_val = dataloaders['train'], dataloaders['val']

# Creation of the linear model
class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        # Une seule couche linéaire
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Avec CIFAR-10, input_dim = 3 * 32 * 32 = 3072 (chaque image est aplatie en un vecteur de 3072 éléments).
input_dim = 3 * 32 * 32  # Image RGB de 32x32
output_dim = 10  # 10 classes pour classification

# Initialisation du modèle
model = SimpleLinearModel(input_dim, output_dim)


def train(model, train_loader, criterion, optimizer, task):
    model.train()
    running_loss = 0.0
    for task_idx, (inputs, labels) in tqdm(train_loader):
        if task_idx == task:
            print_loss = True
            # Aplatir les données (32, 3, 32, 32) -> (32, 3072)
            inputs = inputs.view(inputs.size(0), -1)  # -1 permet de calculer la taille automatiquement

            optimizer.zero_grad()
            outputs = model(inputs)  # Passer les données dans le modèle
            loss = criterion(outputs, labels)  # Calculer la perte
            loss.backward()  # Backward pass
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()
    if print_loss:
        print(f'Epoch {task}, Loss: {running_loss/len(train_loader)}')

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)  

def evaluate(model, val_loader, task):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Pas besoin de gradients pour l'évaluation
        for task_idx, (inputs, labels) in tqdm(val_loader):
            if task_idx == task:
                inputs = inputs.view(inputs.size(0), -1)  # Aplatir les images
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)  # Prendre la classe prédite
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy


avg_accuracy = 0

for i in range(len(num_tasks)):

    print(f'Task {num_tasks[i]} training')
    train(model, dl_train, criterion, optimizer, task = i)

    print(f'Task {num_tasks[i]} validating')
    avg_accuracy += evaluate(model, dl_val, task = i)

avg_accuracy = avg_accuracy/len(num_tasks)

print(f'Average Val Accuracy = {avg_accuracy}%')
