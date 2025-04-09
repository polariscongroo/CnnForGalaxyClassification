import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    """Gère l'entraînement et l'évaluation du modèle"""
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.class_names = ['Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 'Cigar Shaped Smooth Galaxies',
                          'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge']
        self.train_losses = []

    def train(self, epochs=3):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # Enregistrer la loss dans la liste (avant `loss.backward()`)
                running_loss += loss.item()
                self.optimizer.step()
            # Calculer la loss moyenne pour cette epoch et l'ajouter à la liste
            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)  # Ajouter à la liste des losses
            self.validate(epoch)
        self.plot_loss()
      

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}: Loss = {val_loss/len(self.val_loader):.4f}, "
              f"Val Acc = {correct/total:.2%}")

    def visualize_predictions(self, num_images=9):
        self.model.eval()
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            if i >= num_images:
                break

            img, label = self.val_loader.dataset[i]
            with torch.no_grad():
                output = self.model(img.unsqueeze(0))
                _, pred = torch.max(output, 1)
                true_label = label
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"True: {self.class_names[true_label]}\n"
                         f"Pred: {self.class_names[pred.item()]}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    
    def plot_loss(self):
        plt.plot(self.train_losses)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()