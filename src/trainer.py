import matplotlib.pyplot as plt
from numpy import copy
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
        # Stockage des métriques
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # Enregistrer la loss dans la liste (avant `loss.backward()`)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calcul des métriques d'entraînement
            train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Phase de validation
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Affichage progressif
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}')

        self.plot_metrics()
        self.visualize_predictions()

        # Restaure le meilleur modèle
        self.model.load_state_dict(self.best_model_weights)
      

    def validate(self):
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
        
        return val_loss / len(self.val_loader), correct / total
    
    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        # Graphique des losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Graphique des accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_predictions(self, num_images=9):
        # Votre code existant amélioré
        self.model.eval()
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        images_shown = 0
        
        for images, labels in self.val_loader:
            with torch.no_grad():
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
            
            for img, true_label, pred_label in zip(images, labels, preds):
                if images_shown >= num_images:
                    break
                    
                ax = axes[images_shown // 3, images_shown % 3]
                img = img.permute(1, 2, 0).cpu().numpy()
                # Normalisation pour l'affichage
                img = (img - img.min()) / (img.max() - img.min())
                
                ax.imshow(img)
                ax.set_title(f'True: {self.class_names[true_label]}\n'
                              f'Pred: {self.class_names[pred_label]}',
                              color='green' if true_label == pred_label else 'red')
                ax.axis('off')
                images_shown += 1
            
            if images_shown >= num_images:
                break
        
        plt.tight_layout()
        plt.show()