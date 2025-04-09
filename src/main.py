from classifiecnn import Classifiercnn
from dataloader import Dataloader
from trainer import Trainer  

def main():
    print("Début du programme")
    
    try:
        # 1. Chargement des données©
        data_loader = Dataloader("data/Galaxy10_DECals.h5")
        train_loader, val_loader = data_loader.get_dataloaders()
        print("1 : Chargement des données terminé")

        # 2. Initialisation modèle
        model = Classifiercnn()
        print("2 : Initialisation modèle terminée")

        # 3. Entraînement
        trainer = Trainer(model, train_loader, val_loader)
        trainer.train(epochs=3)
        print("3 : Entraînement terminé")

        # 4. Visualisation
        trainer.visualize_predictions()
        print("4 : Visualisation terminée")

    except Exception as e:
        print(f"Erreur : {str(e)}")

if __name__ == "__main__":
    main()
    print("Fin du programme")