# MNIST

Entraînement d'un CNN (PyTorch) sur le dataset MNIST de chiffres manuscrits.

## Résultat

| Métrique | Valeur |
|---|---|
| Précision sur le jeu de test (10 000 images) | **98,95 %** |
| Époques | 3 |
| Temps d'entraînement (CPU, 4 cœurs) | ~35 s / époque |

## Architecture

`MNISTNet` — CNN simple :
```
Conv2d(1→32, 3×3) → ReLU → MaxPool 2×2      # 28×28 → 14×14
Conv2d(32→64, 3×3) → ReLU → MaxPool 2×2     # 14×14 → 7×7
Flatten → Dropout(0.25) → Linear(3136→128) → ReLU
Dropout(0.5) → Linear(128→10)
```
Entraîné avec Adam (lr = 1e-3) et CrossEntropyLoss, batch de 64,
normalisation standard de MNIST (mean 0.1307, std 0.3081).

## Utilisation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python train.py                     # 3 époques par défaut
python train.py --epochs 10 --lr 5e-4 --out mon_modele.pt
```

Le modèle est sauvegardé dans `mnist_cnn.pt` (state_dict + métadonnées :
précision, époques, arguments).

## Recharger et prédire

```python
import torch
from train import MNISTNet

ckpt = torch.load("mnist_cnn.pt", map_location="cpu", weights_only=False)
model = MNISTNet()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# x : tenseur [1, 1, 28, 28] normalisé
predite = model(x).argmax(dim=1).item()
```

## Interface graphique

```bash
python app.py                 # http://localhost:7860
python app.py --port 8080 --share   # autre port / lien public
```

Interface web **Gradio** (`app.py`) :
- canvas de dessin (gomme, gomme effacer) avec prédiction en direct + bouton **Prédire**
- chiffre prédit + probabilités des 10 classes
- prétraitement canonique MNIST appliqué au dessin : inversion si fond clair,
  recadrage sur la boîte englobante, mise à l'échelle 20×20 (proportions conservées),
  centrage par le centre de masse sur 28×28, normalisation (mean 0.1307, std 0.3081)
- canvas vide → message « Dessine un chiffre ! » au lieu d'une fausse prédiction

Vérifié par l'API Gradio : **11/11 prédictions correctes** (images MNIST brutes,
inversées et grands canvas 500×500 simulés) + gestion du canvas vide.

## Fichiers

- `train.py` — script d'entraînement (télécharge MNIST dans `data/` automatiquement)
- `app.py` — interface graphique web de prédiction (Gradio)
- `mnist_cnn.pt` — modèle entraîné (non versionné, voir `.gitignore`)
- `data/` — dataset MNIST téléchargé (non versionné)
