"""Interface graphique web pour le modèle MNIST entraîné (Gradio).

Usage :
    python app.py [--port 7860] [--model mnist_cnn.pt] [--share]

Ouvre un canvas de dessin, prédit le chiffre avec le CNN entraîné
et affiche les probabilités de chaque classe.
"""

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train import MNISTNet

# Normalisation standard de MNIST (identique à train.py)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_model(path: str) -> MNISTNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MNISTNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    acc = ckpt.get("best_acc")
    print(f"Modèle chargé depuis {path}" + (f" (précision {acc:.2%})" if acc else ""))
    return model


MODEL: MNISTNet | None = None  # chargé au démarrage dans main()


def _as_array(value) -> np.ndarray | None:
    """Normalise la sortie du canvas Gradio (dict ImageEditor ou tableau) en tableau 2D."""
    if value is None:
        return None
    if isinstance(value, dict):
        composite = value.get("composite")
        if composite is not None:
            value = composite
        else:
            # Recomposer : traits sombres par-dessus le fond clair
            bg = value.get("background")
            base = None if bg is None else np.asarray(bg)
            for layer in value.get("layers") or []:
                if layer is None:
                    continue
                lay = np.asarray(layer)
                if base is None:
                    base = lay
                elif lay.shape == base.shape:
                    base = np.minimum(base, lay)
            value = base
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr


def preprocess(img) -> torch.Tensor | None:
    """Prépare l'image du canvas au format MNIST : 28×28, chiffre blanc sur fond noir."""
    arr = _as_array(img)
    if arr is None:
        return None

    # Canvas RGBA éventuel : composer sur fond blanc
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    pil = Image.fromarray(arr.astype("uint8")).convert("L")
    arr = np.asarray(pil, dtype=np.uint8)

    # Canvas vide : aucun contraste (aucun trait)
    if int(arr.max()) - int(arr.min()) < 30:
        return None

    # MNIST : chiffre blanc (255) sur fond noir (0) → inverser si fond clair
    if arr.mean() > 127:
        arr = 255 - arr

    # Recadrer sur la zone du trait pour être robuste à la position du dessin
    ys, xs = np.where(arr > 30)
    if not len(ys):
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    arr = arr[y0 : y1 + 1, x0 : x1 + 1]

    # Prétraitement canonique MNIST :
    # 1) boîte englobante mise à l'échelle pour tenir dans 20×20 (proportions conservées)
    # 2) centrage par le centre de masse dans un carré noir 28×28 (comme le jeu MNIST)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    box = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)

    gy, gx = np.mgrid[0:new_h, 0:new_w]
    weights = box.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    cy = (gy * weights).sum() / total
    cx = (gx * weights).sum() / total
    oy, ox = int(round(13 - cy)), int(round(13 - cx))

    square = np.zeros((28, 28), dtype=np.uint8)
    dst_y0, dst_y1 = max(0, oy), min(28, oy + new_h)
    dst_x0, dst_x1 = max(0, ox), min(28, ox + new_w)
    if dst_y0 < dst_y1 and dst_x0 < dst_x1:
        sy0, sx0 = dst_y0 - oy, dst_x0 - ox
        square[dst_y0:dst_y1, dst_x0:dst_x1] = box[
            sy0 : sy0 + (dst_y1 - dst_y0), sx0 : sx0 + (dst_x1 - dst_x0)
        ]
    pil = Image.fromarray(square)

    tensor = transforms.ToTensor()(pil)                  # [1, 28, 28] 0..1
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.unsqueeze(0)                            # [1, 1, 28, 28]


def predict(img):
    """Retourne (label, dictionnaire chiffre → probabilité)."""
    if MODEL is None:
        raise RuntimeError("Modèle non chargé")
    x = preprocess(img)
    if x is None:
        return "✏️  Dessine un chiffre !", {str(i): 0.0 for i in range(10)}
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1).squeeze(0)
    top = int(probs.argmax().item())
    dist = {str(i): float(probs[i]) for i in range(10)}
    return f"**{top}**  ({probs[top]:.1%})", dist


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MNIST — reconnaissance de chiffres") as demo:
        gr.Markdown(
            "# ✍️ Reconnaissance de chiffres MNIST\n"
            "Dessine un chiffre (0–9) dans le cadre, puis clique sur **Prédire**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                canvas = gr.Sketchpad(
                    label="Dessin",
                    type="numpy",
                    image_mode="L",
                    height=280,
                    brush=gr.Brush(colors=["#000000"], default_size=18),
                    eraser=gr.Eraser(default_size=24) if hasattr(gr, "Eraser") else None,
                )
                btn = gr.Button("Prédire", variant="primary")
                clear = gr.Button("Effacer")
            with gr.Column(scale=1):
                label = gr.Markdown(label="Résultat")
                probs = gr.Label(label="Probabilités par classe", num_top_classes=10)

        btn.click(predict, inputs=canvas, outputs=[label, probs])
        canvas.change(predict, inputs=canvas, outputs=[label, probs])
        clear.click(lambda: None, outputs=canvas)

        gr.Markdown(
            "_CNN PyTorch entraîné sur MNIST — précision 98,95 % sur le jeu de test._"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interface web MNIST")
    parser.add_argument("--model", default="mnist_cnn.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="lien public gradio.live")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global MODEL
    MODEL = load_model(args.model)

    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
