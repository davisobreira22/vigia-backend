import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from torchvision import models


class ViolenceDetector:

    def __init__(self, model_path, device="cpu"):

        # ============================================================
        # DISPOSITIVO
        # ============================================================

        self.device = torch.device(device)

        print(f"🧠 Dispositivo R3D-18: {self.device}")

        # ============================================================
        # MODELO
        # ============================================================

        self.model = models.video.r3d_18(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

        # ============================================================
        # CARREGAMENTO DO CHECKPOINT
        # ============================================================

        try:

            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )

            if isinstance(checkpoint, dict):

                state_dict = checkpoint.get(
                    "model_state_dict",
                    checkpoint.get(
                        "state_dict",
                        checkpoint
                    )
                )

            else:

                state_dict = checkpoint

            self.model.load_state_dict(
                state_dict
            )

            # GARANTE QUE TODO O MODELO VAI PARA CUDA
            self.model = self.model.to(
                self.device
            )

            self.model.eval()

            # ========================================================
            # VERIFICAÇÃO DO MODELO
            # ========================================================

            first_parameter = next(
                self.model.parameters()
            )

            print(
                f"✅ Engine carregada em: "
                f"{first_parameter.device}"
            )

        except Exception as e:

            raise Exception(
                f"❌ Erro ao carregar modelo: {e}"
            )

        # ============================================================
        # NORMALIZAÇÃO
        # ============================================================

        self.mean = torch.tensor(
            [0.432, 0.394, 0.376],
            dtype=torch.float32
        ).view(
            1, 3, 1, 1
        )

        self.std = torch.tensor(
            [0.228, 0.221, 0.216],
            dtype=torch.float32
        ).view(
            1, 3, 1, 1
        )

        # ============================================================
        # COLOCA NORMALIZAÇÃO NO MESMO DISPOSITIVO
        # ============================================================

        self.mean = self.mean.to(
            self.device
        )

        self.std = self.std.to(
            self.device
        )

        print(
            f"📐 Mean: {self.mean.device}"
        )

        print(
            f"📐 Std: {self.std.device}"
        )


    # ================================================================
    # PROCESSAMENTO DA SEQUÊNCIA
    # ================================================================

    def process_sequence(
        self,
        frames_list
    ):

        if len(frames_list) != 16:

            return None

        processed_frames = []

        for frame in frames_list:

            # --------------------------------------------------------
            # BGR → RGB
            # --------------------------------------------------------

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # --------------------------------------------------------
            # NUMPY → TORCH
            # --------------------------------------------------------

            frame = torch.from_numpy(
                frame
            )

            # --------------------------------------------------------
            # HWC → CHW
            # --------------------------------------------------------

            frame = frame.permute(
                2,
                0,
                1
            )

            # --------------------------------------------------------
            # FLOAT
            # --------------------------------------------------------

            frame = frame.float()

            # --------------------------------------------------------
            # 0-255 → 0-1
            # --------------------------------------------------------

            frame = frame / 255.0

            # --------------------------------------------------------
            # COLOCA O FRAME DIRETAMENTE NA GPU
            # --------------------------------------------------------

            frame = frame.to(
                self.device,
                dtype=torch.float32
            )

            # --------------------------------------------------------
            # REDIMENSIONAMENTO
            # --------------------------------------------------------

            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            # --------------------------------------------------------
            # NORMALIZAÇÃO
            # --------------------------------------------------------

            frame = (
                frame - self.mean
            ) / self.std

            frame = frame.squeeze(0)

            processed_frames.append(
                frame
            )

        # ============================================================
        # 16 FRAMES
        # ============================================================

        clip = torch.stack(
            processed_frames
        )

        # ============================================================
        # [16, 3, 224, 224]
        # →
        # [3, 16, 224, 224]
        # ============================================================

        clip = clip.permute(
            1,
            0,
            2,
            3
        )

        # ============================================================
        # [3, 16, 224, 224]
        # →
        # [1, 3, 16, 224, 224]
        # ============================================================

        clip = clip.unsqueeze(0)

        # ============================================================
        # GARANTIA FINAL
        # ============================================================

        clip = clip.to(
            self.device,
            dtype=torch.float32
        )

        return clip


    # ================================================================
    # PREDIÇÃO
    # ================================================================

    def predict(
        self,
        tensor,
        threshold=0.35
    ):

        if tensor is None:

            return 0, 0.0

        # ============================================================
        # VERIFICA DIMENSÕES
        # ============================================================

        if tensor.ndim != 5:

            raise ValueError(
                "Tensor inválido para R3D-18: "
                f"{tuple(tensor.shape)}"
            )

        if tensor.shape[1] != 3:

            raise ValueError(
                f"Canais inválidos: "
                f"{tensor.shape[1]}. "
                f"Esperado: 3."
            )

        if tensor.shape[2] != 16:

            raise ValueError(
                f"Frames inválidos: "
                f"{tensor.shape[2]}. "
                f"Esperado: 16."
            )

        # ============================================================
        # GARANTE GPU
        # ============================================================

        tensor = tensor.to(
            self.device,
            dtype=torch.float32
        )

        # ============================================================
        # GARANTE QUE O MODELO ESTÁ NO MESMO DEVICE
        # ============================================================

        model_device = next(
            self.model.parameters()
        ).device

        if model_device != self.device:

            self.model = self.model.to(
                self.device
            )

        # ============================================================
        # INFERÊNCIA
        # ============================================================

        with torch.no_grad():

            output = self.model(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            violence_conf = probabilities[
                0,
                1
            ].item()

            prediction = (
                1
                if violence_conf >= threshold
                else 0
            )

        return (
            prediction,
            violence_conf
        )