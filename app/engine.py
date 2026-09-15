import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms


class ViolenceDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)

        self.model = models.video.r3d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            if self.device.type == "cuda":
                self.model.half()

            print(f"✅ Engine carregada em: {self.device}")

        except Exception as e:
            raise Exception(f"❌ Erro ao carregar modelo: {e}")

        self.normalize = transforms.Normalize(
            mean=[0.432, 0.394, 0.376],
            std=[0.228, 0.221, 0.216]
        )

    def process_sequence(self, frames_list):
        if len(frames_list) != 16:
            return None

        frames = []
        for f in frames_list:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = torch.from_numpy(f).to(self.device)
            frames.append(f)

        clip = torch.stack(frames)
        clip = clip.permute(3, 0, 1, 2).float() / 255.0
        clip = clip.unsqueeze(0)
        clip = torch.nn.functional.interpolate(
            clip, size=(16, 224, 224), mode='trilinear', align_corners=False
        )
        clip = clip.squeeze(0)
        clip = self.normalize(clip)

        if self.device.type == "cuda":
            clip = clip.half()

        return clip.unsqueeze(0)

    def predict(self, tensor, threshold=0.75):
        if tensor is None:
            return 0, 0.0

        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.softmax(output, dim=1)
            conf = prob[0][1].item()
            pred = 1 if conf >= threshold else 0
            return pred, conf
