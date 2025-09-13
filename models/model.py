import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
import numpy as np

from models.base import BaseModel
from utils.post_processing import enhance_color, enhance_contrast
from torchvision.transforms import functional as TF


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        """Must to init BaseModel with kwargs."""
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)

        # Losses / criterions
        self.criterion = nn.MSELoss()  # base loss
        self.ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

        # VGG for perceptual loss
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:20].to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False  # no training for VGG

        # Loss weights (puoi sperimentare)
        self.lambda_mse = 1.0
        self.lambda_vgg = 0.25
        self.lambda_ssim = 0.5
        self.lambda_lpips = 0.5

    def composite_loss(self, outputs, targets):
        """Computes the composite loss: MSE + perceptual (VGG) + SSIM + LPIPS."""
        mse_loss = self.criterion(outputs, targets)
        vgg_loss = F.mse_loss(self.vgg(outputs), self.vgg(targets))

        # SSIM -> maggiore è meglio, quindi loss = 1 - ssim
        ssim_loss = 1 - self.ssim_metric(outputs, targets)

        # LPIPS -> è già una loss percettiva
        lpips_loss = self.lpips_metric(outputs, targets)

        # Weighted sum
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_vgg * vgg_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_lpips * lpips_loss
        )

        return total_loss

    def generate_output_images(self, outputs, save_dir):
        """Generates and saves output images to the specified directory."""
        os.makedirs(save_dir, exist_ok=True)
        for i, output_image in enumerate(outputs):
            output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)
            output_resized = TF.resize(output_image, (400, 600))

            output_path = os.path.join(save_dir, f'output_{i + 1}.png')
            output_resized.save(output_path)
        print(f'{len(outputs)} output images generated and saved to {save_dir}')

    def train_step(self):
        """Trains the model."""
        train_losses = np.zeros(self.epoch)
        best_loss = float('inf')
        self.network.to(self.device)

        for epoch in range(self.epoch):
            train_loss = 0.0
            dataloader_iter = tqdm(
                self.dataloader, desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.composite_loss(outputs, targets)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            train_loss = train_loss / len(self.dataloader)

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model(self.network)

            train_losses[epoch] = train_loss

            print(f"Epoch [{epoch + 1}/{self.epoch}] Train Loss: {train_loss:.4f}")

    def test_step(self):
        """Test the model."""
        path = os.path.join(self.model_path, self.model_name)
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

        psnr = PeakSignalNoiseRatio().to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

        with torch.no_grad():
            test_loss = 0.0
            test_psnr = 0.0
            test_ssim = 0.0
            test_lpips = 0.0
            self.network.eval()
            self.optimizer.zero_grad()
            if self.is_dataset_paired:
                for inputs, targets in tqdm(self.dataloader, desc='Testing...'):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.network(inputs)
                    if self.apply_post_processing:
                        outputs = enhance_contrast(outputs, contrast_factor=1.12)
                        outputs = enhance_color(outputs, saturation_factor=1.35)

                    loss = self.composite_loss(outputs, targets)
                    test_loss += loss.item()
                    test_psnr += psnr(outputs, targets)
                    test_ssim += ssim(outputs, targets)
                    test_lpips += lpips(outputs, targets)
            else:
                for inputs in tqdm(self.dataloader, desc='Testing...'):
                    inputs = inputs.to(self.device)
                    outputs = self.network(inputs)

            test_loss = test_loss / len(self.dataloader)
            test_psnr = test_psnr / len(self.dataloader)
            test_ssim = test_ssim / len(self.dataloader)
            test_lpips = test_lpips / len(self.dataloader)

            if self.is_dataset_paired:
                print(
                    f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}, Test LPIPS: {test_lpips:.4f}')

            self.generate_output_images(outputs, self.output_images_path)
