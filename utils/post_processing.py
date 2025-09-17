import torch
import torch.nn.functional as F


def enhance_contrast(images, contrast_factor=1.1):
    """
    Enhance the contrast of the input images.
    """
    if images.max() > 1.0:
        images = images / 255.0

    mean_intensity = images.mean(dim=(2, 3), keepdim=True)
    enhanced_images = (images - mean_intensity) * contrast_factor + mean_intensity
    enhanced_images = torch.clamp(enhanced_images, 0.0, 1.0)
    return enhanced_images


def enhance_color(images, saturation_factor=1.1):
    """
    Enhance the color (saturation) of the input images.
    """
    if images.max() > 1.0:
        images = images / 255.0

    grayscale = 0.2989 * images[:, 0, :, :] + 0.5870 * images[:, 1, :, :] + 0.1140 * images[:, 2, :, :]
    grayscale = grayscale.unsqueeze(1)

    enhanced_images = grayscale + saturation_factor * (images - grayscale)
    enhanced_images = torch.clamp(enhanced_images, 0.0, 1.0)
    return enhanced_images


def sharpen(images, strength=0.5):
    """
    Apply a sharpening filter to the input images.
    """
    if images.max() > 1.0:
        images = images / 255.0

    kernel = torch.tensor(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=torch.float32, device=images.device
    ).unsqueeze(0).unsqueeze(0)

    kernel = kernel * strength + torch.eye(3, device=images.device).unsqueeze(0).unsqueeze(0)
    kernel = kernel / kernel.sum()

    # Apply to each channel separately
    kernel = kernel.repeat(images.shape[1], 1, 1, 1)
    sharpened = F.conv2d(images, kernel, padding=1, groups=images.shape[1])
    sharpened = torch.clamp(sharpened, 0.0, 1.0)
    return sharpened


def soft_denoise(images, sigma=0.2):
    """
    Apply a very light Gaussian-like blur for denoising.
    sigma controls the blending: 0 = no blur, 1 = full blur
    """
    if images.max() > 1.0:
        images = images / 255.0

    kernel = torch.tensor(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],
        dtype=torch.float32, device=images.device
    ).unsqueeze(0).unsqueeze(0)
    kernel = kernel / kernel.sum()

    kernel = kernel.repeat(images.shape[1], 1, 1, 1)
    blurred = F.conv2d(images, kernel, padding=1, groups=images.shape[1])
    blended = (1 - sigma) * images + sigma * blurred
    blended = torch.clamp(blended, 0.0, 1.0)
    return blended