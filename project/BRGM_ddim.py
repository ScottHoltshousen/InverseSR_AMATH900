# Code is adpated from: https://huggingface.co/spaces/Warvito/diffusion_brain/blob/main/app.py and
# https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing#scrollTo=4XDeCy-Vj59b
# A lot of thanks to the author of the code

# Reference:
# [1] Pinaya, W. H., et al. (2022). "Brain Imaging Generation with Latent Diffusion Models." arXiv preprint arXiv:2209.07162.
# [2] Marinescu, R., et al. (2020). Bayesian Image Reconstruction using Deep Generative Models.

import math, os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import csv
import dnnlib
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import apply_transform
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models.BRGM.forward_models import (
    ForwardDownsample,
    ForwardFillMask,
    ForwardAbstract,
)
from models.ddim import DDIMSampler
from utils.transorms import get_preprocessing
from utils.plot import draw_corrupted_images, draw_images, draw_img
from utils.add_argument import add_argument
from utils.utils import (
    setup_noise_inputs,
    load_target_image,
    load_pre_trained_model,
    create_corruption_function,
    sampling_from_ddim,
    getVggFeatures,
)
from utils.const import (
    INPUT_FOLDER,
    MASK_FOLDER,
    PRETRAINED_MODEL_DDPM_PATH,
    PRETRAINED_MODEL_VGG_PATH,
    OUTPUT_FOLDER,
)

def load_vgg_perceptual(
    hparams: Namespace, target: torch.Tensor, device: torch.device
) -> Tuple[Any, torch.Tensor]:
    with open(PRETRAINED_MODEL_VGG_PATH / "vgg16.pt", "rb") as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_features = getVggFeatures(hparams, target, vgg16)
    return vgg16, target_features


def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_gender else 1
    mask_cond[:, 1] = 0 if not hparams.update_age else 1
    mask_cond[:, 2] = 0 if not hparams.update_ventricular else 1
    mask_cond[:, 3] = 0 if not hparams.update_brain else 1
    return mask_cond


def project(
    ddim: DDIMSampler,
    decoder: torch.nn.Module,
    forward: ForwardFillMask,
    target: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    verbose: bool = False,
):
    cond, latent_variable = setup_noise_inputs(device=device, hparams=hparams)

    update_params = []
    if hparams.update_latent_variables:
        latent_variable.requires_grad = True
        update_params.append(latent_variable)
    if hparams.update_conditioning:
        cond.requires_grad = True
        update_params.append(cond)

    optimizer_adam = torch.optim.Adam(
        update_params,
        betas=(0.9, 0.999),
        lr=hparams.learning_rate,
    )
    latent_variable_out = torch.zeros(
        [hparams.num_steps] + list(latent_variable.shape[1:]),
        dtype=torch.float32,
        device=device,
    )
    cond_out = torch.zeros(
        [hparams.num_steps] + list(cond.shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    mask_cond = create_mask_for_backprop(hparams, device)
    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)

    total_num_pixels = (
        target_img_corrupted.numel()
        if hparams.corruption != "mask"
        else math.prod(forward.mask.shape) - forward.mask.sum()
    )

    patience = 20
    best_ssim = 0

    for step in range(hparams.start_steps+1, hparams.num_steps+1):

        def closure():
            optimizer_adam.zero_grad()

            synth_img = sampling_from_ddim(
                ddim=ddim,
                decoder=decoder,
                latent_variable=latent_variable,
                cond=cond,
                hparams=hparams,
            )
            synth_img_corrupted = forward(synth_img)  # f(G(w))

            loss = 0
            pixelwise_loss = (
                synth_img_corrupted - target_img_corrupted
            ).abs().sum() / total_num_pixels
            loss += pixelwise_loss

            synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
            perceptual_loss = (target_features - synth_features).abs().mean()
            loss += hparams.lambda_perc * perceptual_loss

            loss.backward(create_graph=False)
            if cond.requires_grad:
                cond.grad *= mask_cond

            return (
                loss,
                pixelwise_loss,
                perceptual_loss,
                synth_img,
                synth_img_corrupted,
            )

        (
            loss,
            pixelwise_loss,
            perceptual_loss,
            synth_img,
            synth_img_corrupted,
        ) = optimizer_adam.step(closure=closure)

        synth_img_np = synth_img[0, 0].detach().cpu().numpy()
        target_np = target[0, 0].detach().cpu().numpy()
        ssim_ = ssim(
            synth_img_np,
            target_np,
            win_size=11,
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
        # Code for computing PSNR is adapted from
        # https://github.com/agis85/multimodal_brain_synthesis/blob/master/error_metrics.py#L32
        data_range = np.max([synth_img_np.max(), target_np.max()]) - np.min(
            [synth_img_np.min(), target_np.min()]
        )
        psnr_ = psnr(target_np, synth_img_np, data_range=data_range)
        mse_ = mse(target_np, synth_img_np)
        nmse_ = nmse(target_np, synth_img_np)

        writer.add_scalar("loss", loss, global_step=step)
        writer.add_scalar("pixelwise_loss", pixelwise_loss, global_step=step)
        writer.add_scalar("perceptual_loss", perceptual_loss, global_step=step)
        writer.add_scalar("ssim", ssim_, global_step=step)
        writer.add_scalar("psnr", psnr_, global_step=step)
        writer.add_scalar("mse", mse_, global_step=step)
        writer.add_scalar("nmse", nmse_, global_step=step)

        if hparams.update_conditioning:
            if hparams.update_gender:
                writer.add_scalar("inversed_gender", cond[0, 0], global_step=step)
            if hparams.update_age:
                writer.add_scalar("inversed_age", cond[0, 1], global_step=step)
            if hparams.update_ventricular:
                writer.add_scalar("inversed_ventricular", cond[0, 2], global_step=step)
            if hparams.update_brain:
                writer.add_scalar("inversed_brain", cond[0, 3], global_step=step)
        logprint(
            f"step {step:>4d}/{hparams.num_steps}: total_loss {float(loss):<5.8f} pix_loss {float(pixelwise_loss):<5.8f} perc_loss {float(perceptual_loss):<1.15f} SSIM {float(ssim_):<5.8f} PSNR {float(psnr_):<5.8f} MSE {float(mse_):<5.8f} NMSE {float(nmse_):<5.8f}",
            verbose=verbose,
        )

        step_ = f"{step}".zfill(4)

        images_dir = OUTPUT_FOLDER / "images"
        if not os.path.isdir(images_dir):
            os.makedirs(images_dir)

        if step % 10 == 0:
            draw_img(
                synth_img_np,
                title="synth",
                step=step_,
                output_folder=images_dir,
            )

        if step % 25 == 0:
            if hparams.corruption != "None":
                imgs = draw_corrupted_images(
                    synth_img_np,
                    target_np,
                    synth_img_corrupted[0, 0].detach().cpu().numpy(),
                    target_img_corrupted[0, 0].detach().cpu().numpy(),
                    ssim_=ssim_,
                )
            else:
                imgs = draw_images(
                    synth_img_np,
                    target_np,
                    ssim_=ssim_,
                )
            writer.add_figure(f"step: {step_}", imgs, global_step=step)
            plt.close(imgs)

        latent_variable_out[step-1] = latent_variable.detach()[0]
        cond_out[step-1] = cond.detach()[0]

        # Early stopping condition. save model and statistics if found new best
        if ssim_ >= best_ssim: # New best
            best_ssim = ssim_ # Update best SSIM
            patience = 20 # Reset patience
            print(f"New best model at step {step}. Saving...")

            # Save model
            torch.save(
                {
                    "epoch": step,
                    "latent_variable": latent_variable,
                    "cond": cond,
                    "optimizer": optimizer_adam.state_dict(),
                },
                OUTPUT_FOLDER / "checkpoint.pth",
            )

            header = [
                "Step",
                "SSIM",
                "PSNR",
                "MSE",
                "NMSE",
                "Gender",
                "Age",
                "Ventricular",
                "Brain Volume",
            ]

            row = [
                step,
                ssim_,
                psnr_,
                mse_,
                nmse_,
                cond[0, 0].clone().detach().item(),
                cond[0, 1].clone().detach().item() * (82 - 44) + 44,
                cond[0, 2].clone().detach().item(),
                cond[0, 3].clone().detach().item(),
                ]

            # Save the statistics result to csv file
            output_file = OUTPUT_FOLDER / "statistics.csv"
            with open(
                output_file,
                "a",
                newline=""
            ) as file:
                for h, r in zip(header, row):
                    file.write(f"{h}: {r}\n\n")
                file.write(f"END OF REPORT FOR STEP {step}\n\n")

        else: # SSIM got worse
            patience -= 1
            if patience == 0:
                print(f"Early stopping triggered at step {step}")
                break

    writer.flush()
    writer.close()

    return latent_variable_out, cond_out


def main(hparams: Namespace) -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_tensor = load_target_image(hparams, device=device)
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger)

    # Create forward corruption model that masks the image with the given mask
    # Make the mask function work
    forward = create_corruption_function(hparams=hparams, device=device)
    diffusion, decoder = load_pre_trained_model(device=device)
    ddim = DDIMSampler(diffusion)

    # Call projector code
    start_time = perf_counter()
    latent_variable_out, cond_out = project(
        ddim,
        decoder,
        writer=writer,
        hparams=hparams,
        forward=forward,
        target=img_tensor,
        device=device,
        verbose=True,
    )
    print(f"Elapsed: {(perf_counter() - start_time):.1f} s")

    torch.save(
        {"latent_variable": latent_variable_out, "cond": cond_out},
        OUTPUT_FOLDER / hparams.experiment_name / "latent_cond.pth",
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    OUTPUT_FOLDER = OUTPUT_FOLDER / hparams.experiment_dir / hparams.experiment_name
    # seed_everything(42)
    main(hparams)
