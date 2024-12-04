from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--tensor_board_logger",
        default=r"~/miniconda3/AMATH900/Research_Project/InverseSR/tensorboard",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--data_format",
        default="pth",
        type=str,
        choices=["pth", "nii", "img"],
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--update_latent_variables",
        default="store_true",
    )
    parser.add_argument(
        "--update_conditioning",
        default="store_true",
    )
    parser.add_argument(
        "--subject_id",
        default="019",
        type=str,
    )
    parser.add_argument(
        "--experiment_dir",
        default="empty",
        type=str,
    )
    parser.add_argument(
        "--experiment_name",
        default="log_inversed_conditions",
        type=str,
    )
    parser.add_argument(
        "--lambda_perc",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--perc_dim",
        default="axial",
        type=str,
        choices=["axial", "coronal", "sagittal"],
    )
    parser.add_argument(
        "--start_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_steps",
        default=250,
        type=int,
    )
    parser.add_argument(
        "--update_gender",
        action="store_true",
    )
    parser.add_argument(
        "--update_age",
        action="store_true",
    )
    parser.add_argument(
        "--update_ventricular",
        action="store_true",
    )
    parser.add_argument(
        "--update_brain",
        action="store_true",
    )
    parser.add_argument(
        "--corruption",
        default="None",
        type=str,
        choices=["downsample", "mask", "None"],
    )
    parser.add_argument(
        "--mask_id",
        default="0",
        type=str,
    )
    parser.add_argument(
        "--prior_every",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--ddim_num_timesteps",
        default=250,
        type=int,
    )
    parser.add_argument(
        "--downsample_factor",
        default=4,
        type=int,
        choices=[2, 4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--kernel_size",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--downsampling_loss",
        action="store_true",
    )
    parser.add_argument(
        "--mean_latent_vector",
        action="store_true",
    )
    parser.add_argument(
        "--alpha_downsampling_loss",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--downsampling_loss_factor",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--prior_after",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--n_latent_samples",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--n_samples",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--bandwidth",
        default=10,
        type=float,
    )
    parser.add_argument(
        "--ddim_eta",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--k",
        default=1,
        type=int,
    )
