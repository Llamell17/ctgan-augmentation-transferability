import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

import os
import random

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def fit_and_sample_ctgan(df_in,
                         n_sample,
                         seed,
                         chart=False,
                         save_dir=None,
                         save_model=False,
                         model_name="ctgan_model",
                         save_metadata_json=False,
                         **kwargs):
    """
    Augmentate df_in in n_sample and graph the discriminator and generator loss.
    Return a DataFrame with the augmentated data and optionally plot the generator/discriminator loss if chart=True.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_in)
    metadata.update_column("W", sdtype="numerical", computer_representation="Int64")
    metadata.update_column("E", sdtype="numerical", computer_representation="Int64")

    seed_everything(seed)

    # Synthetizer
    synthesizer = CTGANSynthesizer(
        metadata,
        **kwargs,
        verbose=True
    )
    synthesizer.fit(df_in)

    # Save model
    if save_model and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.zip")
        synthesizer.save(model_path)
        if save_metadata_json:
            meta_json = os.path.join(save_dir, f"{model_name}_metadata.json")
            synthesizer.get_metadata().save_to_json(meta_json)

    # Synthetic data
    synt_data = synthesizer.sample(num_rows=n_sample)

    # Chart
    if chart:
        fig = synthesizer.get_loss_values_plot()
        fig.show()

    return synt_data