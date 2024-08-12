import os
from typing import Any

import torch
import sys
import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, Callback
import pytorch_lightning as pl

from torchmetrics import AUROC

from .kdad_vit import AD_ViT
from .simplenet import SimpleNet
from .general_ad import General_AD
from .load_data import prepare_loader


class CustomTQDMProgressBar(TQDMProgressBar):
    def on_validation_batch_end(self, trainer: "pl.Trainer", *_: Any) -> None:
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass


class AUROCCallback(Callback):
    def __init__(self, pos_label=1):
        super().__init__()
        self.auroc = AUROC(pos_label=pos_label)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.preds = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        logits, labels = outputs['logits'], outputs['labels']
        self.preds.append(logits)
        self.targets.append(labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.preds).cpu()
        targets = torch.cat(self.targets).cpu()
        auroc = self.auroc(preds, targets)
        pl_module.log("val_image_auroc", auroc, on_epoch=True, prog_bar=True)


def run(args):
    # device
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit()

    device = torch.device("cuda:0")
    print("Device:", device)

    # lightning set-up
    trainer = Trainer(
        log_every_n_steps=args.log_every_n_steps,
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor=f"val_{args.val_monitor}"),
            LearningRateMonitor("epoch"),
            CustomTQDMProgressBar(),
            AUROCCallback()
        ],
        enable_progress_bar=True
    )

    # data loaders
    train_loader, test_loader = prepare_loader(image_size=args.image_size,
                                                        path=args.data_dir,
                                                        dataset_name=args.dataset_name,
                                                        class_name=args.normal_class,
                                                        batch_size=args.batch_size,
                                                        test_batch_size=args.test_batch_size,
                                                        num_workers=args.num_workers,
                                                        seed=args.seed,
                                                        shots=args.shots)

    # seeding
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # train / load model
    if args.run_type == 'kdad':
        if args.load_checkpoint:
            model = AD_ViT.load_from_checkpoint(args.checkpoint_dir)
            checkpoint_dir = args.checkpoint_dir
        else:
            model = AD_ViT(embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    patch_size=args.patch_size,
                    num_channels=args.num_channels,
                    num_patches=args.num_patches,
                    dropout=args.dropout,
                    lr=args.lr,
                    hf_path=args.hf_path,
                    milestones=args.milestones,
                    gamma=args.gamma,
                    model_type=args.model_type)
            trainer.fit(model, train_loader, test_loader)
            checkpoint_dir = trainer.checkpoint_callback.best_model_path
            model = AD_ViT.load_from_checkpoint(checkpoint_dir)
    elif args.run_type == 'simplenet':
        if args.load_checkpoint:
            model = SimpleNet.load_from_checkpoint(args.checkpoint_dir)
            checkpoint_dir = args.checkpoint_dir
        else:
            model = SimpleNet(lr=args.lr,
                        lr_adaptor=args.lr_adaptor,
                        hf_path=args.hf_path,
                        layers_to_extract_from=args.layers_to_extract_from,
                        hidden_dim=args.hidden_dim,
                        wd=args.wd,
                        epochs=args.epochs,
                        noise_std=args.noise_std,
                        dsc_layers=args.dsc_layers,
                        pool_size=args.pool_size,
                        image_size=args.image_size,
                        log_pixel_metrics=args.log_pixel_metrics,
                        smoothing_sigma=args.smoothing_sigma,
                        smoothing_radius=args.smoothing_radius)
            trainer.fit(model, train_loader, test_loader)
            checkpoint_dir = trainer.checkpoint_callback.best_model_path
            model = SimpleNet.load_from_checkpoint(checkpoint_dir)
    elif args.run_type == 'general_ad':
        if args.load_checkpoint:
            model = General_AD.load_from_checkpoint(args.checkpoint_dir)
            checkpoint_dir = args.checkpoint_dir
        else:
            model = General_AD(lr=args.lr,
                        lr_decay_factor=args.lr_decay_factor,
                        hf_path=args.hf_path,
                        layers_to_extract_from=args.layers_to_extract_from,
                        hidden_dim=args.hidden_dim,
                        wd=args.wd,
                        epochs=args.epochs,
                        noise_std=args.noise_std,
                        dsc_layers=args.dsc_layers,
                        dsc_heads=args.dsc_heads,
                        dsc_dropout=args.dsc_dropout,
                        pool_size=args.pool_size,
                        image_size=args.image_size,
                        num_fake_patches=args.num_fake_patches,
                        fake_feature_type=args.fake_feature_type,
                        top_k=args.top_k,
                        log_pixel_metrics=args.log_pixel_metrics,
                        smoothing_sigma=args.smoothing_sigma,
                        smoothing_radius=args.smoothing_radius)
            trainer.fit(model, train_loader, test_loader)
            # trainer.fit(model, train_loader, None)
            checkpoint_dir = trainer.checkpoint_callback.best_model_path
            model = General_AD.load_from_checkpoint(checkpoint_dir)
            print('fin')
    else:
        print("This is not a valid method name.")
        sys.exit()

    # test
    test_result = trainer.test(model, test_loader, verbose=True)

    print("Checkpoint directory:", checkpoint_dir)

    return