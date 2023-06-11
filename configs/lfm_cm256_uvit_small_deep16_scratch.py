import ml_collections
import os


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 12340
    config.z_shape = (4, 32, 32)
    config.vis_num = 16

    config.pretrained_path = None  # imagenet256_uvit_huge.pth
    # imagenet512_uvit_large.pth

    config.autoencoder = d(pretrained_path="assets/stable-diffusion/autoencoder_kl.pth")

    config.train = d(
        n_steps=500000,
        batch_size=256,
        mode="uncond",
        log_interval=100,
        eval_interval=5000,
        save_interval=10000,
    )

    config.optimizer = d(
        name="adam",  # default adamw
        lr=1e-4,
        weight_decay=0.03,  # 0.03 default uvit
        betas=(0.9, 0.999),  # (0.99, 0.999),
    )

    config.lr_scheduler = d(name="customized", warmup_steps=0)

    config.nnet = d(
        name="uvit",
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=512,
        depth=16,  # 12->16
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,  # set it to 1001 for imagenet256, bcz we train on this pretrained weight.
        use_checkpoint=True,
    )

    config.dynamic = d(
        sigma_min=1e-4,
    )

    config.dataset = d(
        name="celebamask256_features_cond",
        path="assets/datasets/celebamask256_features_with_supervision",
        cfg=False,
        p_uncond=0.15,
    )
    config.dl = d(
        num_workers=8,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=50,  # the decoder is large
        path="samples/" + config.dataset.name,
    )

    return config
