
def FisherForgettingconfig(args, config):

    config['ff_noise_scale'] = args.ff_noise_scale
    config['ff_alpha'] = args.ff_alpha
    config['fisher_mode'] = args.fisher_mode
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"FISHERFORGETTING_noise{config['ff_noise_scale']}_alpha{config['ff_alpha']}"
        f"_mode{config['fisher_mode']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    # Cache path for Fisher diagonal.
    # In "subtract" mode we cache the *train* Fisher (the forget one is
    # always recomputed because it changes per experiment).
    # In "retain" mode we cache the retain Fisher directly.
    if config['fisher_mode'] == 'subtract':
        config['fisher_path'] = (
            f"fishers/{config['original_name']}_{config['version']}_epoch_{config['epoch']}_train_fisher.pt"
        )

    return config
