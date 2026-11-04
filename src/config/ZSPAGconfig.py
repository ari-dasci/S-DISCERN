def ZSPAGconfig(args, config):
    """Config builder for ZS-PAG unlearning method.

    Reads all zspag_* arguments from the parsed CLI args and injects them into
    the shared config dict.  Also builds the checkpoint/hyperparameter path
    following the naming convention used by all other methods.
    """
    config['zspag_pgd_eps']      = args.zspag_pgd_eps
    config['zspag_pgd_alpha']    = args.zspag_pgd_alpha
    config['zspag_pgd_steps']    = args.zspag_pgd_steps
    config['zspag_n_adv']        = args.zspag_n_adv
    config['zspag_pl_mode']      = args.zspag_pl_mode
    config['zspag_pl_steps']     = args.zspag_pl_steps
    config['zspag_pl_lr']        = args.zspag_pl_lr
    config['zspag_pl_lambda']    = args.zspag_pl_lambda
    config['zspag_sv_threshold'] = args.zspag_sv_threshold
    config['freeze']             = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"ZSPAG_eps{config['zspag_pgd_eps']}_n{config['zspag_n_adv']}"
        f"_pl{config['zspag_pl_mode']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    return config
