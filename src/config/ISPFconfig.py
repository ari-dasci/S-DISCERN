
def ISPFconfig(args, config):

    config['ispf_nz'] = args.ispf_nz
    config['ispf_ngf'] = args.ispf_ngf
    config['ispf_g_steps'] = args.ispf_g_steps
    config['ispf_kd_steps'] = args.ispf_kd_steps
    config['ispf_lr_g'] = args.ispf_lr_g
    config['ispf_lr_s'] = args.ispf_lr_s
    config['ispf_synthesis_bs'] = args.ispf_synthesis_bs
    config['ispf_kd_T'] = args.ispf_kd_T
    config['ispf_pf_threshold'] = args.ispf_pf_threshold
    config['ispf_adv'] = args.ispf_adv
    config['ispf_oh'] = args.ispf_oh
    config['ispf_bn'] = args.ispf_bn
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"ISPF_nz{config['ispf_nz']}_T{config['ispf_kd_T']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    return config
