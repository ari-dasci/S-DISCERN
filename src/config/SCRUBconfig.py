
def SCRUBconfig(args, config):

    config['scrub_alpha'] = args.scrub_alpha
    config['scrub_gamma'] = args.scrub_gamma
    config['scrub_msteps'] = args.scrub_msteps
    config['scrub_kd_T'] = args.scrub_kd_T
    config['scrub_smoothing'] = args.scrub_smoothing
    config['scrub_sstart'] = args.scrub_sstart
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"SCRUB_a{config['scrub_alpha']}_g{config['scrub_gamma']}"
        f"_m{config['scrub_msteps']}_T{config['scrub_kd_T']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    return config
