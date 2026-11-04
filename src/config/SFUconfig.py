
def SFUconfig(args, config):

    config['sfu_damping'] = args.sfu_damping
    config['sfu_scale'] = args.sfu_scale
    config['sfu_hessian_data'] = args.sfu_hessian_data
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"SFU_damping{config['sfu_damping']}_scale{config['sfu_scale']}"
        f"_hess{config['sfu_hessian_data']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    return config
