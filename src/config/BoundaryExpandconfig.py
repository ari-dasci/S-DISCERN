
def BoundaryExpandconfig(args, config):

    config['boundary_lr'] = args.boundary_lr
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"BOUNDARYEXPAND_lr{config['boundary_lr']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    return config
