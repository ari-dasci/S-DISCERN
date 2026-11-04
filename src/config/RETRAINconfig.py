def RETRAINconfig(args, config):
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/RETRAIN{freeze_suffix}/{config['version_output']}"
    config['hyperparameters_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/RETRAIN{freeze_suffix}/{config['version_output']}"

    return config