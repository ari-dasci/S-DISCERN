
def SSDconfig(args, config):

    config['conservative'] = args.SSD_coefficients
    config['forget_threshold'] = 1.0 # Hyperparameter that controls the threshold for forgetting. Higher values means more aggressive unlearning.
    config['selection_weighting'] = args.SSD_alpha # Hyperparameter alpha in the paper that weights the importance of the coefficients in the selection of synapses to regularize. Higher values means more conservative unlearning.
    config['dampening_constant'] = args.SSD_lambda # Hyperparameter lambda in the paper that weights the importance of the coefficients in the dampening of the update. Higher values means more conservative unlearning.
    config['exponent'] = 1.0 # Hyperparameter that controls the exponent
    config['lower_bound'] = 1.0 # Hyperparameter that controls the lower bound of the update. Higher values means more conservative unlearning.
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    # Note: train_batch_size already in config from general_config()
    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/SSD_{config['conservative']}_lambda_{config['dampening_constant']}_alpha_{config['selection_weighting']}{freeze_suffix}/{config['version_output']}"
    config['hyperparameters_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/SSD_{config['conservative']}_lambda_{config['dampening_constant']}_alpha_{config['selection_weighting']}{freeze_suffix}/{config['version_output']}"

    return config