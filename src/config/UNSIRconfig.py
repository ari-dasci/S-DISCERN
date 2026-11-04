
def UNSIRconfig(args, config):

    config['unsir_noise_batch_size'] = args.unsir_noise_batch_size
    config['unsir_noise_epochs'] = args.unsir_noise_epochs
    config['unsir_noise_steps'] = args.unsir_noise_steps
    config['unsir_noise_lr'] = args.unsir_noise_lr
    config['unsir_noise_reg'] = args.unsir_noise_reg
    config['unsir_num_noise_batches'] = args.unsir_num_noise_batches
    config['unsir_impair_epochs'] = args.unsir_impair_epochs
    config['unsir_impair_lr'] = args.unsir_impair_lr
    config['unsir_repair_epochs'] = args.unsir_repair_epochs
    config['unsir_repair_lr'] = args.unsir_repair_lr
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"UNSIR_impairEp{config['unsir_impair_epochs']}_repairEp{config['unsir_repair_epochs']}{freeze_suffix}"
        f"/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']

    return config
