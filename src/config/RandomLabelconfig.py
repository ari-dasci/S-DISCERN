from src.utils.load_data import number_dataset_classes


def RandomLabelconfig(args, config):

    config['num_classes'] = number_dataset_classes(args.dataset)
    config['rl_retain_fraction'] = args.rl_retain_fraction
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    config['use_forbidden_labels'] = args.use_forbidden_labels

    retain_tag = f"ret{config['rl_retain_fraction']}" if config['rl_retain_fraction'] > 0 else "noret"
    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""

    if config['use_forbidden_labels']:
        freeze_suffix += f"_forbidden"

    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"RANDOMLABEL_{retain_tag}{freeze_suffix}/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']
    
    config['forbidden_labels'] = args.classes_to_remove

    return config
