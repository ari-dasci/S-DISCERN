
from src.utils.load_data import number_dataset_classes


def BadTeacherconfig(args, config):

    config['bt_temperature'] = args.bt_temperature
    config['bt_use_retain'] = args.bt_use_retain
    config['freeze'] = getattr(args, 'freeze', 'nofreeze')

    retain_tag = "withRetain" if config['bt_use_retain'] else "forgetOnly"
    freeze_suffix = f"_{config['freeze']}" if config['freeze'] != 'nofreeze' else ""
    config['checkpoint_path'] = (
        f"{config['output_base_dir']}/{config['experiment_name']}/"
        f"BADTEACHER_T{config['bt_temperature']}_{retain_tag}{freeze_suffix}/{config['version_output']}"
    )
    config['hyperparameters_path'] = config['checkpoint_path']
    config['num_classes'] = number_dataset_classes(args.dataset)

    return config
