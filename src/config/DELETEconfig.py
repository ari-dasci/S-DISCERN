

def DELETEconfig(args, config):

    config['freeze'] = args.freeze_delete
    config['classes_to_remove'] = args.classes_to_remove

    if config['freeze'] == "freeze":
        freeze_str = "freeze"
    elif config['freeze'] == "embs":
        freeze_str = "embs"
    else:
        freeze_str = "nofreeze"

    config['checkpoint_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/DELETE_{freeze_str}/{config['version_output']}"
    config['hyperparameters_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/DELETE_{freeze_str}/{config['version_output']}"

    return config