from src.utils.load_data import number_dataset_classes


def EWCconfig(args, config):

    config["lambda"] = args.lambda_ewc
    config["coefficients"] = args.ewc_coefficients
    config['loss_forget'] = args.loss_forget
    config['embunfreeze'] = args.emb_unfreeze
    config['num_classes'] = number_dataset_classes(args.dataset)

    config['checkpoint_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/EWC_{config['lambda']}_lambda_{config['coefficients']}_{config['embunfreeze']}_{config['loss_forget']}/{config['version_output']}"
    config['hyperparameters_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/EWC_{config['lambda']}_lambda_{config['coefficients']}_{config['embunfreeze']}_{config['loss_forget']}/{config['version_output']}"

    return config
