from src.utils.load_data import number_dataset_classes




def NOISYconfig(args, config):

    config['clustering'] = args.clustering
    config['robust'] = args.robust
    # config['calculate_covariance'] = args.calculate_covariance
    config['update_centroids'] = args.update_centroids
    config['update_covariance'] = args.calculate_covariance
    config['embunfreeze'] = args.emb_unfreeze
    config['loss_forget'] = args.loss_forget
    config['forbidden_labels'] = args.classes_to_remove

    config['fixed_syntetic'] = args.fixed_syntetic
    config['update_syntetic'] = args.update_syntetic
    config['syntetic_samples'] = args.syntetic_samples

    config['num_classes'] = number_dataset_classes(args.dataset)

    config['covariance_type'] = args.covariance_type
    config['n_pca_components'] = args.n_pca_components

    config['self_forget'] = args.self_forget
    config['forget_syntetic_samples'] = args.forget_syntetic_samples

    if  config['robust']:
        using_robust = "robust"
    else:
        using_robust = "standard"

    if config['clustering']:
        using_robust = using_robust + "_clustering"

    cov_type = config.get("covariance_type", "full")
    if cov_type != "full":
        cov_tag = f"_cov_{cov_type}"
        n_pca = config.get("n_pca_components")
        if cov_type == "pca" and n_pca is not None:
            cov_tag += f"_k{n_pca}"
        using_robust += cov_tag

    if (config['update_centroids'] or config['update_covariance']) and (config["embunfreeze"] != "freeze"):
        using_robust = using_robust + "_centroids_moving"

    if config['syntetic_samples'] != -1:
        if config['fixed_syntetic']:
            using_robust = using_robust + "_syntetic"
        
            if config['update_syntetic']:
                using_robust = using_robust + "_moving"
            else:
                using_robust = using_robust + "_fixed"
        
    
    if config["embunfreeze"] == "embs":
        using_robust += "_embs"
    elif config["embunfreeze"] == "freeze":
        using_robust += "_freeze"

    else: 
        using_robust += "_nofreeze"

    if config["self_forget"]:
        using_robust += f"_self_forget_{config['forget_syntetic_samples']}"
    
    using_robust = using_robust + f"_forget_{config['loss_forget']}"

    using_robust = using_robust

    config['checkpoint_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/NOISY_{config['syntetic_samples']}_{using_robust}/{config['version_output']}"
    config['hyperparameters_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/NOISY_{config['syntetic_samples']}_{using_robust}/{config['version_output']}"

    

    return config