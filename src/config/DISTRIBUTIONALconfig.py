from src.utils.load_data import number_dataset_classes


def DISTRIBUTIONALconfig(args, config):

    config['clustering'] = args.clustering
    config['robust'] = args.robust
    config['update_centroids'] = args.update_centroids
    config['update_covariance'] = args.calculate_covariance
    config['embunfreeze'] = args.emb_unfreeze
    config['loss_forget'] = args.loss_forget
    config['forbidden_labels'] = args.classes_to_remove

    config['fixed_syntetic'] = args.fixed_syntetic
    config['update_syntetic'] = args.update_syntetic
    config['syntetic_samples'] = args.syntetic_samples

    config['covariance_type'] = args.covariance_type
    config['n_pca_components'] = args.n_pca_components
    config['model_selection'] = args.model_selection

    config['icu_adapt_distributions'] = getattr(args, 'icu_adapt_distributions', False)
    config['icu_adapt_n_samples'] = getattr(args, 'icu_adapt_n_samples', 500)
    config['icu_adapt_ridge_alpha'] = getattr(args, 'icu_adapt_ridge_alpha', 1e-3)
    config['icu_adapt_pca_variance'] = getattr(args, 'icu_adapt_pca_variance', None)
    config['icu_adapt_transforms_param'] = getattr(args, 'icu_adapter_transforms_param', False)
    config['icu_adapt_extra_datasets'] = getattr(args, 'icu_adapt_extra_datasets', None) or []
    config['icu_adapt_extra_max_samples'] = getattr(args, 'icu_adapt_extra_max_samples', 1000)
    config['icu_adapt_extra_datasets'] = getattr(args, 'icu_adapt_extra_datasets', None) or []
    config['icu_adapt_current_forget'] = getattr(args, 'icu_adapt_current_forget', False)

    config['classifier_anchor_lambda'] = getattr(args, 'classifier_anchor_lambda', 0.0)

    config['logit_profile_lambda'] = getattr(args, 'logit_profile_lambda', 0.0)
    config['logit_profile_mode'] = getattr(args, 'logit_profile_mode', 'logits')

    config['grad_proj_forget_variance'] = getattr(args, 'grad_proj_forget_variance', None)
    config['grad_proj_retain_ortho'] = getattr(args, 'grad_proj_retain_ortho', False)
    config['grad_proj_layers'] = getattr(args, 'grad_proj_layers', 'embeddings')

    config['train_temperature'] = getattr(args, 'train_temperature', False)
    config['temperature_penalty_lambda'] = getattr(args, 'temperature_penalty_lambda', 1.0)

    config['forget_temperature_calibration'] = getattr(args, 'forget_temperature_calibration', False)
    config['forget_temperature_always_apply'] = getattr(args, 'forget_temperature_always_apply', True)
    config['forget_temperature_monotonic_decrease'] = getattr(args, 'forget_temperature_monotonic_decrease', False)
    config['temperature_inference_only'] = getattr(args, 'temperature_inference_only', False)

    config['self_forget'] = args.self_forget
    config['forget_syntetic_samples'] = args.forget_syntetic_samples
    config['notop_margin'] = args.notop_margin
    config['lambda_margin'] = args.lambda_margin

    config['num_classes'] = number_dataset_classes(args.dataset)
    config["checkpoint_path"] = f"{config['output_base_dir']}/{config['experiment_name']}/"

    if  config['robust']:
        using_robust = "robust"
    else:
        using_robust = "standard"

    if config['clustering']:
        using_robust = using_robust + "_clustering"

    # Encode covariance type in cache path so different configs don't clash
    cov_type = config.get("covariance_type", "full")
    if cov_type != "full":
        cov_tag = f"_cov_{cov_type}"
        n_pca = config.get("n_pca_components")
        if cov_type == "pca" and n_pca is not None:
            cov_tag += f"_k{n_pca}"
        using_robust += cov_tag

    ms = config.get("model_selection", "bic")
    if ms != "bic":
        using_robust += f"_sel_{ms}"

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
    
    _loss_tag = config['loss_forget']
    if config['loss_forget'] in ('notop', 'notop_corrected_confidence_penalty', 'notop_confidence_penalty_balanced'):
        _loss_tag += f"_margin_{config['notop_margin']}"
    using_robust = using_robust + f"_forget_{_loss_tag}"

    if config.get("lambda_margin", 0.0) > 0.0:
        using_robust += f"_lambda_{config['lambda_margin']}"

    if config.get("icu_adapt_distributions", False):
        n_samp = config.get("icu_adapt_n_samples", 500)
        ridge_a = config.get("icu_adapt_ridge_alpha", 1e-3)
        pca_var = config.get("icu_adapt_pca_variance", None)
        using_robust += f"_icuadapt_n{n_samp}_r{ridge_a}"
        if pca_var is not None:
            using_robust += f"_pcavar{pca_var}"
        if config.get("icu_adapt_transforms_param", False):
            using_robust += "_analytic"
        if config.get("icu_adapt_current_forget", False):
            using_robust += "_adaptcurfgt"
        extra_ds = config.get("icu_adapt_extra_datasets", [])
        if extra_ds:
            extra_tag = "-".join(ds.lower() for ds in extra_ds)
            extra_max = config.get("icu_adapt_extra_max_samples", 1000)
            using_robust += f"_extra_{extra_tag}_m{extra_max}"

    _anchor_lam = config.get('classifier_anchor_lambda', 0.0)
    if _anchor_lam > 0.0:
        using_robust += f"_anchor{_anchor_lam}"

    _lp_lam = config.get('logit_profile_lambda', 0.0)
    if _lp_lam > 0.0:
        using_robust += f"_logitprofile_{config['logit_profile_mode']}_{_lp_lam}"

    _gp_var = config.get('grad_proj_forget_variance')
    _gp_ro = config.get('grad_proj_retain_ortho', False)
    if _gp_var is not None:
        using_robust += f"_gproj_fv{_gp_var}_{config['grad_proj_layers']}"
    elif _gp_ro:
        using_robust += f"_gproj_ro_{config['grad_proj_layers']}"

    if config.get('train_temperature', False):
        using_robust += f"_temp_lam{config['temperature_penalty_lambda']}"

    if config.get('forget_temperature_calibration', False):
        using_robust += "_ftcal"
        if not config.get('forget_temperature_always_apply', True):
            using_robust += "_noauto"
        if config.get('forget_temperature_monotonic_decrease', False):
            using_robust += "_monodec"

    if config.get('temperature_inference_only', False):
        using_robust += "_tempinferonly"

    config['checkpoint_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/DISTRIBUTIONAL_{config['syntetic_samples']}_{using_robust}/{config['version_output']}"
    config['hyperparameters_path'] = f"{config['output_base_dir']}/{config['experiment_name']}/DISTRIBUTIONAL_{config['syntetic_samples']}_{using_robust}/{config['version_output']}"

    return config
