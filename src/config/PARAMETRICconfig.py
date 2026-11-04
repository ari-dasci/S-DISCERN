import argparse

from .DISTRIBUTIONALconfig import DISTRIBUTIONALconfig


# DISTRIBUTIONAL options that have no effect in PARAMETRIC (one LW-regularised
# family per class, chosen by CV).  They are reset so they do not leak into
# the output path.
_IGNORED = {
    "clustering": False,
    "robust": False,
    "covariance_type": "full",
    "n_pca_components": None,
    "model_selection": "bic",
    "icu_adapt_distributions": False,
}


def PARAMETRICconfig(args, config):
    if args.cv_folds < 2:
        raise ValueError(f"--cv-folds must be >= 2 (got {args.cv_folds})")

    p_args = argparse.Namespace(**vars(args))
    for name, default in _IGNORED.items():
        value = getattr(p_args, name, default)
        if value not in (default, None):
            print(f"⚠️  PARAMETRIC ignores --{name.replace('_', '-')}={value}")
        setattr(p_args, name, default)

    config = DISTRIBUTIONALconfig(p_args, config)
    config['cv_folds'] = args.cv_folds
    config['parametric_selection'] = getattr(args, 'parametric_selection', 'll')
    config['parametric_ks_max'] = getattr(args, 'parametric_ks_max', 0.1)
    config['parametric_subspace'] = getattr(args, 'parametric_subspace', 'none')
    config['parametric_pca_variance'] = getattr(args, 'parametric_pca_variance', 0.99)

    config['cv_repeats'] = getattr(args, 'cv_repeats', 1)
    config['parametric_tiebreak'] = getattr(args, 'parametric_tiebreak', 'ks')
    config['parametric_em_max_iter'] = getattr(args, 'parametric_em_max_iter', 20)
    config['parametric_tie_margin'] = getattr(args, 'parametric_tie_margin', 0.05)
    config['parametric_pipeline'] = getattr(args, 'parametric_pipeline', 'cv')
    if config['cv_repeats'] < 1:
        raise ValueError(f"--cv-repeats must be >= 1 (got {config['cv_repeats']})")

    # Tag shared by the output path and the dist_infos cache directory
    tag = f"cv{args.cv_folds}"
    if config['cv_repeats'] > 1:
        tag += f"x{config['cv_repeats']}"
    if config['parametric_em_max_iter'] != 20:
        tag += f"_it{config['parametric_em_max_iter']}"
    if config['parametric_selection'] == 'll_ks':
        tag += f"_ks{config['parametric_ks_max']}"
    elif config['parametric_selection'] == 'll_ksnorm':
        tag += f"_kn{config['parametric_ks_max']}"
    elif config['parametric_selection'] == 'll_tiebreak':
        tag += {"ks": "_tb", "energy": "_tbe", "ks_norm": "_tbn"}[config['parametric_tiebreak']]
        if config['parametric_tie_margin'] > 0:
            tag += f"_m{config['parametric_tie_margin']}"
    elif config['parametric_selection'] == 'energy':
        tag += "_energy"
    if config['parametric_subspace'] == 'pca':
        tag += f"_pca{config['parametric_pca_variance']}"
    if config['parametric_pipeline'] == 'tests':
        tag += "_tests"
    config['parametric_tag'] = tag

    # Same layout as DISTRIBUTIONAL, under PARAMETRIC_{tag}_... instead of DISTRIBUTIONAL_...
    base = f"{config['output_base_dir']}/{config['experiment_name']}/"
    old_prefix = base + "DISTRIBUTIONAL_"
    new_prefix = base + f"PARAMETRIC_{tag}_"
    for key in ('checkpoint_path', 'hyperparameters_path'):
        assert config[key].startswith(old_prefix), config[key]
        config[key] = new_prefix + config[key][len(old_prefix):]

    return config
