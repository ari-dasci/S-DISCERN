import argparse
import os
from src.utils.Constants import IMPLEMENTED_METHODS, VALID_DATASETS
from src.utils.load_data import number_dataset_classes


def _validate_dataset(value):
    """Convert dataset name to uppercase and validate it."""
    value_upper = value.upper()
    if value_upper not in VALID_DATASETS:
        raise argparse.ArgumentTypeError(f"Invalid dataset '{value}'. Valid options are: {', '.join(VALID_DATASETS)}")
    return value_upper

# Contains every argument that could be eventually used in the unlearning process or training.
class Parser():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Script to train our models')


        # Probably all general arguments that are used in all methods, like data path, output path, seed, fold split, experiment name, version, epoch, base model, batch size for training and validation, etc.
        self.parser.add_argument('-o', '--output-base-dir', type=str, required=True, help='Directory where all models are stored')
        self.parser.add_argument('-e', '--experiment-name', type=str, required=True, help='Name of the experiment used to reference it in the future')
        self.parser.add_argument('-m', '--base-model', type=str, required=True, help='Model to use')
        self.parser.add_argument('-ep','--epoch', default='best', help='Checkpoint to load: "best" for checkpoint_best.pt (default), "succed" for checkpoint_succed.pt, or an epoch number for checkpoint_epoch_XX.pt')
        self.parser.add_argument('-v', '--version', default='original', help='If provided, the training will continue from that training version')
        self.parser.add_argument('--save-epoch', type=int, default=5, help='Save the model in the current epoch each X epochs')
        self.parser.add_argument('--save-best-only', action=argparse.BooleanOptionalAction, default=True,
                                 help='If enabled (default), only save the best model on validation. '
                                      'Use --no-save-best-only to save every --save-epoch epochs instead.')
        self.parser.add_argument('--max-epoch', type=int, default=30, help='Maximum number of epochs to train')
        self.parser.add_argument('-d', '--data-path', required=True, type=str, help='Root directory for dataset storage (e.g. ./data/)')
        self.parser.add_argument('--dataset', required=True, type=_validate_dataset, help='Dataset to use for training/unlearning (case-insensitive: CIFAR10, CIFAR100, MNIST, DTD, LETTERS)')
        self.parser.add_argument('--extract-dataset', required=False, default=None, type=_validate_dataset, help='Dataset used for feature extraction; defaults to --dataset if not provided')
        self.parser.add_argument('--data-mode', type=str, default="train", help='Chose which type of unlearn do')

        # When extra_layer is enabled it will add an extra fully connected layer between the predefined architecture and the new classifier head. 
        self.parser.add_argument('--extra-layer', action=argparse.BooleanOptionalAction)
        self.parser.add_argument('--embedding-size', type=int, default=256, help='Size of the embedding layer when extra_layer is enabled. Ignored otherwise.')
        self.parser.add_argument('--pretrained-model', action=argparse.BooleanOptionalAction, default=True, help='If enabled (default), it will use a pretrained model as backbone. Use --no-pretrained-model to use random initialization instead.')

        self.parser.add_argument('--seed', type=int, required=False, default=42, help='Seed for training and unlearning')
        self.parser.add_argument('--train-batch-size', type=int, required=True, help='Batch size in training phase')
        self.parser.add_argument('--val-batch-size', type=int, required=False, default=64, help='Batch size in inference phase')

        self.parser.add_argument('--unlearning-method', type=str, required=False, choices=IMPLEMENTED_METHODS, help='The implemented unlearning method to use. Choose from: ' + ', '.join(IMPLEMENTED_METHODS), default="confidence_penalty")

        # In case of data-mode == CR. These arguments specify which classes to remove from the training data.
        self.parser.add_argument('--classes-to-remove', nargs="+", type=int, default=None, help='List of classes to remove from training data')

        # In case of data-mode == backdoor. These arguments specify which classes to attack with backdoors, the percentage of samples to poison for each class and the objective class for each attack.
        # Rigth now only support one class attack another class not more than one
        self.parser.add_argument("--backdoors", nargs="+", type=int, help="List of datasets with backdoors", default=None)
        self.parser.add_argument('--percentages-backdoors',  nargs="+", type=float, help='List of percentages of backdoors for each dataset', default=None)
        self.parser.add_argument('--objectives-backdoors', nargs="+", type=int, help="Objective class for backdoor attacks (-1 means all classes)", default=None)

        # Common argument
        self.parser.add_argument('--emb-unfreeze', type=str, choices=["freeze", "embs", "nofreeze"], default="freeze",required=False, help='If enabled, it will unfreeze embbedings on training')
        self.parser.add_argument('--freeze', type=str, choices=["freeze", "embs", "nofreeze"], default="nofreeze", required=False, help='Controls which layers to freeze during unlearning. "freeze" freezes all layers except the classifier, "embs" freezes all layers except embeddings and classifier, "nofreeze" freezes nothing (default).')
        self.parser.add_argument('--loss-forget', type=str, default="confidence_penalty", help='Indicates the loss to use for forget samples.')


        # Noisy and Distributional arguments
        self.parser.add_argument('--robust', action=argparse.BooleanOptionalAction, required=False, help='If enabled, it will use robust Mahalanobis distance')
        self.parser.add_argument('--syntetic-samples', type=int, default=-1, help='Number of syntetic samples to create')
        self.parser.add_argument('--clustering', action=argparse.BooleanOptionalAction, required=False, help='If enabled, it will use robust Mahalanobis distance')
        self.parser.add_argument('--calculate-covariance', action=argparse.BooleanOptionalAction, required=False, help='If enabled, it will update the covariance matrix before unlearning. Incompatible with robust.')
        self.parser.add_argument('--update-centroids', action=argparse.BooleanOptionalAction, required=False, help='If enabled, it will update the centroids on after each training step. Incompatible with freezing the embeddings.')
        self.parser.add_argument('--fixed-syntetic', action=argparse.BooleanOptionalAction, required=False, help='If enabled, it will use fixed syntetic samples during training. Must be used with --syntetic-samples > 0. Incompatible with updating centroids.')
        self.parser.add_argument('--update-syntetic', action=argparse.BooleanOptionalAction, required=False, help='If enabled, it will update syntetic samples before each training step. Must be used with --syntetic-samples > 0. Incompatible with updateing centroids.')

        # Self-forget is a special mode that can be used in combination with Noisy or Distributional. When enabled, the model will use syntetic samples to forget the class instead of using the real forget samples. This is useful in scenarios where we want to forget a class without having access to the original training data (e.g. backdoor attacks). It is not compatible with backdoor mode because in that case we need to use the real forget samples to unlearn the backdoor attack.
        self.parser.add_argument('--self-forget', action=argparse.BooleanOptionalAction, required=False, default=False, help='If enabled the model will use syntetic samples to forget the class, not compatible with backdoor mode')
        self.parser.add_argument('--forget-syntetic-samples', type=int, default=500, help='Number of syntetic samples to create')
        
        self.parser.add_argument('--covariance-type', type=str, default='full',
                                 choices=['full', 'diagonal', 'spherical', 'lasso', 'ledoit_wolf', 'pca'],
                                 help='Structural constraint on Gaussian/t-Student covariance. '
                                      '"diagonal" keeps only per-dimension variances; '
                                      '"spherical" uses a single shared variance; '
                                      '"lasso" uses Graphical Lasso (L1-sparse precision '
                                      'matrix, cross-validated alpha); '
                                      '"ledoit_wolf" uses Ledoit-Wolf optimal shrinkage '
                                      'estimator (well-conditioned, no CV needed); '
                                      '"pca" retains the top-k eigendirections plus isotropic noise '
                                      '(dimensionality is NOT reduced). Default: full.')
        self.parser.add_argument('--n-pca-components', type=int, default=None,
                                 help='Number of PCA components when --covariance-type=pca. '
                                      'Auto-selected to capture 95%% variance when omitted.')

        # ICU distribution adaptation (DISTRIBUTIONAL method only)
        self.parser.add_argument('--icu-adapt-distributions', action=argparse.BooleanOptionalAction,
                                 default=False,
                                 help='If enabled, adapt fitted distributions after each epoch '
                                      'during ICU unlearning to compensate for embedding drift. '
                                      'Only effective when --emb-unfreeze != freeze.')
        self.parser.add_argument('--icu-adapt-n-samples', type=int, default=500,
                                 help='Fixed samples generated per retain class for ICU '
                                      'distribution adaptation (default: 500).')
        self.parser.add_argument('--icu-adapt-ridge-alpha', type=float, default=1e-3,
                                 help='Ridge regularization for the affine transport map '
                                      'estimated during ICU distribution adaptation (default: 1e-3).')
        self.parser.add_argument('--icu-adapt-extra-datasets', nargs='*', default=None,
                                 help='Optional list of extra dataset names (e.g. CIFAR10 DTD) '
                                      'whose training split is used as additional data for '
                                      'estimating the rigid motion in ICU distribution adaptation.')
        self.parser.add_argument('--icu-adapt-extra-max-samples', type=int, default=1000,
                                 help='Maximum number of samples randomly drawn from each extra '
                                      'dataset per ICU step (default: 1000).')
        self.parser.add_argument('--icu-adapt-pca-variance', type=float, default=None,
                                 help='If set (e.g. 0.9), reduce embeddings via PCA keeping enough '
                                      'components to explain this fraction of variance before fitting '
                                      'the affine transport map. Improves conditioning by discarding '
                                      'uninformative embedding dimensions. Default: None (no PCA).')
        self.parser.add_argument('--icu-adapter-transforms-param', action='store_true', default=False,
                                 help='If enabled, propagate distribution parameters analytically '
                                      'under the affine transport map (closed-form, no sampling for '
                                      'refitting). Each family has an exact or near-exact formula: '
                                      'Gaussian/t-Student: mu\'=Rmu+t, Sigma\'=RSigmaR^T; '
                                      'Laplace: scale propagated via |R|; '
                                      'vMF: direction via R*mu/||R*mu||, kappa*=sigma_min(R). '
                                      'Default: False (use sampling-based refit).')
        self.parser.add_argument('--icu-adapt-current-forget', action=argparse.BooleanOptionalAction,
                                 default=False,
                                 help='If enabled, also adapt the distribution of the class being '
                                      'forgotten in the current ICU step (in addition to retain '
                                      'classes). Classes forgotten in previous ICU steps stay frozen '
                                      'regardless of this flag. Only effective together with '
                                      '--icu-adapt-distributions. Default: False.')

        # Distributional specific arguments (not used in Noisy config)
        self.parser.add_argument('--model-selection', type=str, default='bic',
                                 choices=['bic', 'aic', 'aicc'],
                                 help='Information criterion for distribution/cluster '
                                      'selection. "aicc" is corrected AIC, recommended '
                                      'when n/d ratio is small. Default: bic.')

        # PARAMETRIC: per-class family chosen by K-fold CV log-likelihood
        self.parser.add_argument('--cv-folds', type=int, default=5,
                                 help='PARAMETRIC only: number of folds of the per-class '
                                      'cross-validation used to choose among Gaussian, '
                                      't-Student, elliptical Laplace, vMF+norm and '
                                      'tangent-normal ("Kent")+norm. Default: 5.')
        self.parser.add_argument('--parametric-selection', type=str, default='ll',
                                 choices=['ll', 'll_ks', 'll_ksnorm', 'll_tiebreak', 'energy'],
                                 help='PARAMETRIC only. "ll": highest mean CV log-likelihood. '
                                      '"ll_ks": same, but only among families whose samples '
                                      'pass a KS test on ||x - mean|| against the validation '
                                      'fold (mean D <= --parametric-ks-max + KS noise at 5%%). '
                                      '"ll_ksnorm": same filter but on the KS of the norms ||x|| (the shell '
                                      'the classifier sees), then highest log-likelihood. '
                                      '"ll_tiebreak": families whose CV log-likelihood is not significantly '
                                      'worse than the best (Nadeau-Bengio p > 0.05) tie; among them '
                                      'the lowest KS D wins (D within KS noise: highest ll). '
                                      '"energy": lowest mean energy score of the samples vs the '
                                      'validation fold (strictly proper, density-free). '
                                      'Default: ll.')
        self.parser.add_argument('--cv-repeats', type=int, default=1,
                                 help='PARAMETRIC: repetitions of the K-fold CV (different shuffles). '
                                      'The paired tests use all K*r folds with the corrected '
                                      'repeated k-fold t-test (more power). Default: 1.')
        self.parser.add_argument('--parametric-tiebreak', type=str, default='ks', choices=['ks', 'energy', 'ks_norm'],
                                 help='PARAMETRIC ll_tiebreak: criterion that resolves the tie: "ks" (KS of '
                                      '||x - mean||), "energy" (energy score) or "ks_norm" (KS of ||x||, '
                                      'the spherical shell). The winners under all three are always '
                                      'logged. Default: ks.')
        self.parser.add_argument('--parametric-pipeline', type=str, default='cv', choices=['cv', 'tests'],
                                 help='PARAMETRIC. "cv": every family competes in the CV selection. '
                                      '"tests": same logic as DISTRIBUTIONAL (Henze-Zirkler -> gaussian, '
                                      'shell+Rayleigh -> vMF/Kent, uniformity -> uniform) and, only if no '
                                      'test accepts, the CV selection (--parametric-selection and its '
                                      'tie-breaks) instead of the BIC duel. Default: cv.')
        self.parser.add_argument('--parametric-tie-margin', type=float, default=0.05,
                                 help='PARAMETRIC ll_tiebreak: families whose CV log-likelihood is '
                                      'within this many nats PER DIMENSION of the best also tie '
                                      '(practical equivalence), besides the non-significant ones. '
                                      '0 disables it. Default: 0.05.')
        self.parser.add_argument('--parametric-em-max-iter', type=int, default=20,
                                 help='PARAMETRIC: max EM iterations for t / Laplace; reaching it '
                                      'without converging marks the fit as degenerate (skipped). Default: 20.')
        self.parser.add_argument('--parametric-ks-max', type=float, default=0.1,
                                 help='PARAMETRIC ll_ks: tolerated KS D beyond sampling noise. Default: 0.1.')
        self.parser.add_argument('--parametric-subspace', type=str, default='none',
                                 choices=['none', 'pca'],
                                 help='PARAMETRIC only. "pca": fit the elliptical families '
                                      '(gaussian, t, Laplace) in span(mean, top PCs) plus an '
                                      'isotropic Gaussian residual; vMF/Kent always in full R^p. '
                                      'Default: none.')
        self.parser.add_argument('--parametric-pca-variance', type=float, default=0.99,
                                 help='PARAMETRIC pca: fraction of centred variance kept. Default: 0.99.')

        # Classifier weight anchor (DISTRIBUTIONAL method only)
        self.parser.add_argument('--classifier-anchor-lambda', type=float, default=0.0,
                                 help='If > 0, add a quadratic penalty on the classifier rows '
                                      'for retain classes to prevent them from drifting away '
                                      'from their initial values. Acts like EWC without Fisher '
                                      'weighting, restricted to the classifier head. '
                                      'Default: 0.0 (disabled).')

        # Logit-profile regularizer (DISTRIBUTIONAL method only)
        self.parser.add_argument('--logit-profile-lambda', type=float, default=0.0,
                                 help='If > 0, add a regularization term that pulls the '
                                      'classifier output (logits or softmax, see '
                                      '--logit-profile-mode) of synthetic retain-class '
                                      'samples towards the per-class mean profile computed '
                                      'once, in setup(), from the originally generated '
                                      'synthetic samples. Retain-only: never applied to the '
                                      'forget set. Default: 0.0 (disabled).')
        self.parser.add_argument('--logit-profile-mode', type=str, default='logits',
                                 choices=['logits', 'softmax'],
                                 help='Whether the logit-profile regularizer (see '
                                      '--logit-profile-lambda) compares raw logits or '
                                      'softmax probabilities against the reference '
                                      'per-class profile. Default: logits.')

        # Gradient subspace projection (DISTRIBUTIONAL method only)
        self.parser.add_argument('--grad-proj-forget-variance', type=float, default=None,
                                 help='If set (0–1), project gradients onto the top-k singular '
                                      'vectors of forget embeddings that capture this fraction of '
                                      'variance. Only the forget-specific subspace is updated. '
                                      'Mutually exclusive with --grad-proj-retain-ortho. '
                                      'Default: None (disabled).')
        self.parser.add_argument('--grad-proj-retain-ortho', action='store_true', default=False,
                                 help='If enabled, compute the retain subspace dimension at the '
                                      'classifier layer, express it as a fraction of embedding '
                                      'dimension, and use that fraction as the variance threshold '
                                      'for the forget subspace projection. '
                                      'Mutually exclusive with --grad-proj-forget-variance.')
        self.parser.add_argument('--grad-proj-layers', type=str, default='embeddings',
                                 choices=['embeddings', 'classifier', 'both'],
                                 help='Which linear layer(s) to apply gradient subspace projection '
                                      'to. "embeddings": the linear projection from backbone to '
                                      'embedding space (requires --extra-layer). "classifier": the '
                                      'final Linear(emb_dim, num_classes). "both": both. '
                                      'Default: embeddings.')

        # Learnable temperature (DISTRIBUTIONAL method only)
        self.parser.add_argument('--train-temperature', action='store_true', default=False,
                                 help='If set, the TemperatureScaling parameter appended to the '
                                      'classifier is trained during unlearning. The temperature '
                                      'divides the logits (T=1 → identity; T<1 → sharper). '
                                      'A hinge-squared penalty activates when T < 0.1. '
                                      'Default: False (temperature fixed at 1).')
        self.parser.add_argument('--temperature-penalty-lambda', type=float, default=1.0,
                                 help='Weight of the hinge-squared penalty relu(0.1 - T)^2 '
                                      'applied when the learnable temperature falls below 0.1. '
                                      'Only active when --train-temperature is set. '
                                      'Default: 1.0.')

        # Post-hoc forget-temperature calibration (DISTRIBUTIONAL method only)
        self.parser.add_argument('--forget-temperature-calibration', action='store_true', default=False,
                                 help='If set, once unlearning finishes, calibrate the classifier '
                                      'temperature so that classifier logits on freshly generated '
                                      'synthetic samples of the forgotten classes match (in a '
                                      'least-squares sense) the mean logits the model gives to the '
                                      'real held-out forgotten-class samples. Requires real forget '
                                      'data (ignored when --self-forget is set, since no real '
                                      'forgotten-class samples exist there). '
                                      'Default: False (disabled).')
        self.parser.add_argument('--forget-temperature-always-apply', action=argparse.BooleanOptionalAction, default=True,
                                 help='Only relevant with --forget-temperature-calibration. If True '
                                      '(default), the calibrated temperature stays active and scales '
                                      'every subsequent inference call (e.g. later ICU rounds). If '
                                      'False (--no-forget-temperature-always-apply), the calibrated '
                                      'value is computed and stored on the model but the classifier '
                                      "forward pass bypasses it (acts as identity) until it's "
                                      'explicitly reactivated.')
        self.parser.add_argument('--forget-temperature-monotonic-decrease', action=argparse.BooleanOptionalAction, default=False,
                                 help='Only relevant with --forget-temperature-calibration. If set, '
                                      'the newly calibrated temperature only replaces the current one '
                                      '(loaded from checkpoint_best.pt, e.g. carried over from a '
                                      'previous ICU round) when it is strictly lower — otherwise the '
                                      'current temperature is kept unchanged. Prevents a single noisy '
                                      'round from undoing a stronger correction accumulated over '
                                      'previous rounds. Default: False (always overwrite).')
        self.parser.add_argument('--temperature-inference-only', action=argparse.BooleanOptionalAction, default=False,
                                 help='If set, the classifier temperature (see '
                                      '--forget-temperature-calibration / --train-temperature) is '
                                      'only applied when the model is in eval() mode — bypassed '
                                      '(identity) whenever the model is in train() mode. This keeps '
                                      'training-time logits (synthetic conservative loss, notop, '
                                      'confidence-penalty terms, etc.) unaffected by the temperature, '
                                      'while validation/inference still sees it scaled. '
                                      'Default: False (temperature, if active, applies in both modes).')

        # EWC
        self.parser.add_argument('--lambda-ewc', type=float, default=1.0, required=False, help='Lambda value for EWC regularization')
        self.parser.add_argument('--ewc-coefficients', type=str, choices=["MAS", "FISHER"], required=False, default="FISHER", help='Method to calculate EWC coefficients')

        # SCAR

        self.parser.add_argument('--surrogate-dataset', type=str, help='Surrogate dataset to use')
        self.parser.add_argument('--scar-mode', type=str, default='HR')
        self.parser.add_argument('--scar-temperature', type=float, default=1.0)
        self.parser.add_argument('--scar-lambda-1', type=float, default=1.0)
        self.parser.add_argument('--scar-lambda-2', type=float, default=5.0)
        self.parser.add_argument('--scar-gamma1', type=float, default=5.0)
        self.parser.add_argument('--scar-gamma2', type=float, default=5.0)
        self.parser.add_argument('--scar-delta', type=float, default=5.0)
        self.parser.add_argument('--scar-num-retain-samp', type=int, default=5)
        # self.parser.add_argument('--scar-lr-unlearn', type=float, default=0.001)
        # self.parser.add_argument('--scar-wd-unlearn', type=float, default=0.0)

        # SSD
        self.parser.add_argument('--SSD-coefficients', type=str, choices=["MAS", "FISHER"], required=False, default="FISHER", help='Method to calculate SSD coefficients')
        self.parser.add_argument('--SSD-lambda', type=float, default=1.0, help='Lambda parameter in SSD')
        self.parser.add_argument('--SSD-alpha', type=float, default=10.0, help='Alpha parameter in SSD')

        # DELETE
        self.parser.add_argument('--freeze-delete', type=str, choices=["freeze", "embs", "nofreeze"], default="nofreeze", required=False, help='Indicates which layers to freeze in DELETE method. "freeze" means freezing all the layers except classifier, "embs" means freezing only the embedding layers and classifier and "nofreeze" means not freezing any layer.')

        # Retrain
        # No arguments for now

        # BadTeacher
        self.parser.add_argument('--bt-temperature', type=float, default=1.0, help='Temperature for KL divergence in BadTeacher')
        self.parser.add_argument('--bt-use-retain', action=argparse.BooleanOptionalAction, default=True,
                                 help='If enabled (default), distill with competent teacher on retain + incompetent on forget. '
                                      'If disabled (--no-bt-use-retain), only use incompetent teacher on forget data.')

        # BoundaryShrink / BoundaryExpand
        self.parser.add_argument('--boundary-eps', type=float, default=0.1,
                                 help='Epsilon (L-inf bound) for FGSM perturbation in BoundaryShrink '
                                      '(original repo default: 0.1)')
        self.parser.add_argument('--boundary-lr', type=float, default=1e-5,
                                 help='Learning rate for Boundary unlearning fine-tuning')

        # SCRUB
        self.parser.add_argument('--scrub-alpha', type=float, default=0.5,
                                 help='Weight for KD loss on retain set in SCRUB')
        self.parser.add_argument('--scrub-gamma', type=float, default=1.0,
                                 help='Weight for gradient ascent on forget set in SCRUB')
        self.parser.add_argument('--scrub-msteps', type=int, default=3,
                                 help='Number of maximize steps per minimize step in SCRUB')
        self.parser.add_argument('--scrub-kd-T', type=float, default=4.0,
                                 help='Temperature for KL divergence distillation in SCRUB')
        self.parser.add_argument('--scrub-smoothing', type=float, default=0.0,
                                 help='SWA smoothing factor (0 = disabled) in SCRUB')
        self.parser.add_argument('--scrub-sstart', type=int, default=10,
                                 help='Epoch at which SWA starts in SCRUB')

        # RandomLabel
        self.parser.add_argument('--rl-retain-fraction', type=float, default=0.0,
                                 help='Fraction of retain set to include when training with random labels')

        self.parser.add_argument('--use-forbidden-labels', action=argparse.BooleanOptionalAction, default=False,
                                 help='If enabled, it will only use a subset of random labels for the forget samples')

        # Lambda margin for separate-logits loss term
        self.parser.add_argument('--lambda-margin', type=float, default=0.0,
                                 help='Weight for the margin separation loss (loss_separate_logits). '
                                      '0.0 disables it (default). Overrides the automatic value set by random/confidence_penalty losses.')

        # Notop margin
        self.parser.add_argument('--notop-margin', type=float, default=10.0,
                                 help='Margin for the notop hinge loss (default: 10.0). '
                                      'Used when --loss-forget notop is selected.')

        # FisherForgetting
        self.parser.add_argument('--ff-noise-scale', type=float, default=1.0,
                                 help='Scale of noise injected inversely proportional to Fisher in FisherForgetting')
        self.parser.add_argument('--ff-alpha', type=float, default=1e-6,
                                 help='Damping term added to Fisher diagonal for numerical stability')
        self.parser.add_argument('--fisher-mode', type=str, default='retain',
                                 choices=['retain', 'subtract'],
                                 help='How to compute the Fisher diagonal: "retain" computes on retain set; '
                                      '"subtract" computes F_train - F_forget')

        # SFU (Source-Free Unlearning)
        self.parser.add_argument('--sfu-damping', type=float, default=1e-4,
                                 help='Damping factor for Hessian diagonal in SFU')
        self.parser.add_argument('--sfu-scale', type=float, default=1.0,
                                 help='Scale factor for Newton update step in SFU')
        self.parser.add_argument('--sfu-hessian-data', type=str, default='retain',
                                 choices=['retain', 'forget'],
                                 help='Which data split to use for Hessian estimation in SFU. '
                                      'Paper uses forget; default retain for backwards compatibility.')

        # UNSIR (Fast Machine Unlearning – Impair/Repair)
        self.parser.add_argument('--unsir-noise-batch-size', type=int, default=256,
                                 help='Batch size for noise optimisation in UNSIR')
        self.parser.add_argument('--unsir-noise-epochs', type=int, default=5,
                                 help='Epochs to optimise error-maximising noise in UNSIR')
        self.parser.add_argument('--unsir-noise-steps', type=int, default=8,
                                 help='Gradient steps per noise epoch in UNSIR')
        self.parser.add_argument('--unsir-noise-lr', type=float, default=0.1,
                                 help='Learning rate for noise optimisation in UNSIR')
        self.parser.add_argument('--unsir-noise-reg', type=float, default=0.1,
                                 help='L2 regularisation weight on noise magnitude in UNSIR')
        self.parser.add_argument('--unsir-num-noise-batches', type=int, default=20,
                                 help='Number of noise batches to generate per forget class')
        self.parser.add_argument('--unsir-impair-epochs', type=int, default=1,
                                 help='Epochs for the impair step in UNSIR')
        self.parser.add_argument('--unsir-impair-lr', type=float, default=0.02,
                                 help='Learning rate for impair step in UNSIR')
        self.parser.add_argument('--unsir-repair-epochs', type=int, default=1,
                                 help='Epochs for the repair step in UNSIR')
        self.parser.add_argument('--unsir-repair-lr', type=float, default=0.01,
                                 help='Learning rate for repair step in UNSIR')

        # ZS-PAG (Zero-Shot Proxy Adversarial data Generation)
        self.parser.add_argument('--zspag-pgd-eps', type=float, default=0.03,
                                 help='L-inf bound ε for PGD adversarial proxy generation in ZS-PAG')
        self.parser.add_argument('--zspag-pgd-alpha', type=float, default=0.003,
                                 help='PGD step size α for adversarial proxy generation in ZS-PAG '
                                      '(defaults to ε/10)')
        self.parser.add_argument('--zspag-pgd-steps', type=int, default=20,
                                 help='Number of PGD iterations for adversarial proxy generation in ZS-PAG')
        self.parser.add_argument('--zspag-n-adv', type=int, default=100,
                                 help='Maximum number of adversarial proxy samples to generate in ZS-PAG '
                                      '(paper uses 100 per remaining class)')
        self.parser.add_argument('--zspag-pl-mode', type=str, default='simple',
                                 choices=['simple', 'influence'],
                                 help='Pseudo-label computation mode in ZS-PAG. '
                                      '"simple" zeros forget-class probabilities from model predictions; '
                                      '"influence" runs the full influence-based gradient-descent optimisation '
                                      'with a diagonal-Fisher Hessian approximation.')
        self.parser.add_argument('--zspag-pl-steps', type=int, default=10,
                                 help='Gradient-descent steps for pseudo-label optimisation in ZS-PAG '
                                      '(only used when --zspag-pl-mode influence)')
        self.parser.add_argument('--zspag-pl-lr', type=float, default=0.01,
                                 help='Learning rate for pseudo-label optimisation in ZS-PAG')
        self.parser.add_argument('--zspag-pl-lambda', type=float, default=1.0,
                                 help='Weight λ of the KL-divergence regularisation in the pseudo-label '
                                      'influence objective (ZS-PAG)')
        self.parser.add_argument('--zspag-sv-threshold', type=float, default=0.99,
                                 help='Fraction of variance captured by the selected singular vectors when '
                                      'building the per-layer remaining-class subspace in ZS-PAG')

        # ISPF (Inhibited Synthetic PostFilter – Data-Free)
        self.parser.add_argument('--ispf-nz', type=int, default=256,
                                 help='Latent dimension for the generator in ISPF')
        self.parser.add_argument('--ispf-ngf', type=int, default=64,
                                 help='Base number of generator filters in ISPF')
        self.parser.add_argument('--ispf-g-steps', type=int, default=5,
                                 help='Generator update iterations per epoch in ISPF')
        self.parser.add_argument('--ispf-kd-steps', type=int, default=50,
                                 help='KD (student) update iterations per epoch in ISPF '
                                      '(original repo: ep_steps=50 with 5 rounds of kd_steps=10)')
        self.parser.add_argument('--ispf-lr-g', type=float, default=1e-3,
                                 help='Learning rate for the generator in ISPF')
        self.parser.add_argument('--ispf-lr-s', type=float, default=0.05,
                                 help='Learning rate for the student (SGD) in ISPF '
                                      '(original repo default: 0.05)')
        self.parser.add_argument('--ispf-synthesis-bs', type=int, default=256,
                                 help='Batch size for synthetic data generation in ISPF')
        self.parser.add_argument('--ispf-kd-T', type=float, default=1.0,
                                 help='Temperature for KL divergence in ISPF KD '
                                      '(original repo default: 1.0)')
        self.parser.add_argument('--ispf-pf-threshold', type=float, default=0.01,
                                 help='Probability threshold for PostFilter redistribution')
        self.parser.add_argument('--ispf-adv', type=float, default=1.0,
                                 help='Weight for adversarial loss in generator')
        self.parser.add_argument('--ispf-oh', type=float, default=1.0,
                                 help='Weight for one-hot loss in generator')
        self.parser.add_argument('--ispf-bn', type=float, default=1.0,
                                 help='Weight for BatchNorm regularisation in generator '
                                      '(original repo default: 1.0)')

        self.args = None

    def parse_args(self):
        self.args = self.parser.parse_args()
        if self.args.extract_dataset is None:
            self.args.extract_dataset = self.args.dataset
        return self.args


    def general_config(self):
        config = {
            'output_base_dir': self.args.output_base_dir,
            'base_model': self.args.base_model,
            'seed': self.args.seed,
            'epoch': self.args.epoch,
            'version': self.args.version,
            'save_epoch': self.args.save_epoch,
            'save_best_only': self.args.save_best_only,
            'data_path': self.args.data_path,
            'dataset': self.args.dataset,
            'data_mode': self.args.data_mode,
            'extra_layer': self.args.extra_layer,
            'embedding_size': self.args.embedding_size,
            'train_batch_size': self.args.train_batch_size,
            'val_batch_size': self.args.val_batch_size,
            'unlearning_method': self.args.unlearning_method,
            'version_output': self.args.version + "_" + self.args.epoch if self.args.version is not None and self.args.epoch is not None else None,
            'max_epoch': self.args.max_epoch,
            'original_name': self.args.experiment_name,
            'num_classes': number_dataset_classes(self.args.dataset),
            'pretrained_model': self.args.pretrained_model,
            # "hyperparameters_path": self.hyperparameters_path(),
            # "checkpoints_path": self.checkpoints_path(),
            'experiment_name': self.experiment_name()
        }

        return config
    
    # def hyperparameters_path(self):
    #     return os.path.join(self.args.output_base_dir, self.experiment_name(), "hyperparameters")
    
    # def checkpoints_path(self):
    #     return os.path.join(self.args.output_base_dir, self.experiment_name(), "checkpoints")
    
    def experiment_name(self):
        experiment_name = self.args.experiment_name
        if self.args.data_mode == "backdoor":
            for i in range(len(self.args.backdoors)):
                attacker_class = self.args.backdoors[i]
                percentage_backdoor = self.args.percentages_backdoors[i]
                objective_backdoor = self.args.objectives_backdoors[i]
                experiment_name += f"_attack_class_{attacker_class}_{percentage_backdoor}_{objective_backdoor}"

        if self.args.data_mode == "CR":
            if self.args.classes_to_remove is not None:
                classes_to_remove_str = "_".join([str(cls) for cls in self.args.classes_to_remove])
                experiment_name += f"_classes_to_remove_{classes_to_remove_str}"

        return experiment_name
    

class ParserExtractor():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Script to extract embeddings and logits from our models')


        # Probably all general arguments that are used in all methods, like data path, output path, seed, fold split, experiment name, version, epoch, base model, batch size for training and validation, etc.
        self.parser.add_argument('-o', '--output-base-dir', type=str, required=True, help='Directory where all models are stored')

        self.parser.add_argument('--model-directory', type=str, required=True, help='Directory where the model to extract is stored. It should be in the format model_base_dir')
        self.parser.add_argument('-e', '--experiment-name', type=str, required=True, help='Name of the experiment used to reference it in the future')
        self.parser.add_argument('--unlearning-method', type=str, required=False, default=None, help='The implemented unlearning method used with their parameters')
        self.parser.add_argument('-m', '--base-model', type=str, required=True, help='Model to use')
        
        self.parser.add_argument('-ep','--checkpoint', type=str, required=True,
                                 help='Checkpoint to load: "best" for checkpoint_best.pt, '
                                      '"succed" for checkpoint_succed.pt, or an epoch number '
                                      '(e.g. "09") for checkpoint_epoch_09.pt')
        self.parser.add_argument('-v', '--version', type=str, required=True, help='If provided, the training will continue from that training version')

        
        self.parser.add_argument('-d', '--data-path', required=True, type=str, help='Root directory for dataset storage (e.g. ./data/)')
        self.parser.add_argument('--dataset', required=True, type=_validate_dataset, help='Dataset to use for training/unlearning (case-insensitive: CIFAR10, CIFAR100, MNIST, DTD, LETTERS)')
        self.parser.add_argument('--extract-dataset', required=False, default=None, type=_validate_dataset, help='Dataset to use for training/unlearning (case-insensitive: CIFAR10, CIFAR100, MNIST, DTD, LETTERS)')

        self.parser.add_argument('--data-mode', type=str, default="train", help='Chose which type of unlearn do')

        # When extra_layer is enabled it will add an extra fully connected layer between the predefined architecture and the new classifier head. 
        self.parser.add_argument('--extra-layer', action=argparse.BooleanOptionalAction)
        self.parser.add_argument('--embedding-size', type=int, default=256, help='Size of the embedding layer when extra_layer is enabled. Ignored otherwise.')

        # In case of data-mode == CR. These arguments specify which classes to remove from the training data.
        self.parser.add_argument('--classes-to-remove', nargs="+", type=int, default=None, help='List of classes to remove from training data')

        # Backdoor arguments
        self.parser.add_argument("--backdoors", nargs="+", type=int, help="List of datasets with backdoors", default=None)
        self.parser.add_argument('--percentages-backdoors',  nargs="+", type=float, help='List of percentages of backdoors for each dataset', default=None)
        self.parser.add_argument('--objectives-backdoors', nargs="+", type=int, help="Objective class for backdoor attacks (-1 means all classes)", default=None)

        self.args = None

    def general_config(self):

        if self.args.extract_dataset is None:
            self.args.extract_dataset = self.args.dataset

        config = {
            'output_base_dir': self.args.output_base_dir,
            'model_directory': self.args.model_directory,
            'base_model': self.args.base_model,
            'unlearning_method': self.args.unlearning_method,
            'checkpoint': self.args.checkpoint,
            'version': self.args.version,
            'data_path': self.args.data_path,
            'dataset': self.args.dataset,
            'extract_dataset': self.args.extract_dataset,
            'data_mode': self.args.data_mode,
            'extra_layer': self.args.extra_layer,
            'embedding_size': self.args.embedding_size,
            'original_name': self.args.experiment_name,
            # "hyperparameters_path": self.hyperparameters_path(),
            # "checkpoints_path": self.checkpoints_path(),
            'experiment_name': self.experiment_name()
        }

        return config

    def experiment_name(self):
        experiment_name = self.args.experiment_name
        if self.args.data_mode == "backdoor":
            for i in range(len(self.args.backdoors)):
                attacker_class = self.args.backdoors[i]
                percentage_backdoor = self.args.percentages_backdoors[i]
                objective_backdoor = self.args.objectives_backdoors[i]
                experiment_name += f"_attack_class_{attacker_class}_{percentage_backdoor}_{objective_backdoor}"

        if self.args.data_mode == "CR":
            if self.args.classes_to_remove is not None:
                classes_to_remove_str = "_".join([str(cls) for cls in self.args.classes_to_remove])
                experiment_name += f"_classes_to_remove_{classes_to_remove_str}"

        return experiment_name

    def parse_args(self):
        self.args = self.parser.parse_args()

        return self.args