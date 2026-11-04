import os

import torch
import src.methods as methods

import numpy as np
import src.config as cfg_module
from src.utils.load_data import load_data_general, number_dataset_classes
from src.models.ClassificationModel import ClassificationModel
from src.models.transforms import load_transforms
from src.models.optimizer import load_optimizer

method_class_dictionary = {
    'EWC': lambda model, optimizer, cfg: methods.EWCMethod(model, optimizer, cfg),
    # 'SCAR': lambda model, optimizer, cfg: methods.BasePostprocessor(model, optimizer, cfg),
    'DELETE': lambda model, optimizer, cfg: methods.DELETEMethod(model, optimizer, cfg),
    'NOISY': lambda model, optimizer, cfg: methods.NOISYMethod(model, optimizer, cfg),
    'SSD': lambda model, optimizer, cfg: methods.SSDMethod(model, optimizer, cfg),
    'RETRAIN': lambda model, optimizer, cfg: methods.RETRAINMethod(model, optimizer, cfg),
    'DISTRIBUTIONAL': lambda model, optimizer, cfg: methods.DistributionalMethod(model, optimizer, cfg),
    'PARAMETRIC': lambda model, optimizer, cfg: methods.ParametricMethod(model, optimizer, cfg),
    'BADTEACHER': lambda model, optimizer, cfg: methods.BadTeacherMethod(model, optimizer, cfg),
    'BOUNDARYSHRINK': lambda model, optimizer, cfg: methods.BoundaryShrinkMethod(model, optimizer, cfg),
    'BOUNDARYEXPAND': lambda model, optimizer, cfg: methods.BoundaryExpandMethod(model, optimizer, cfg),
    'SFU': lambda model, optimizer, cfg: methods.SFUMethod(model, optimizer, cfg),
    'SCRUB': lambda model, optimizer, cfg: methods.SCRUBMethod(model, optimizer, cfg),
    'RANDOMLABEL': lambda model, optimizer, cfg: methods.RandomLabelMethod(model, optimizer, cfg),
    'FISHERFORGETTING': lambda model, optimizer, cfg: methods.FisherForgettingMethod(model, optimizer, cfg),
    'UNSIR': lambda model, optimizer, cfg: methods.UNSIRMethod(model, optimizer, cfg),
    'ISPF': lambda model, optimizer, cfg: methods.ISPFMethod(model, optimizer, cfg)
}

method_config_dictionary = {
    'EWC': lambda args, cfg: cfg_module.EWCconfig(args, cfg),
    # 'SCAR': lambda args, cfg: cfg_module.SCARconfig(args, cfg),
    'DELETE': lambda args, cfg: cfg_module.DELETEconfig(args, cfg),
    'NOISY': lambda args, cfg: cfg_module.NOISYconfig(args, cfg),
    'SSD': lambda args, cfg: cfg_module.SSDconfig(args, cfg),
    'RETRAIN': lambda args, cfg: cfg_module.RETRAINconfig(args, cfg),
    'DISTRIBUTIONAL': lambda args, cfg: cfg_module.DISTRIBUTIONALconfig(args, cfg),
    'PARAMETRIC': lambda args, cfg: cfg_module.PARAMETRICconfig(args, cfg),
    'BADTEACHER': lambda args, cfg: cfg_module.BadTeacherconfig(args, cfg),
    'BOUNDARYSHRINK': lambda args, cfg: cfg_module.BoundaryShrinkconfig(args, cfg),
    'BOUNDARYEXPAND': lambda args, cfg: cfg_module.BoundaryExpandconfig(args, cfg),
    'SFU': lambda args, cfg: cfg_module.SFUconfig(args, cfg),
    'SCRUB': lambda args, cfg: cfg_module.SCRUBconfig(args, cfg),
    'RANDOMLABEL': lambda args, cfg: cfg_module.RandomLabelconfig(args, cfg),
    'FISHERFORGETTING': lambda args, cfg: cfg_module.FisherForgettingconfig(args, cfg),
    'UNSIR': lambda args, cfg: cfg_module.UNSIRconfig(args, cfg),
    'ISPF': lambda args, cfg: cfg_module.ISPFconfig(args, cfg)
}

def main():

    parser = cfg_module.Parser()

    args = parser.parse_args()

    cfg = parser.general_config()
    
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    # Load data
    x_train, y_train, x_retain, y_retain, x_forget, y_forget, forbidden_labels = load_data_general(args, train=True)

    transform_train, transform_retain, transform_forget = load_transforms(args)

    # Classes weight should be used in the loss function in unbalanced datasets, but for now we will not use it in any method, so we will set it to None. It could be calculated as the inverse of the number of samples of each class in the training set.
    model = ClassificationModel(args.base_model, number_dataset_classes(args.dataset), cfg, classes_weights=None, extra_layers=args.extra_layer)

    # Load pretrained checkpoint from training phase
    epoch_value = str(cfg['epoch'])
    if epoch_value.lower() == 'best':
        ckpt_filename = 'checkpoint_best.pt'
    elif epoch_value.lower() == 'succed':
        ckpt_filename = 'checkpoint_succed.pt'
    else:
        ckpt_filename = f'checkpoint_epoch_{int(epoch_value):02}.pt'
    checkpoint_path = os.path.join(cfg['output_base_dir'], cfg['original_name'], cfg['version'], ckpt_filename)
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=False)
    print(f"✅ Pre-trained model loaded from {checkpoint_path}")

    optimizer = load_optimizer(model, method=cfg['unlearning_method'], config=cfg)

    cfg = method_config_dictionary[cfg['unlearning_method']](args, cfg)

    cfg['forbidden_labels'] = forbidden_labels

    method_unlearning = method_class_dictionary[cfg['unlearning_method']](model, optimizer, cfg)
    train_X, train_y, retain_X, retain_y, forget_X, forget_y = method_unlearning.setup(train_X= x_train, train_y= y_train, retain_X= x_retain, retain_y= y_retain, forget_X= x_forget, forget_y= y_forget, transform_train= transform_train, transform_retain= transform_retain, transform_forget = transform_forget)

    model_unlearned = method_unlearning.unlearn(train_X= train_X, train_y= train_y, retain_X= retain_X, retain_y= retain_y, forget_X= forget_X, forget_y= forget_y, transform_train= transform_train, transform_retain= transform_retain, transform_forget = transform_forget)

if __name__ == "__main__":
    main()