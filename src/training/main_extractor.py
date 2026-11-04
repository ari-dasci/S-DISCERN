import os

import src.config as cfg_module
from src.utils.load_data import load_data_general, number_dataset_classes
from src.models.ClassificationModel import ClassificationModel
from src.models.transforms import load_transforms

import torch
import pandas as pd
from tqdm import tqdm

from src.utils import ClassificationDataset
from torch.utils.data import DataLoader

def process_and_save(model, dataloader, save_path, mode):
    model.eval()
    logits_list = []
    embs_list = []
    label_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Extracting : {mode}', position=0, leave=True):
            images, labels = batch
            data = images.cuda()
            data = data.float()

            logits, embs = model(data, return_feature=True)
            logits = logits*15
            logits_list = logits_list + logits.cpu().numpy().tolist()
            embs_list = embs_list + embs.cpu().numpy().tolist()
            label_list =  label_list + labels.cpu().numpy().tolist()

    df_logits = pd.DataFrame(logits_list)
    df_logits.columns = [f'logits{i}' for i in range(df_logits.shape[1])]

    df_embs = pd.DataFrame(embs_list)
    df_embs.columns = [f'embeddings{i}' for i in range(df_embs.shape[1])]

    df_labels = pd.DataFrame(label_list, columns=['labels'])

    # Concatenate everything into a single DataFrame
    df_total = pd.concat([df_logits, df_embs, df_labels], axis=1)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_total.to_csv(f"{save_path}/{mode}.csv", index=False)

def main():
    parser = cfg_module.ParserExtractor()

    args = parser.parse_args()

    cfg = parser.general_config()

    transform_train, transform_retain, transform_forget = load_transforms(args)

    # Load data
    x_train, y_train, x_retain, y_retain, x_forget, y_forget, forbidden_labels = load_data_general(args, train=True)
    x_test, y_test, x_retain_test, y_retain_test, x_forget_test, y_forget_test, forbidden_labels = load_data_general(args, train=False)

    test_dataset = ClassificationDataset(x_retain_test, y_retain_test, transforms=transform_train)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=False, persistent_workers=True)

    train_dataset_unlearn = ClassificationDataset(x_retain, y_retain, transforms=transform_train)
    train_loader_unlearn = DataLoader(train_dataset_unlearn, batch_size=64, shuffle=False, num_workers=16, pin_memory=False, persistent_workers=True)

    if x_forget_test is not None and y_forget_test is not None:
        test_dataset_unlearn = ClassificationDataset(x_forget_test, y_forget_test, transforms=transform_train)
        test_loader_unlearn = DataLoader(test_dataset_unlearn, batch_size=64, shuffle=False, num_workers=16, pin_memory=False, persistent_workers=True)
    else:
        test_loader_unlearn = None

    # Classes weight should be used in the loss function in unbalanced datasets, but for now we will not use it in any method, so we will set it to None. It could be calculated as the inverse of the number of samples of each class in the training set.
    model = ClassificationModel(args.base_model, number_dataset_classes(args.dataset), cfg, classes_weights=None, extra_layers=args.extra_layer)

    # Resolve checkpoint filename from --checkpoint argument
    checkpoint_arg = cfg['checkpoint']
    if checkpoint_arg == 'best':
        checkpoint_filename = 'checkpoint_best.pt'
    elif checkpoint_arg == 'succed':
        checkpoint_filename = 'checkpoint_succed.pt'
    else:
        checkpoint_filename = f'checkpoint_epoch_{int(checkpoint_arg):02}.pt'

    if cfg['unlearning_method'] is None:
        model_path = os.path.join(cfg['model_directory'], cfg['original_name'], cfg['version'], checkpoint_filename)
    else:
        model_path = os.path.join(cfg['model_directory'], cfg['experiment_name'], cfg['unlearning_method'], cfg['version'], checkpoint_filename)
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=False)
    model.eval()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    if cfg['unlearning_method'] is None:
        save_path = f"{cfg['output_base_dir']}/{cfg['experiment_name']}/{cfg['version']}/{checkpoint_arg}"
    else:
        save_path = f"{cfg['output_base_dir']}/{cfg['experiment_name']}/{cfg['unlearning_method']}/{cfg['version']}/{checkpoint_arg}"
    process_and_save(model, test_loader, save_path, "test")
    # process_and_save(model, train_loader_unlearn, model_path.replace(f"checkpoint_epoch_{cfg['epoch']}.pt", "train_unlearn"), mode="train_unlearn")
    if test_loader_unlearn is not None:
        process_and_save(model, test_loader_unlearn, save_path, mode="test_unlearn") 
    print("Data extraction completed and saved to CSV files.")

if __name__ == "__main__": 


    main()