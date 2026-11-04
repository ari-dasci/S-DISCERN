import os
from sklearn.model_selection import train_test_split
import yaml
import src.config as cfg_module
from src.utils import ClassificationDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


from src.models.ClassificationModel import ClassificationModel


from src.utils.load_data import load_data_general, number_dataset_classes
from src.models.transforms import load_transforms
from src.models.optimizer import load_optimizer


def main():

    

    parser = cfg_module.Parser()

    args = parser.parse_args()

    cfg = parser.general_config()
    torch.manual_seed(cfg['seed'])
    cfg['checkpoint_path'] = f"{cfg['output_base_dir']}/{cfg['experiment_name']}/original"
    cfg['hyperparameters_path'] = f"{cfg['output_base_dir']}/{cfg['experiment_name']}/original"

    # Load data
    x_train, y_train, _, _, _, _, _ = load_data_general(args, train=True)

    transform_train, _, _ = load_transforms(args)

    # Classes weight should be used in the loss function in unbalanced datasets, but for now we will not use it in any method, so we will set it to None. It could be calculated as the inverse of the number of samples of each class in the training set.
    model = ClassificationModel(args.base_model, number_dataset_classes(args.dataset), cfg, classes_weights=None, extra_layers=args.extra_layer) # We will load the model later, we need to know the epoch to load the correct one
    optimizer = load_optimizer(model, config=cfg)

    train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

            
    # Create DataLoaders
    val_dataset = ClassificationDataset(val_x,val_y, transforms=transform_train)
    val_loader = DataLoader(val_dataset, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)


    train_dataset_unlearn = ClassificationDataset(train_x, train_y, transforms=transform_train)
    train_loader = DataLoader(train_dataset_unlearn, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=4)

    
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Save hyperparameters
    os.makedirs(cfg['hyperparameters_path'], exist_ok=True)
    hyperparameters = {
        "max_epoch": cfg["max_epoch"]
    }
    with open(f'{cfg["hyperparameters_path"]}/hyperparameters.yml', 'w') as outfile:
        yaml.dump(hyperparameters, outfile, default_flow_style=False)

    print(f"✅ Hyperparameters saved to {cfg['hyperparameters_path']}/hyperparameters.yml")

    n_step = 0
    best_acc_val = -1.0
    save_best_only = cfg.get('save_best_only', True)

    for epoch in range(cfg['max_epoch']):
        model.train()
        for batch_idx, (images, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            n_step += 1
            images, target = images.to(model.device), target.to(model.device)
            optimizer.zero_grad()
            output = model(images)
            loss = model.criterion(output, target)
            loss.backward()
            optimizer.step()

        validation_loss = 0
        acc_val = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                images, target = images.to(model.device), target.to(model.device)
                output = model(images)
                loss = model.criterion(output, target)
                validation_loss += loss.item()
                acc_val += (output.argmax(dim=1) == target).sum().item()

            validation_loss /= len(val_loader.dataset)
            acc_val /= len(val_loader.dataset)

        print(f"Epoch {epoch} validation completed. \n Validation Loss: {validation_loss}, \n Acc val: {acc_val}\n \n \n", flush=True)

        os.makedirs(cfg['checkpoint_path'], exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "global_step": n_step,
            "state_dict": model.state_dict()
        }

        if save_best_only:
            # Save only when the validation accuracy improves
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                torch.save(checkpoint, os.path.join(cfg['checkpoint_path'], "checkpoint_best.pt"))
                print(f"✅ New best model (acc={acc_val:.4f}) saved to {cfg['checkpoint_path']}/checkpoint_best.pt")
        else:
            # Periodic save every save_epoch epochs
            if epoch % cfg['save_epoch'] == (cfg['save_epoch'] - 1):
                print(f"Saving checkpoint at epoch {epoch}", flush=True)
                torch.save(checkpoint, os.path.join(cfg['checkpoint_path'], f"checkpoint_epoch_{epoch:02}.pt"))
                print(f"✅ Checkpoint guardado en {cfg['checkpoint_path']}")

if __name__ == "__main__":
    main()