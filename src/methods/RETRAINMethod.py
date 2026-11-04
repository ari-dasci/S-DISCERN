
import os
from sklearn.model_selection import train_test_split
import yaml
from src.models.optimizer import load_optimizer
from src.utils import ClassificationDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


from src.models.ClassificationModel import ClassificationModel
from .baseMethod import BaseMethod
from .utils import should_save_checkpoint


class RETRAINMethod(BaseMethod):
    def __init__(self, model, optimizer, config):
        super(RETRAINMethod, self).__init__(model, optimizer, config)

    def save_results(self, epoch, global_step, filename=None):
        # Saving model...
        # Final folder that will hold 'version_X'
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.model.state_dict()
        }

        

        os.makedirs(self.config['checkpoint_path'], exist_ok=True)
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:02}.pt"
        torch.save(checkpoint, os.path.join(self.config['checkpoint_path'], filename))
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}/{filename}")

    def setup(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):

        self.model = ClassificationModel(self.model._base_model, self.model._num_classes, self.model._hyperparameters, classes_weights=self.model.classes_weights, extra_layers=self.model._extra_layers)

        # Apply freeze if configured
        freeze_mode = self.config.get('freeze', 'nofreeze')
        if freeze_mode == 'freeze':
            self.model.freeze()
        elif freeze_mode == 'embs':
            self.model.emb_freeze()

        self.optimizer = load_optimizer(self.model, method="RETRAIN", config=self.config)
        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    def unlearn(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):
        
        retain_train_x, val_x, retain_train_y, val_y = train_test_split(retain_X, retain_y, test_size=0.2, stratify=retain_y, random_state=42)
        
            
        # Create DataLoaders
        val_dataset = ClassificationDataset(val_x,val_y, transforms=transform_retain)
        val_loader = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)

        # Forget validation loader for early stopping
        forget_train_x, forget_val_x, forget_train_y, forget_val_y = train_test_split(
            forget_X, forget_y, test_size=0.2, stratify=forget_y, random_state=42)
        val_loader_forget = DataLoader(
            ClassificationDataset(forget_val_x, forget_val_y, transforms=transform_forget),
            batch_size=self.config["val_batch_size"], shuffle=False, num_workers=4)


        train_dataset_unlearn = ClassificationDataset(retain_train_x, retain_train_y, transforms=transform_retain)
        train_loader = DataLoader(train_dataset_unlearn, batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4)

        best_acc_val = -1.0
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Save hyperparameters
        os.makedirs(self.config['hyperparameters_path'], exist_ok=True)
        hyperparameters = {
            "max_epoch": self.config["max_epoch"]
        }
        with open(f'{self.config["hyperparameters_path"]}/hyperparameters.yml', 'w') as outfile:
            yaml.dump(hyperparameters, outfile, default_flow_style=False)

        print(f"✅ Hyperparameters saved to {self.config['hyperparameters_path']}/hyperparameters.yml")

        n_step = 0

        for epoch in range(self.config['max_epoch']):
            self.model.train()
            for batch_idx, (images, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
                n_step += 1
                images, target = images.to(self.model.device), target.to(self.model.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.model.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            validation_loss = 0
            acc_val = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    images, target = images.to(self.model.device), target.to(self.model.device)
                    output = self.model(images)
                    loss = self.model.criterion(output, target)
                    validation_loss += loss.item()
                    acc_val += (output.argmax(dim=1) == target).sum().item()

                validation_loss /= len(val_loader.dataset)
                acc_val /= len(val_loader.dataset)


            print(f"Epoch {epoch} validation completed. \n Validation Loss: {validation_loss}, \n Acc val: {acc_val}\n \n \n", flush=True)

            if acc_val > best_acc_val:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": n_step,
                    "state_dict": self.model.state_dict()
                }
                best_acc_val = acc_val
                torch.save(checkpoint, os.path.join(self.config['checkpoint_path'], "checkpoint_best.pt"))
                print(f"✅ New best model (acc={acc_val:.4f}) saved to {self.config['checkpoint_path']}/checkpoint_best.pt")
            else:
                # Periodic save every save_epoch epochs
                checkpoint = {
                        "epoch": epoch,
                        "global_step": n_step,
                        "state_dict": self.model.state_dict()
                    }
                if epoch % self.config['save_epoch'] == (self.config['save_epoch'] - 1):
                    print(f"Saving checkpoint at epoch {epoch}", flush=True)
                    torch.save(checkpoint, os.path.join(self.config['checkpoint_path'], f"checkpoint_epoch_{epoch:02}.pt"))
                    print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}")

            # Forget validation for early stopping
            acc_forget = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (images, target) in enumerate(val_loader_forget):
                    images, target = images.to(self.model.device), target.to(self.model.device)
                    output = self.model(images)
                    acc_forget += (output.argmax(dim=1) == target).sum().item()
                acc_forget /= len(val_loader_forget.dataset)
            print(f"  Forget acc: {acc_forget:.4f}", flush=True)

        return self.model
