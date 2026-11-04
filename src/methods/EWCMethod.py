
import os
import yaml

from src.utils.dataset_loader import ClassificationDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from .utils import compute_fisher, compute_mas_coefficients, should_save_checkpoint

from .baseMethod import BaseMethod

class EWCMethod(BaseMethod):
    
    def __init__(self, model, optimizer, config):
        super(EWCMethod, self).__init__(model, optimizer, config)
        self.coeffs_importance = None


    def setup(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        train_loader = DataLoader(ClassificationDataset(train_X, train_y, transforms=transform_train), batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4)
        if self.config['coefficients'] == 'FISHER':
            if os.path.exists(os.path.join("fishers", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt")):
                self.coeffs_importance = compute_fisher(self.model, train_loader, os.path.join("fishers", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))  
            else:
                self.coeffs_importance = compute_fisher(self.model, train_loader)
                os.makedirs(os.path.join("fishers"), exist_ok=True)
                torch.save(self.coeffs_importance, os.path.join("fishers", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))
            
        
        # Compute MAS coefficients
        elif self.config['coefficients'] == 'MAS':
            if os.path.exists(os.path.join("mas", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt")):
                self.coeffs_importance = compute_mas_coefficients(self.model, train_loader, os.path.join("mas", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))
            
            else:
                self.coeffs_importance = compute_mas_coefficients(self.model, train_loader)
                os.makedirs(os.path.join("mas"), exist_ok=True)
                torch.save(self.coeffs_importance, os.path.join("mas", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))

        if self.config["embunfreeze"] == "embs":
            self.model.emb_freeze()
        elif self.config["embunfreeze"] == "freeze":
            self.model.freeze()

        return train_X, train_y, retain_X, retain_y, forget_X, forget_y
    
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

    def unlearn(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):
        if self.coeffs_importance is None:
            raise ValueError("Setup must be called before unlearning.")
        
        # Training with the EWC penalty based on self.coeffs_importance
        # (the penalty is added to the loss over the model parameters during training).

        forget_train_x, forget_val_x, forget_train_y, forget_val_y = train_test_split(forget_X, forget_y, test_size=0.2, stratify=forget_y, random_state=42) # With a fixed dataset the split would not be needed, but as a general method we keep a validation set specific to the forget set.

            
        # Create DataLoaders
        val_dataset = ClassificationDataset(forget_val_x,forget_val_y, transforms=transform_forget)
        val_loader_forget = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)

        unlearning_x = forget_train_x
        unlearning_y = forget_train_y

        train_dataset_unlearn = ClassificationDataset(unlearning_x, unlearning_y, transforms=transform_forget)
        train_loader_unlearn = DataLoader(train_dataset_unlearn, batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4)

        # Save hyperparameters
        os.makedirs(self.config['hyperparameters_path'], exist_ok=True)
        hyperparameters = {
            "lambda": self.config['lambda'],
            "coefficients": self.config['coefficients'],
            "loss_forget": self.config['loss_forget'],
            "embunfreeze": self.config['embunfreeze']
        }
        with open(f'{self.config["hyperparameters_path"]}/hyperparameters.yml', 'w') as outfile:
            yaml.dump(hyperparameters, outfile, default_flow_style=False)

        print(f"✅ Hyperparameters saved to {self.config['hyperparameters_path']}/hyperparameters.yml")

        # Optimizer and scheduler set up
        n_step = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        original_parameters = {name: param.clone().detach().to(device) for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        print("Starting unlearning process", flush=True)
        for epoch in range(self.config['max_epoch']):
            self.model.train()
            for batch_idx, (images, target) in tqdm(enumerate(train_loader_unlearn), total=len(train_loader_unlearn)):
                n_step += 1
                images, labels = images.to(self.model.device), target.to(self.model.device)
                self.optimizer.zero_grad()
                loss = 0

                if self.config['loss_forget'] != "mahalanobis":
                    logits = self.model(images)
                else:
                    raise ValueError("Mahalanobis loss is not supported in EWC. Use NOISY method instead.")

                if self.config['loss_forget'] == "softmax":  # Basic OE (Uniform KL-Div Loss)
                    uniform_target = torch.ones_like(logits) / logits.size(1)
                    forget_loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_target, reduction="batchmean")
                
                elif self.config['loss_forget'] == "energy":  # EnergyOE (Softplus on Energy Score)
                    energy = 1.0* torch.logsumexp(logits / 1.0, dim=1) # 1.0 is the temperature parameter, can be tuned
                    forget_loss = torch.mean(F.softplus(energy))

                elif self.config['loss_forget'] == "confidence_penalty":  # Confidence Penalty
                    probs = torch.softmax(logits, dim=1)
                    forget_loss = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                elif self.config['loss_forget'] == "random_class": # Random Class labeling on each batch
                    random_labels = torch.randint(0, self.config['num_classes'], (logits.size(0),), device=logits.device)
                    forget_loss = self.model.criterion(logits, random_labels)

                elif self.config['loss_forget'] == "negative_entropy": # Negative Cross Entropy
                    forget_loss = - self.model.criterion(logits, labels)

                elif self.config['loss_forget'] == "notop":  # Not-on-top hinge loss
                    cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                    cols = cols.to(logits.device)
                    mask = cols != labels.unsqueeze(1)
                    mask = mask.to(logits.device)
                    other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                    best_logits = torch.topk(other_logits, k=1, dim=1).values
                    mylogit = logits[:, labels]
                    forget_loss = F.relu(mylogit - best_logits + 10).mean()
                
                else:
                    raise ValueError("Invalid loss_forget for EWC! Choose 'softmax', 'energy', 'confidence_penalty', 'random_class', 'negative_entropy' or 'notop'.")

                # EWC penalty
                ewc_loss = 0
                for n,p in self.model.named_parameters():
                    if n in original_parameters:
                        ewc_loss += (self.coeffs_importance[n] * (p - original_parameters[n]).pow(2)).sum()


                loss = forget_loss + self.config['lambda'] * ewc_loss
                loss.backward()
                self.optimizer.step()

            # Validation step
            self.model.eval()
            acc_forget = 0
            ewc_loss_val = 0
            with torch.no_grad():
                for batch_idx, (images, target) in tqdm(enumerate(val_loader_forget), total=len(val_loader_forget)):
                
                    images, labels = images.to(self.model.device), target.to(self.model.device)
                    logits = self.model(images)
                    acc_forget += (logits.argmax(dim=1) == labels).sum().item()
                
                for n,p in self.model.named_parameters():
                    if n in original_parameters:
                        ewc_loss_val += (self.coeffs_importance[n] * (p - original_parameters[n]).pow(2)).sum()

                ewc_loss_val = self.config['lambda'] * ewc_loss_val.item()
            
                acc_forget /= len(val_loader_forget.dataset)

                print(f"Epoch {epoch+1}/{self.config['max_epoch']} - Acc on forget set: {acc_forget:.4f} - EWC Loss: {ewc_loss_val:.4f}", flush=True)

            # Check if checkpoint should be saved / early stopping
            checkpoints_to_save, stop = should_save_checkpoint(epoch, acc_forget, self.config['num_classes'], self.config, early_stop_state)
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                self.save_results(epoch, n_step, filename=ckpt_name)
            if stop:
                print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                break
        
        return self.model