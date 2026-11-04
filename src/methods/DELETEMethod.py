from copy import deepcopy
import os
import yaml

from src.utils.dataset_loader import ClassificationDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from .baseMethod import BaseMethod
from .utils import should_save_checkpoint

class DELETEMethod(BaseMethod):

    def __init__(self, model, optimizer, config):
        super(DELETEMethod, self).__init__(model, optimizer, config)
        self.original_model = None

    def setup(self, train_X=None, train_y=None, retain_X=None, retain_y=None, forget_X=None, forget_y=None, transform_train=None, transform_retain=None, transform_forget=None):

        self.original_model = deepcopy(self.model)

        if self.config['freeze'] == "freeze":
            self.model.freeze()
            
        elif self.config['freeze'] == "embs":
            self.model.emb_freeze()



        return train_X, train_y, retain_X, retain_y, forget_X, forget_y
    
    def save_results(self, epoch, global_step, filename=None): # Deberia de ser un metodo de baseMethod
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

    def unlearn(self, train_X=None, train_y=None, retain_X=None, retain_y=None, forget_X=None, forget_y=None, transform_train=None, transform_retain=None, transform_forget=None):

        forget_train_x, forget_val_x, forget_train_y, forget_val_y = train_test_split(forget_X, forget_y, test_size=0.2, stratify=forget_y, random_state=42) # With a fixed dataset the split would not be needed, but as a general method we keep a validation set specific to the forget set.

            
        # Create DataLoaders
        val_dataset = ClassificationDataset(forget_val_x,forget_val_y, transforms=transform_forget)
        val_loader_forget = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)

        unlearning_x = forget_train_x
        unlearning_y = forget_train_y

        train_dataset_unlearn = ClassificationDataset(unlearning_x, unlearning_y, transforms=transform_forget)
        train_loader_unlearn = DataLoader(train_dataset_unlearn, batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4)

        #Saving hyperparameters...
        os.makedirs(self.config['hyperparameters_path'], exist_ok=True)
        hyperparameters = {
            "freeze": self.config['freeze']
        }
        with open(f'{self.config["hyperparameters_path"]}/hyperparameters.yml', 'w') as outfile:
            yaml.dump(hyperparameters, outfile, default_flow_style=False)

        print(f"✅ Hyperparameters saved to {self.config['hyperparameters_path']}/hyperparameters.yml")

        # Optimizer and scheduler set up
        n_step = 0
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.original_model.cuda() if torch.cuda.is_available() else self.original_model.cpu()
        self.original_model.eval()
        
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
                images, _ = images.to(self.model.device), target.to(self.model.device)
                self.optimizer.zero_grad()
                # Forward pass through student model (trainable)
                logits_student = self.model(images)

                # Forward pass through frozen teacher model (no gradients)
                with torch.no_grad():
                    logits_teacher = self.original_model(images)
                
                # Create decoupled distillation mask: suppress forget classes
                mask = torch.zeros(logits_teacher.size(), device=logits_student.device)
                
                # Mask forget classes with -inf to push their probability to ~0
                if len(self.config['classes_to_remove'] ) > 0:
                    forget_classes_device = torch.tensor(self.config['classes_to_remove'], device=logits_teacher.device)
                    mask[:, forget_classes_device] = -float('inf')

                # Apply mask to teacher logits
                logits_teacher_masked = logits_teacher + mask

                # Compute soft targets using teacher with masked logits
                soft_targets = F.softmax(logits_teacher_masked, dim=1)
                
                # Compute log softmax for student logits
                softmax_student = F.softmax(logits_student, dim=1)

                # KL divergence loss: student learns to match teacher's distribution
                # Format: KL(softmax(student) || softmax(teacher))
                loss = F.kl_div(softmax_student, soft_targets, reduction='batchmean')

                loss.backward()
                self.optimizer.step()

            # Validation step
            self.model.eval()
            acc_forget = 0
            with torch.no_grad():
                for batch_idx, (images, target) in tqdm(enumerate(val_loader_forget), total=len(val_loader_forget)):
                
                    images, labels = images.to(self.model.device), target.to(self.model.device)
                    logits = self.model(images)
                    acc_forget += (logits.argmax(dim=1) == labels).sum().item()
                

            
                acc_forget /= len(val_loader_forget.dataset)

                print(f"Epoch {epoch+1}/{self.config['max_epoch']} - Acc on forget set: {acc_forget:.4f}", flush=True)

            # Check if checkpoint should be saved / early stopping
            checkpoints_to_save, stop = should_save_checkpoint(epoch, acc_forget, self.config['num_classes'], self.config, early_stop_state)
            for ckpt_name, reason in checkpoints_to_save:
                print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                self.save_results(epoch, n_step, filename=ckpt_name)
            if stop:
                print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                break

        return self.model