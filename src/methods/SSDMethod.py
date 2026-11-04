
import os
import yaml
import numpy as np
from src.utils.dataset_loader import ClassificationDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import compute_fisher, compute_mas_coefficients

from .baseMethod import BaseMethod


class SSDMethod(BaseMethod):
    
    def __init__(self, model, optimizer, config):
        super(SSDMethod, self).__init__(model, optimizer, config)
        self.coeffs_train = None
        self.coeffs_forget = None
        self.setup_done = False

    def setup(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):
        
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = DataLoader(ClassificationDataset(train_X, train_y, transforms=transform_train), batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4)
        forget_loader = DataLoader(ClassificationDataset(forget_X, forget_y, transforms=transform_forget), batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4)
        if self.config['conservative'] == 'FISHER':
            print("Calculating Fisher coefficients for train set...")
            if os.path.exists(os.path.join("fishers", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt")):
                coeffs_train = compute_fisher(self.model, train_loader, os.path.join("fishers", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))  
            else:
                coeffs_train = compute_fisher(self.model, train_loader)
                os.makedirs(os.path.join("fishers"), exist_ok=True)
                torch.save(coeffs_train, os.path.join("fishers", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))
            
            print("Calculating Fisher coefficients for forget set...")
            coeffs_forget = compute_fisher(self.model, forget_loader)
        
        # Compute MAS coefficients
        elif self.config['conservative'] == 'MAS':
            print("Calculating MAS coefficients for train set...")
            if os.path.exists(os.path.join("mas", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt")):
                coeffs_train = compute_mas_coefficients(self.model, train_loader, os.path.join("mas", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))
            
            else:
                coeffs_train = compute_mas_coefficients(self.model, train_loader)
                os.makedirs(os.path.join("mas"), exist_ok=True)
                torch.save(coeffs_train, os.path.join("mas", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}.pt"))

            print("Calculating MAS coefficients for forget set...")
            coeffs_forget = compute_mas_coefficients(self.model, forget_loader)

        self.coeffs_train = coeffs_train
        self.coeffs_forget = coeffs_forget
        self.setup_done = True

        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    def unlearn(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):
        if not self.setup_done:
            raise ValueError("Setup must be called before unlearning.")

        self.model.train()

        freeze_mode = self.config.get('freeze', 'nofreeze')

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                self.coeffs_train.items(),
                self.coeffs_forget.items(),
            ):
                # When freeze is active, only modify classifier parameters
                if freeze_mode == 'freeze' and not n.startswith('classifier.'):
                    continue

                if freeze_mode == 'embs' and not self.model.is_emb_param(n):  # classifier + last backbone stage
                    continue

                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.config['selection_weighting'])
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.config['dampening_constant'])).div(fimp)).pow(
                    self.config['exponent']
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.config['lower_bound'])
                update[min_locs] = self.config['lower_bound']
                p[locations] = p[locations].mul(update)

        self.save_results()

        return self.model
    
    def save_results(self):
        
        # Saving model...
        # Final folder that will hold 'version_X'
        checkpoint = {
            "epoch": 0,
            "global_step": 0,
            "state_dict": self.model.state_dict()
        }

        os.makedirs(self.config['checkpoint_path'], exist_ok=True)
        torch.save(checkpoint, os.path.join(self.config['checkpoint_path'], "checkpoint_epoch_00.pt"))
        print(f"✅ Checkpoint guardado en {self.config['checkpoint_path']}")

        #Saving hyperparameters...
        hyperparameters = {
            "conservative": self.config['conservative'],
            "lower_bound": self.config['lower_bound'], 
            "exponent": self.config['exponent'], 
            "forget_threshold": self.config['forget_threshold'], 
            "dampening_constant": self.config['dampening_constant'], 
            "selection_weighting": self.config['selection_weighting'] 
        }


        with open(f'{self.config["hyperparameters_path"]}/hyperparameters.yml', 'w') as outfile:
            yaml.dump(hyperparameters, outfile, default_flow_style=False)

        print(f"✅ Hyperparameters saved to {self.config['hyperparameters_path']}/hyperparameters.yml")
        