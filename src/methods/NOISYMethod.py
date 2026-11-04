import os

import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import numpy as np
from src.utils import ClassificationDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.distributions import MultivariateNormal

from .baseMethod import BaseMethod
from .utils import calculate_centroids, should_save_checkpoint

class ConditionalTransform:
    """
    Wrapper that applies the transform only if the input is NOT a 1D tensor (embedding).
    If it is 1D (embedding), it is returned unchanged.
    If it is a PIL image (2D+), the transform is applied.
    """
    def __init__(self, transform=None):
        self.transform = transform
    
    def __call__(self, x):
        # 1D tensor: return it untransformed
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            # PIL Image: apply the transform
            if self.transform is not None:
                return self.transform(x)
            return x
        
        # No transform: convert to tensor
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        
        return x

def collate_fn_mixed(batch):
    """
    Custom collate function that handles tensors of different shapes (images vs embeddings).
    Returns lists of tensors instead of stacked tensors.
    """
    images, labels = zip(*batch)
    labels = torch.stack([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels])
    
    # Do not stack: return a list if the shapes are incompatible
    try:
        images = torch.stack(images)
    except RuntimeError:
        # If they cannot be stacked (different shapes), return a list
        images = list(images)
    
    return images, labels


class NOISYMethod(BaseMethod):
    
    def __init__(self, model, optimizer, config):
        super(NOISYMethod, self).__init__(model, optimizer, config)
        self.using_robust = None
        self.centroids = None
        self.covariance = None
        self.percentages = None
        self.intra_percentage = None
        self.num_samples_original = None
        self.pre_centroids = None
        self.pre_covariance = None
        self.gaussian_noise = None
        self.lambda_margin = 0.0
        self.config["retain_labels"] = [i for i in range(self.config["num_classes"]) if i not in self.config["forbidden_labels"]]

    @property
    def device(self):
        """Return the device of the model's first parameter."""
        return next(self.model.parameters()).device

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

    def calculate_minimun_pre_data(self):
        if self.centroids is None:
            raise RuntimeError("calculate_centroids must be called before calculate_minimun_preembeddings()")
        
        head = self.model.get_embs_layer()  
        W = head.weight.detach()
        b = head.bias.detach()
        
        aux_data_centroids = [None for _ in range(self.config["num_classes"])]
        aux_data_covariance = [None for _ in range(self.config["num_classes"])]

        W_pinv = torch.linalg.pinv(W)

        if self.config['update_covariance']:
            # In case  of Bayesian reconstruction of the covariance matrix, due to lack of information we are guessing gaussian noisy on the invisible proyections 
            I = torch.eye(W.shape[1], device=W.device)
            P_null = I - torch.matmul(W_pinv, W)
            in_visible_pre_covariance = torch.matmul(P_null, torch.matmul(I * 1e-8, P_null.t()))

        if self.config['clustering']:
            for i in range(self.config["num_classes"]):
                aux_data_centroids[i] = []
                aux_data_covariance[i] = []
                for j in range(len(self.centroids[i])):
                    if self.config['update_centroids']:
                        X = self.centroids[i][j].unsqueeze(1)  # To column vector
                        aux_data_centroids[i].append( W_pinv @ (X - b).squeeze(1))  # Back to row vector

                    if self.config['update_covariance']:
                        current_covariance = self.covariance[i][j]
                        visible_pre_covariance = torch.matmul(W_pinv, torch.matmul(current_covariance, W_pinv.t()))

                        aux_data_covariance[i].append(visible_pre_covariance + in_visible_pre_covariance)
                    
        else:
            for i in range(self.config["num_classes"]):
                if self.config['update_centroids']:
                    X = self.centroids[i].unsqueeze(1)  # To column vector
                    aux_data_centroids[i] = W_pinv @ (X - b).squeeze(1)  # Back to row vector

                if self.config['update_covariance']:
                    current_covariance = self.covariance[i]
                    visible_pre_covariance = torch.matmul(W_pinv, torch.matmul(current_covariance, W_pinv.t()))


                    aux_data_covariance[i] = visible_pre_covariance + in_visible_pre_covariance


        if self.config['update_centroids']:
            self.pre_centroids = aux_data_centroids

        if self.config['update_covariance']:
            self.pre_covariance = aux_data_covariance

    def calculate_minimun_preEmbeddings(self, syntetic_embeddings):
        head = self.model.get_embs_layer()  # The classifier is assumed to be the second head
        W = head.weight.detach()    # Classifier weight
        b = head.bias.detach()      # Classifier bias

        preembeddings = torch.linalg.pinv(W) @ (syntetic_embeddings.t() - b.unsqueeze(1))
        return preembeddings.t()
    
    def actualizeGaussianNoise(self):
        if self.config['clustering']:
                self.gaussian_noise = {i: [] for i in range(self.config['num_classes'])}
                for i in range(self.config['num_classes']):
                    for j in range(len(self.intra_percentage[i])):
                        self.gaussian_noise[i].append(
                            MultivariateNormal(
                                loc=self.centroids[i][j],
                                covariance_matrix=self.covariance[i][j]
                            )
                        )
        else:
            self.gaussian_noise = {}
            for i in range(self.config['num_classes']):
                self.gaussian_noise[i] = MultivariateNormal(
                    loc=self.centroids[i],
                    covariance_matrix=self.covariance[i]
                )

    def generate_syntetic_embeddings(self, n_samples):
        """
        Generates 'n_samples' synthetic embeddings, sampling each class according to percentage_class,
        and returns the generated embeddings together with their labels.
        """
        if self.percentages is None or self.gaussian_noise is None:
            raise RuntimeError("calculate_centroids must be called before generate_syntetic_embeddings()")

        seed = self.config['seed']
        # Tensor on the device so that multinomial returns indices on the same device
        probs = self.percentages.to(self.device).float()
        
        syntetic_labels = []
        while len(syntetic_labels) < n_samples:
            torch.manual_seed(seed)
            candidate_labels = torch.multinomial(probs, num_samples=n_samples - len(syntetic_labels), replacement=True)
            for lbl in candidate_labels.tolist():
                if lbl not in self.config['forbidden_labels']:
                    syntetic_labels.append(lbl)
            seed += 1  # Change the seed for the next iteration if needed

        syntetic_labels = torch.tensor(syntetic_labels, dtype=torch.long, device=self.device)
        syntetic_embeddings = []

        # iterate over the indices (.tolist() to get ints)
        for idx in syntetic_labels.tolist():
            if self.config['clustering']:
                # pick a sub-cluster according to intra_percentage
                intra_probs = self.intra_percentage[idx].to(self.device).float()
                sub_idx = torch.multinomial(intra_probs, num_samples=1).item()
                noise = self.gaussian_noise[idx][sub_idx].sample().to(self.device)
            else:
                noise = self.gaussian_noise[idx].sample().to(self.device)
            
            syntetic_embeddings.append(noise)

        syntetic_embeddings = torch.stack(syntetic_embeddings, dim=0).float()
        return syntetic_embeddings, syntetic_labels + self.config['num_classes']  # Labels outside the original range
 

    def setup(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):

        if retain_X is None:
            retain_X = []
        if retain_y is None:
            retain_y = []

        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = DataLoader(ClassificationDataset(train_X, train_y, transforms=ConditionalTransform(transform_train)), batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4,)

        # Configure  model and unlearning version
        if self.config["embunfreeze"] == "embs":
            self.model.emb_freeze()
        elif self.config["embunfreeze"] == "freeze":
            self.model.freeze()
                      
        print("Calculating class centroids", flush=True)
        if  self.config['robust']:
            using_robust = "robust"
        else:
            using_robust = "standard"

        if self.config['clustering']:
            using_robust = using_robust + "_clustering"

        # Include covariance type in cache path to avoid collisions
        cov_type = self.config.get('covariance_type', 'full')
        if cov_type != 'full':
            using_robust = using_robust + f"_cov_{cov_type}"
            if cov_type == 'pca' and self.config.get('n_pca_components') is not None:
                using_robust = using_robust + f"_k{self.config['n_pca_components']}"

        if os.path.exists(os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_num_samples.pt")):
            if self.config['clustering']:
                centroids, covariance, percentages, intra_percentage, num_samples_original = calculate_centroids(
                    self.model, train_loader, self.config['num_classes'],
                    self.config['robust'], self.config['clustering'],
                    os.path.join(using_robust, f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}"),
                    covariance_type=self.config.get('covariance_type', 'full'),
                    n_pca_components=self.config.get('n_pca_components'))  # Compute class centroids
                self.centroids = centroids
                self.covariance = covariance
                self.percentages = percentages
                self.intra_percentage = intra_percentage
                self.num_samples_original = num_samples_original

            else: 
                centroids, covariance, percentages, num_samples_original = calculate_centroids(
                    self.model, train_loader, self.config['num_classes'],
                    self.config['robust'], self.config['clustering'],
                    os.path.join(using_robust, f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}"),
                    covariance_type=self.config.get('covariance_type', 'full'),
                    n_pca_components=self.config.get('n_pca_components'))  # Compute class centroids
                self.centroids = centroids
                self.covariance = covariance
                self.percentages = percentages
                self.num_samples_original = num_samples_original
        else:
            if self.config['clustering']:
                centroids, covariance, percentages, intra_percentage, num_samples_original = calculate_centroids(
                    self.model, train_loader, self.config['num_classes'],
                    self.config['robust'], self.config['clustering'],
                    covariance_type=self.config.get('covariance_type', 'full'),
                    n_pca_components=self.config.get('n_pca_components'))
                os.makedirs(os.path.join(f"{using_robust}"), exist_ok=True)
                torch.save(centroids, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_centroids.pt"))
                torch.save(covariance, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_covariance.pt"))
                torch.save(percentages, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_percentage.pt"))
                torch.save(intra_percentage, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_percentage_intra.pt"))
                torch.save(num_samples_original, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_num_samples.pt"))
                self.centroids = centroids
                self.covariance = covariance
                self.percentages = percentages
                self.intra_percentage = intra_percentage
                self.num_samples_original = num_samples_original

            else:
                centroids, covariance, percentages, num_samples_original = calculate_centroids(
                    self.model, train_loader, self.config['num_classes'],
                    self.config['robust'], self.config['clustering'],
                    covariance_type=self.config.get('covariance_type', 'full'),
                    n_pca_components=self.config.get('n_pca_components'))
                os.makedirs(os.path.join(f"{using_robust}"), exist_ok=True)
                torch.save(centroids, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_centroids.pt"))
                torch.save(covariance, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_covariance.pt"))
                torch.save(percentages, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_percentage.pt"))
                torch.save(num_samples_original, os.path.join(f"{using_robust}", f"{self.config['original_name']}_{self.config['version']}_epoch_{self.config['epoch']}_num_samples.pt"))
                self.centroids = centroids
                self.covariance = covariance
                self.percentages = percentages
                self.num_samples_original = num_samples_original

        # Discarded option to update covariance matrix during unlearning, as it does not improve results and is computationally expensive. Kept in case we want to explore it in the future.
        # if self.config['calculate-covariance'] and not self.config['robust']:
        #     print("Substracting forget set from covariance matrix", flush=True)
        #     self.model.update_covariance_matrix(train_loader)
        #     using_robust = using_robust + "_updated"

        # elif self.config['calculate-covariance'] and self.config['robust']:
        #     raise ValueError("Covariance matrix update is only allowed when not using robust Mahalanobis distance.")
        
        if (self.config['update_centroids'] or self.config['update_covariance']) and (self.config["embunfreeze"] != "freeze"):
            print("Lifting data", flush=True)
            self.calculate_minimun_pre_data()
            using_robust = using_robust + "_centroids_moving"

        elif (self.config['update_centroids'] or self.config['update_covariance']) and (self.config["embunfreeze"] == "freeze"):
            raise ValueError("Centroids or Covariance update is only allowed when not freezing the embeddings layer.")
        
        if (self.config["embunfreeze"] == "freeze") and self.config['loss_forget'] == "mahalanobis":
            raise ValueError("Using Mahalanobis distance as conservative loss is not allowed when freezing the embeddings layer.")


        self.actualizeGaussianNoise()

        if self.config['syntetic_samples'] != -1:
            if self.config['fixed_syntetic']:
                print("Generating fixed syntetic samples", flush=True)
                syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(self.config['syntetic_samples'])
                
                using_robust = using_robust + "_syntetic"
            
                if self.config['update_syntetic']:
                    syntetic_preembeddings = self.calculate_minimun_preEmbeddings(syntetic_embeddings)

                    retain_X = [i for i in syntetic_preembeddings.cpu().numpy()]
                    using_robust = using_robust + "_moving"
                else:
                    retain_X = [i for i in syntetic_embeddings.cpu().numpy()]
                    using_robust = using_robust + "_fixed"

                retain_y = [i for i in syntetic_labels.cpu().numpy()]

                if (self.config['update_centroids'] or self.config['update_covariance']):
                    raise ValueError("Fixed syntetic samples cannot be used when centroids or coavriance are updated during training if the loss function do not use centroids or covariance")

            else:
                retain_y = [-1 for i in range(self.config['syntetic_samples'])]
                retain_X = [forget_X[0] for i in range(self.config['syntetic_samples'])] # Dummy data, will be generated inside the model
            
            
            #train_dataset_unlearn = ClassificationDataset(x_train_unlearn,y_train_unlearn, transforms=transform)
        
        if self.config["embunfreeze"] == "embs":
            using_robust += "_embs"
        elif self.config["embunfreeze"] == "freeze":
            using_robust += "_freeze"

        else: 
            using_robust += "_nofreeze"

        if self.config["self_forget"]:
            print("Generating fixed forget synthetic samples", flush=True)
            syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(
                    self.config["forget_syntetic_samples"],
                    self.config["retain_labels"]
                )
            
            if self.config['update_syntetic']:
                    syntetic_embeddings = self.calculate_minimun_preEmbeddings(syntetic_embeddings)
            forget_X = [e for e in syntetic_embeddings.cpu().numpy()]
            forget_y = [l for l in syntetic_labels.cpu().numpy()]

            using_robust += f"_self_forget_{self.config['forget_syntetic_samples']}"
        
        using_robust = using_robust + f"_forget_{self.config['loss_forget']}"

        self.using_robust = using_robust
        # Note: train_batch_size already in config from general_config()
        self.config['checkpoint_path'] = f"{self.config['output_base_dir']}/{self.config['experiment_name']}/NOISY_{self.config['syntetic_samples']}_{using_robust}/{self.config['version_output']}"
        self.config['hyperparameters_path'] = f"{self.config['output_base_dir']}/{self.config['experiment_name']}/NOISY_{self.config['syntetic_samples']}_{using_robust}/{self.config['version_output']}"

        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    def actualizeCentroidsAndCovariance(self):
        if self.pre_centroids is not None:
            with torch.no_grad():
                head = self.model.get_embs_layer()  # The classifier is assumed to be the second head
                W = head.weight.detach()            # Classifier weight
                b = head.bias.detach()              # Classifier bias
                if self.config['clustering']:
                    
                    
                    for i in range(self.config['num_classes']):
                        for j in range(len(self.pre_centroids[i])):
                            X_pre = self.pre_centroids[i][j].unsqueeze(1)  # To column vector
                            X_new = W @ X_pre + b.unsqueeze(1)
                            self.centroids[i][j] = X_new.squeeze(1)
                            
                else:
                    
                    for i in range(self.config['num_classes']):
                        X_pre = self.pre_centroids[i].unsqueeze(1)  # To column vector
                        X_new = W @ X_pre + b.unsqueeze(1)
                        self.centroids[i] = X_new.squeeze(1)
                        
        if self.pre_covariance is not None:
            with torch.no_grad():
                head = self.model.get_embs_layer()  # The classifier is assumed to be the second head
                W = head.weight.detach()    # Classifier weight
                b = head.bias.detach()      # Classifier bias
                if self.config['clustering']:
                    
                    
                    for i in range(self.config['num_classes']):
                        for j in range(len(self.pre_covariance[i])):
                            X_pre = self.pre_covariance[i][j]
                            X_new = torch.matmul(W, torch.matmul(X_pre, W.t()))
                            self.covariance[i][j] = X_new # + (torch.eye(X_new.shape(1)) * 1e-8) numerical stab-ability
                            
                else:
                    
                    for i in range(self.config['num_classes']):
                        X_pre = self.pre_covariance[i]
                        X_new = torch.matmul(W, torch.matmul(X_pre, W.t()))
                        self.covariance[i] = X_new # + (torch.eye(X_new.shape(1) * 1e-8) numerical stab-ability

    def minimize_mahalanobis(self, embeddings, labels):
        """
        Minimize Mahalanobis distance to class centroids.
        """
        loss = 0
        batch_size = embeddings.size(0)

        for i in range(batch_size):
            emb = embeddings[i]
            class_losses = []
            for c in range(self.config['num_classes']):
                if labels[i] != c:
                    if not self.config['clustering']:
                        centroid = self.centroids[c]
                        cov_inv = torch.inverse(self.covariance[c])
                        diff = (emb - centroid).unsqueeze(0)  # shape [1, embedding_size]
                        mahalanobis_distance = torch.sqrt(diff @ cov_inv @ diff.T).squeeze()
                        class_losses.append(mahalanobis_distance * self.percentages[c])
                    else:
                        for j in range(len(self.centroids[c])):
                            centroid = self.centroids[c][j]
                            cov_inv = torch.inverse(self.covariance[c][j])
                            diff = (emb - centroid).unsqueeze(0)  # shape [1, embedding_size]
                            mahalanobis_distance = torch.sqrt(diff @ cov_inv @ diff.T).squeeze()
                            class_losses.append(mahalanobis_distance * self.percentages[c] * self.intra_percentage[c][j])
            loss += torch.min(torch.stack(class_losses))
        
        loss = loss / batch_size
        return loss

    def unlearn(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):

        

        forget_train_x, forget_val_x, forget_train_y, forget_val_y = train_test_split(forget_X, forget_y, test_size=0.2, stratify=forget_y, random_state=42)

            
        # Create DataLoaders
        val_dataset = ClassificationDataset(forget_val_x,forget_val_y, transforms=transform_forget)
        val_loader_forget = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)

        unlearning_x = retain_X + forget_train_x
        unlearning_y = retain_y + forget_train_y

        train_dataset_unlearn = ClassificationDataset(unlearning_x, unlearning_y, transforms=ConditionalTransform(transform_forget))
        train_loader_unlearn = DataLoader(train_dataset_unlearn, batch_size=self.config["train_batch_size"], shuffle=True, num_workers=4, collate_fn=collate_fn_mixed)

        
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Save hyperparameters
        os.makedirs(self.config['hyperparameters_path'], exist_ok=True)
        hyperparameters = {
            "clustering": self.config['clustering'],
            "robust": self.config['robust'],
            "update_centroids": self.config['update_centroids'],
            "update_covariance": self.config['update_covariance'],
            "embunfreeze": self.config['embunfreeze'],
            "loss_forget": self.config['loss_forget'],
            "forbidden_labels": self.config['forbidden_labels'],
            "fixed_syntetic": self.config['fixed_syntetic'],
            "update_syntetic": self.config['update_syntetic'],
            "syntetic_samples": self.config['syntetic_samples'],
            "covariance_type": self.config.get('covariance_type', 'full'),
            "n_pca_components": self.config.get('n_pca_components'),
        }
        with open(f'{self.config["hyperparameters_path"]}/hyperparameters.yml', 'w') as outfile:
            yaml.dump(hyperparameters, outfile, default_flow_style=False)

        print(f"✅ Hyperparameters saved to {self.config['hyperparameters_path']}/hyperparameters.yml")

        # Optimizer and scheduler set up
        n_step = 0
        print("Starting unlearning process", flush=True)
        
        # Early stopping state
        early_stop_state = {
            'succed_saved': False,
            'best_forget_acc': float('inf'),
            'consecutive_low_count': 0,
        }

        if not self.config["self_forget"]:
            for epoch in range(self.config['max_epoch']):
                self.model.train()
                for batch_idx, (images, target) in tqdm(enumerate(train_loader_unlearn), total=len(train_loader_unlearn)):
                    n_step += 1
                    labels = target.to(self.model.device)
                    self.optimizer.zero_grad()
                    if not isinstance(images, list):
                        images = images.to(self.model.device)
                        if images.dim() <= 2:
                            # The whole batch is synthetic embeddings of the same shape; the collate
                            # stacked them into an (N, D) tensor instead of keeping a list.
                            # Turn it into a list so it goes through the mixed branch below.
                            images = list(images)
                    loss = 0
                    has_real_images = True
                    forget_loss = torch.tensor(0.0, device=self.model.device)
                    loss_syntetic = torch.tensor(0.0, device=self.model.device)
                    n_syntetic = 0
                    if isinstance(images, list):
                        images_aux = []
                        labels_aux = []
                        images_syntetic = []
                        labels_syntetic = []
                        for i in range(len(images)):
                            if images[i].ndim > 1:
                                images_aux.append(images[i].to(self.model.device))
                                labels_aux.append(labels[i].to(self.model.device))
                            else:
                                images_syntetic.append(images[i].to(self.model.device))
                                labels_syntetic.append(labels[i].to(self.model.device))

                        if len(images_aux)>0:
                            images = torch.stack(images_aux)
                            labels = torch.stack(labels_aux)
                        else:
                            # Only synthetic samples, no real images
                            has_real_images = False
                        
                        
                        if len(images_syntetic)>0:
                            images_syntetic = torch.stack(images_syntetic)
                            labels_syntetic = torch.stack(labels_syntetic)
                        
                        
                            if self.config['update_syntetic'] and len(images_syntetic)>0:

                                images_syntetic = self.model.get_embs_layer()(images_syntetic)

                            logits_syntetic = self.model.classifier(images_syntetic)
                            # Hinge margin loss: penalise when gap between top-1
                            # and top-2 logits is below `margin` (default 5).
                            top2 = torch.topk(logits_syntetic, k=2, dim=1).values
                            gap = top2[:, 0] - top2[:, 1]
                            margin = 5.0
                            loss_separate_logits = F.relu(margin - gap).mean()
                            loss_syntetic = self.model.criterion(logits_syntetic, labels_syntetic - self.config['num_classes'])
                            print(f"  Syntetic conservative loss: {loss_syntetic.item():.4f}  margin_loss: {self.lambda_margin*loss_separate_logits.item():.4f}")
                            loss += loss_syntetic + self.lambda_margin*loss_separate_logits

                    else:
                        n_syntetic = len(labels)
                        if self.config['syntetic_samples'] != -1:
                            images = images[labels >= 0]
                            labels = labels[labels >= 0]
                            n_syntetic = n_syntetic - images.size(0)
                                

                    if has_real_images and images.size(0) > 0:
                        if self.config['loss_forget'] != "mahalanobis":
                            logits = self.model(images)
                        else:
                            embeddings = self.model(images, return_only_embs=True)

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

                        elif self.config['loss_forget'] == "mahalanobis": # Minimize Mahalanobis Distance
                            forget_loss = self.minimize_mahalanobis(embeddings, labels)

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
                            raise ValueError("Invalid loss_forget! Choose 'softmax', 'energy', 'confidence_penalty', 'random_class', 'negative_entropy', 'mahalanobis' or 'notop'.")
                        
                        # Log individual components
                        

                    if n_syntetic > 0 or self.config['syntetic_samples'] == -1:
                        syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(n_syntetic) # Fixed number of synthetic samples to evaluate the proxy loss every epoch; can be tuned to the dataset size and the desired variability.
                        syntetic_embeddings = syntetic_embeddings.to(self.model.device)
                        syntetic_labels = syntetic_labels.to(self.model.device)
                        syntetic_logits = self.model.classifier(syntetic_embeddings)
                        # Hinge margin loss on on-the-fly synthetic samples
                        top2 = torch.topk(syntetic_logits, k=2, dim=1).values
                        gap = top2[:, 0] - top2[:, 1]
                        margin = 5.0
                        loss_separate_logits = F.relu(margin - gap).mean()
                        noisy_loss = self.model.criterion(syntetic_logits, syntetic_labels - self.config['num_classes'])
                        loss += noisy_loss + self.lambda_margin*loss_separate_logits
                    
                    loss += forget_loss

                    
                    
                    # 4. Backward pass: compute the gradients
                    loss.backward()
                
                    # 5. Actualizar pesos
                    self.optimizer.step()

                    print(f"Epoch {epoch}, Batch {batch_idx}, Forget Loss: {forget_loss.item() if has_real_images else 'N/A'}, Syntetic Loss: {loss_syntetic.item() if n_syntetic > 0 else 'N/A'}")

                    if (self.config['update_centroids'] or self.config['update_covariance']):
                        self.actualizeCentroidsAndCovariance()
                        self.actualizeGaussianNoise() # Maybe is not estrictly necessary to actualize the gaussian noise at each step, but it is not computationally expensive and ensures that the noise is always consistent with the current centroids and covariance matrix.

                print(f"Epoch {epoch} completed. Validation...", flush=True)
                forget_validation_loss = 0
                acc_forget = 0
                acc_syntetic = 0
                syntetic_validation_loss = 0
                self.model.eval()
                with torch.no_grad():
                    n_total_forget = 0
                    for batch_idx, (images, target) in tqdm(enumerate(val_loader_forget), total=len(val_loader_forget)):
                        images, labels = images.to(self.model.device), target.to(self.model.device)
                        logits = self.model(images)
                        val_loss = self.model.criterion(logits, labels)
                        n_total_forget += labels.size(0)
                        forget_validation_loss += val_loss.item() * images.size(0) # Undo the average to get the total loss over the validation set
                        acc_forget += (logits.argmax(dim=1) == labels).sum().item()
                        


                        syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(images.size(0)) # Fixed number of synthetic samples to evaluate the proxy loss every epoch; can be tuned to the dataset size and the desired variability.
                        syntetic_embeddings = syntetic_embeddings.to(self.model.device)
                        syntetic_labels = syntetic_labels.to(self.model.device)
                        syntetic_logits = self.model.classifier(syntetic_embeddings)
                        syntetic_loss = self.model.criterion(syntetic_logits, syntetic_labels - self.config['num_classes']) # lets remenber that syntetic is created on the fly be careful with the interpretation of this loss, as it is not a real validation loss but a proxy of how well the model is maintaining the distribution of the original data.
                        syntetic_validation_loss += syntetic_loss.item() * syntetic_embeddings.size(0) # Undo the average to get the total loss over the validation set
                        acc_syntetic += (syntetic_logits.argmax(dim=1) == syntetic_labels - self.config['num_classes']).sum().item()

                    syntetic_validation_loss = syntetic_validation_loss / n_total_forget
                    forget_validation_loss = forget_validation_loss / n_total_forget
                    acc_forget = acc_forget / n_total_forget
                    acc_syntetic = acc_syntetic / n_total_forget

                    if acc_forget > 0.5:
                        self.lambda_margin = 0.0

                    else:
                        self.lambda_margin = 0 # min(1.0, (1/((self.config['num_classes']-2) - 1/2.0))*acc_forget - (1/(2*(self.config['num_classes']-2) - 1)))
                        
                    
                print(f"Epoch {epoch} validation completed. \n Forget Loss: {forget_validation_loss}, Syntetic Loss: {syntetic_validation_loss}\n Acc Forget: {acc_forget}, Acc Syntetic: {acc_syntetic}\n \n \n", flush=True)

                # Check if checkpoint should be saved / early stopping
                checkpoints_to_save, stop = should_save_checkpoint(epoch, acc_forget, self.config['num_classes'], self.config, early_stop_state)
                for ckpt_name, reason in checkpoints_to_save:
                    print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                    self.save_results(epoch, n_step, filename=ckpt_name)
                if stop:
                    print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                    break

        else:
            for epoch in range(self.config["max_epoch"]):
                self.model.train()
                for batch_idx, (images, target) in tqdm(
                    enumerate(train_loader_unlearn), total=len(train_loader_unlearn)
                ):
                    n_step += 1
                    labels = target.to(self.model.device)
                    self.optimizer.zero_grad()
                    if not isinstance(images, list):
                        images = images.to(self.model.device)

                    labels = labels - self.config["num_classes"]  # Shift labels for synthetic samples
                    loss = torch.tensor(0.0, device=self.model.device)
                    forget_loss = torch.tensor(0.0, device=self.model.device)
                    loss_syntetic = torch.tensor(0.0, device=self.model.device)

                    retain_data = [(images[i], labels[i]) for i in range(len(labels)) if labels[i] in self.config["retain_labels"]]
                    forget_data = [(images[i], labels[i]) for i in range(len(labels)) if labels[i] not in self.config["retain_labels"]]

                    if retain_data:
                        retain_images, retain_labels = zip(*retain_data)
                        retain_images = torch.stack(retain_images)
                        retain_labels = torch.stack(retain_labels)
                    
                    if forget_data:
                        forget_images, forget_labels = zip(*forget_data)
                        forget_images = torch.stack(forget_images)
                        forget_labels = torch.stack(forget_labels)

                    n_syntetic = len(forget_labels) if self.config["syntetic_samples"] == -1 else len(labels) - len(retain_data) - len(forget_data)

                    if self.config["update_syntetic"] and len(retain_images) > 0:
                        retain_images = self.model.get_embs_layer()(retain_images)

                    logits_syntetic = self.model.classifier(retain_images)
                    # Hinge margin loss: penalise when gap between top-1
                    # and top-2 logits is below `margin` (default 5).
                    top2 = torch.topk(logits_syntetic, k=2, dim=1).values
                    gap = top2[:, 0] - top2[:, 1]          # (B,)
                    margin = 5.0
                    loss_separate_logits = F.relu(margin - gap).mean()
                    loss_syntetic = self.model.criterion(
                        logits_syntetic, retain_labels
                    )
                    print(f"  Syntetic conservative loss: {loss_syntetic.item():.4f}  margin_loss: {self.lambda_margin*loss_separate_logits.item():.4f}")
                    loss = loss + loss_syntetic + self.lambda_margin*loss_separate_logits


                    if forget_images.size(0) > 0:
                        if self.config["loss_forget"] != "distances":
                            if self.config["update_syntetic"]:
                                forget_images = self.model.get_embs_layer()(forget_images)

                            logits = self.model.classifier(forget_images)
                            
                        else:
                            if self.config["update_syntetic"]:
                                forget_images = self.model.get_embs_layer()(forget_images)

                            embeddings = forget_images

                        if self.config["loss_forget"] == "softmax":
                            uniform_target = torch.ones_like(logits) / logits.size(1)
                            forget_loss = F.kl_div(
                                F.log_softmax(logits, dim=1), uniform_target, reduction="batchmean"
                            ) 
                            
                        elif self.config["loss_forget"] == "energy":
                            energy = 1.0 * torch.logsumexp(logits / 1.0, dim=1)
                            forget_loss = torch.mean(F.softplus(energy))

                        elif self.config["loss_forget"] == "confidence_penalty":
                            probs = torch.softmax(logits, dim=1)
                            forget_loss = (probs * torch.log(probs + 1e-12)).sum(dim=1).mean() 

                        elif self.config["loss_forget"] == "random_class":
                            random_labels = torch.randint(
                                0, self.config["num_classes"], (logits.size(0),), device=logits.device
                            )
                            forget_loss = self.model.criterion(logits, random_labels) 

                        elif self.config["loss_forget"] == "negative_entropy":
                            forget_loss = -self.model.criterion(logits, labels) 

                        elif self.config["loss_forget"] == "distances":
                            forget_loss = self.minimize_distances(embeddings, labels)

                        elif self.config["loss_forget"] == "notop":  # Not-on-top hinge loss
                            cols = torch.arange(self.config["num_classes"]).repeat(logits.size(0), 1)
                            cols = cols.to(logits.device)
                            mask = cols != labels.unsqueeze(1)
                            mask = mask.to(logits.device)
                            other_logits = logits.masked_select(mask).view(logits.size(0), -1)
                            best_logits = torch.topk(other_logits, k=1, dim=1).values
                            mylogit = logits[:, labels]
                            forget_loss = F.relu(mylogit - best_logits + 10).mean()

                        else:
                            raise ValueError(
                                "Invalid loss_forget! Choose 'softmax', 'energy', 'confidence_penalty', "
                                "'random_class', 'negative_entropy', 'distances' or 'notop'."
                            )

                    if not self.config["fixed_syntetic"]:
                        syntetic_embeddings, syntetic_labels = self.generate_syntetic_embeddings(n_syntetic, self.config['forbidden_labels'])
                        syntetic_embeddings = syntetic_embeddings.to(self.model.device)
                        syntetic_labels = syntetic_labels.to(self.model.device)
                        syntetic_logits = self.model.classifier(syntetic_embeddings)
                        # Hinge margin loss on on-the-fly synthetic samples
                        top2 = torch.topk(syntetic_logits, k=2, dim=1).values
                        gap = top2[:, 0] - top2[:, 1]
                        margin = 5.0
                        loss_separate_logits = F.relu(margin - gap).mean()
                        noisy_loss = self.model.criterion(
                            syntetic_logits, syntetic_labels - self.config["num_classes"]
                        )
                        loss = loss + noisy_loss + self.lambda_margin*loss_separate_logits
                    loss = loss + forget_loss
                    loss.backward()
                    self.optimizer.step()

                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Forget Loss: {forget_loss.item() if has_real_images else 'N/A'}, "
                        f"Syntetic Loss: {loss_syntetic.item() if n_syntetic > 0 else 'N/A'}"
                    )

                    # if self.config["update_centroids"] or self.config["update_covariance"]:
                    #     self.actualizeCentroidsAndCovariance()
                    #     self.actualizeGaussianNoise()

                # ----- Validation -------------------------------------------------
                print(f"Epoch {epoch} completed. Validation...", flush=True)
                forget_validation_loss = 0
                acc_forget = 0
                acc_syntetic = 0
                syntetic_validation_loss = 0
                self.model.eval()
                with torch.no_grad():
                    n_total_forget = 0
                    for batch_idx, (images, target) in tqdm(
                        enumerate(val_loader_forget), total=len(val_loader_forget)
                    ):
                        images = images.to(self.model.device)
                        labels = target.to(self.model.device)
                        logits = self.model(images)
                        val_loss = self.model.criterion(logits, labels)
                        n_total_forget += labels.size(0)
                        forget_validation_loss += val_loss.item() * images.size(0)
                        acc_forget += (logits.argmax(dim=1) == labels).sum().item()

                        # se, sl = self.generate_syntetic_embeddings(images.size(0), self.config['forbidden_labels'])
                        # se = se.to(self.model.device)
                        # sl = sl.to(self.model.device)
                        # s_logits = self.model.classifier(se)
                        # s_loss = self.model.criterion(s_logits, sl - self.config["num_classes"])
                        # syntetic_validation_loss += s_loss.item() * se.size(0)
                        # acc_syntetic += (s_logits.argmax(dim=1) == sl - self.config["num_classes"]).sum().item()

                    syntetic_validation_loss /= max(n_total_forget, 1)
                    forget_validation_loss /= max(n_total_forget, 1)
                    acc_forget /= max(n_total_forget, 1)
                    acc_syntetic /= max(n_total_forget, 1)

                    if acc_forget > 0.5:
                        self.lambda_margin = 0.0

                    else:
                        self.lambda_margin =  0 # min(1.0, (1/((self.config['num_classes']-2) - 1/2.0))*acc_forget - (1/(2*(self.config['num_classes']-2) - 1)))

                print(
                    f"Epoch {epoch} validation completed.\n"
                    f" Forget Loss: {forget_validation_loss}, Syntetic Loss: {syntetic_validation_loss}\n"
                    f" Acc Forget: {acc_forget}, Acc Syntetic: {acc_syntetic}\n\n"
                )

                # Check if checkpoint should be saved / early stopping
                checkpoints_to_save, stop = should_save_checkpoint(epoch, acc_forget, self.config['num_classes'], self.config, early_stop_state)
                for ckpt_name, reason in checkpoints_to_save:
                    print(f"Saving checkpoint at epoch {epoch}: {reason}", flush=True)
                    self.save_results(epoch, n_step, filename=ckpt_name)
                if stop:
                    print(f"🛑 Early stopping at epoch {epoch}", flush=True)
                    break

        return self.model