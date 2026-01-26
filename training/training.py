# Libraries
from copy import deepcopy
import time
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import os
import torch.nn.functional as F
# import wandb
import math

from training.loss import loss_WD
# from utils.dataset import use_prediction

class Trainer(object):
    '''Training class
    -------
    optimizer: torch.optim
        model optimizer (e.g., Adam, SGD)
    loss_type: str
        options: 'RMSE', 'MAE'
    '''
    def __init__(self, optimizer, lr_scheduler=None, max_epochs=1, type_loss='RMSE',
                 report_freq: int=1, patience=3, device='cpu', sampling_ratio = 0.8, 
                 samples_per_step = 5,
                 **training_options):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sampling_ratio = sampling_ratio
        self.samples_per_step = samples_per_step
        self.training_options = training_options
        self.device = device

        self.type_loss = type_loss
        assert type_loss in ['RMSE','MAE'], "loss_type must be either 'RMSE' or 'MAE'"

        self.epoch = 0
        self.max_epochs = max_epochs
        self.report_freq = report_freq
        self.patience = patience

        #create vectors for the training and validation loss
        self.train_losses = []
        self.val_losses = []

        self.early_stop = 0
        self.best_val_loss = 1

    def _training_step(self, model, train_loader, curriculum_epoch=1, 
                      **loss_options):
        '''Function to train the model for one epoch
        ------
        model: nn.Model
            e.g., GNN model
        train_loader: DataLoader
            data loader for training dataset
        curriculum_epoch: int 
            every curriculum_epoch epochs the training window expands (default=1)
        loss_options: dict (see loss.py for more info)
            velocity_scaler: float
                weight loss for velocity terms (default=1)
        '''
        model.train()
        losses = []

        bar_freq = 0 if self.report_freq == 0 else (self.epoch) % self.report_freq == 0
        
        charge_bar = tqdm(train_loader, leave=bar_freq, disable=True)
        for batch in charge_bar:
            # reset gradients
            self.optimizer.zero_grad()

            batch_size = batch.shape[0]

            edge_index_dict = batch[i]["edge_index_dict"]
            edge_attribute_dict = batch[i]["edge_attr_attr"]

            for i in range(batch_size):

                train_metastation_mask = torch.tensor(batch[i]['metastation_mask'], dtype=torch.bool).to(self.device)
                train_rainfallstation_mask = torch.tensor(batch[i]['rainfallstation_mask'], dtype=torch.bool).to(self.device)
                step_loss = 0
                step_count = 0

                training_metastation_indices = train_metastation_mask.nonzero(as_tuple=False)
                training_rainfallstation_indices = train_rainfallstation_mask.nonzero(as_tuple=False)
                gen_x = batch[i]['gen_x']  # [batch_size, num_gen_nodes, gen_features]
                rain_x = batch[i]['rain_x']  # [batch_size, num_rain_nodes, rain_features]
                gen_y = batch[i]['gen_y']
                rain_y = batch[i]['rain_y']

                for idx in training_metastation_indices:
                    gen_x_masked=gen_x[i].clone()
                    rain_x_masked=rain_x[i].clone()
            
                    gen_x_masked[~train_metastation_mask.bool()] = 0
                    rain_x_masked[~train_rainfallstation_mask.bool()] = 0
                    gen_x_masked[idx] = 0

                    x_dict = {
                    'general_station': gen_x_masked,
                    'rainfall_station': rain_x_masked,
                    }
                    # print(x_dict)
                    self.optimizer.zero_grad()
                    out = model(x_dict, edge_index_dict, edge_attribute_dict)

                    # Model prediction
                    gen_predictions = out['general_station'][idx]
                    gen_actual = gen_y[i][idx]

                    training_loss = F.mse_loss(gen_predictions, gen_actual) 
                    step_loss += training_loss
                    step_count += 1

                for idx in training_rainfallstation_indices:
                    gen_x_masked=gen_x[i].clone()
                    rain_x_masked=rain_x[i].clone()
            
                    gen_x_masked[~train_metastation_mask.bool()] = 0
                    rain_x_masked[~train_rainfallstation_mask.bool()] = 0
                    rain_x_masked[idx] = 0

                    x_dict = {
                    'general_station': gen_x_masked,
                    'rainfall_station': rain_x_masked,
                    }
                    # print(x_dict)
                    self.optimizer.zero_grad()
                    out = model(x_dict, edge_index_dict, edge_attribute_dict)

                    # Model prediction
                    rain_predictions = out['rainfall_station'][idx]
                    rainfall_actual = rain_y[i][idx]

                    training_loss = F.mse_loss(rain_predictions, rainfall_actual)
                    step_loss.append(training_loss)
                
                loss = torch.stack(step_loss).mean()
                losses.append(loss.detach())

                #backpropagate
                loss.backward()


                # Update weights
                self.optimizer.step()


        losses = torch.stack(losses).mean().item()

        return losses
        
    def fit(self, model, train_loader, val_dataset, **temporal_test_dataset_parameters):
        assert isinstance(train_loader, DataLoader), "Training requires train_loader to be a Dataloader object"

        #start measuring training time
        start_time = time.time()
        
        torch.autograd.set_detect_anomaly(True)

        progress_bar = tqdm(range(self.epoch, self.max_epochs), total=self.max_epochs,
                            initial=self.epoch,leave=True, 
                            bar_format='{percentage:3.0f}%|{bar:30}| '\
                            'Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        try:
            for _ in progress_bar:
                self.epoch += 1

                # Model training
                train_loss = self._training_step(model, train_loader, **self.training_options)
                
                progress_bar.set_description(f"\tTrain loss = {train_loss:4.4f}   "\
                                            f"Valid loss = {val_loss:1.4f}    "\
                                            )

                # wandb.log({"train_loss": train_loss,
                #            "valid_loss": val_loss})

                self.train_losses.append(train_loss)
                # self.val_losses.append(val_loss) 
                
                self._use_learning_rate_scheduler()
                self._update_best_model(model)
                if self._early_stopping():
                    break
        except KeyboardInterrupt:
            self.epoch -= 1
            
        self.training_time = time.time() - start_time

        min_val_loss = torch.tensor(self.val_losses).min()
        argmin_val_loss = torch.tensor(self.val_losses).argmin()

        # wandb.log({"training_time": self.training_time,
        #            "valid_loss": min_val_loss
        #            })

        # try:
        #     print("Loading best model...")
        #     model.load_state_dict(self.best_model)
        # except:
        #     pass #avoid blocking processes even if loss is very bad

    # def _save_model(self, model, model_name="best_model.h5", save_dir=None):
    #     '''Save model in directory'''
    #     if save_dir is None:
    #         save_dir = wandb.run.dir
    #     save_dir = os.path.join(save_dir, model_name)
    #     torch.save(model.state_dict(), save_dir)

    # def _load_model(self, model, model_name="best_model.h5", save_dir=None):
    #     '''Load model from directory'''
    #     if save_dir is None:
    #         save_dir = wandb.run.dir
    #     save_dir = os.path.join(save_dir, model_name)
    #     model.load_state_dict(torch.load(save_dir, map_location=self.device))

    def _early_stopping(self):
        '''Stop training if validation keeps increasing'''
        should_stop = False

        if len(self.val_losses) < 3:
            pass
        elif self.val_losses[-1]>=self.val_losses[-2]:
            self.early_stop += 1
        else:
            self.early_stop = 0
        
        if self.early_stop == self.patience:
            print("Early stopping! Epoch:", self.epoch)
            should_stop = True

        return should_stop

    def _use_learning_rate_scheduler(self):
        '''If present, use a learning rate scheduler step'''
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _update_best_model(self, model):
        '''Saves the model with best validation loss'''
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            self.best_model = deepcopy(model.state_dict())
