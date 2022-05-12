import time
import copy
import torch
import logging
import os
from math import floor
import pandas as pd
import numpy as np
# import nvidia_smi

def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    file_mode = 'a' if os.path.isfile(log_file) and resume else 'w'

    root_logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # Remove all existing handlers (can't use the `force` option with
    # python < 3.8)
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    # Add the handlers we want to use
    fileout = logging.FileHandler(log_file, mode=file_mode)
    fileout.setLevel(logging.DEBUG)
    fileout.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fileout)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(console)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=200, scheduler=None, save_path="./", show_iteration_info=True, model_name=None, limit_num_epochs=100):
    """
    Function to train a deep learning model. 
        model: a pytorch model 
        dataloaders: a dictionary with 'train' and 'val' keys with their respective dataloaders
        criterion: the loss function 
        optimizer: the optimizer to be used
        num_epochs: number of epochs to train
        scheduler: if it is specified, the scheduler will be used
        save_path: the directory where we can save the model and the log output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_path = os.path.join(save_path, 'models')

    path_created = False
    # Check whether the specified path exists or not
    if not os.path.exists(save_path):
        # Create a new directory because it does not exist 
        os.makedirs(save_path)
        path_created = True

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    setup_logging(os.path.join(save_path, 'log.txt'))

    if path_created:
        logging.info(f"'{save_path}' Directory was created!")

    logging.info(f"Starting training on: device={device}\n")
    logging.info("")
    # nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    # logging.info("Device: {}".format(nvidia_smi.nvmlDeviceGetName(handle)))
    # logging.info("")

    if model_name is not None:
        logging.info(f"Training {model_name}")

    logging.info("Model information: ")
    total_params = 0
    for x in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    logging.info(f"\tTotal number of params: {round(total_params/1e6, 2)}M")
    logging.info(f"\tTotal layers {len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, model.parameters())))}")

    model_name = '' if model_name is None else model_name
    logging.info(f"Optimizer: {optimizer}")
    if scheduler is not None:
        logging.info(f"Scheduler: {scheduler.state_dict()}")

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_loss_updated_on = 0  # start at epoch 0
    df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_time", "val_time"])
    model.to(device)
    since = time.time()

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        info = {'train': None, 'val': None}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_number_of_samples = 0

            # Because of this, we take the time for each epoch during training.
            epoch_start = time.time()

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_number_of_samples += labels.shape[0]
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            f'Loss {loss.item()} (avg: {running_loss/running_number_of_samples})\t'
                            .format(
                                epoch, i, len(dataloaders[phase]),
                                phase=phase))  
                
                if show_iteration_info: 
                    logging.info(report)

            epoch_finish = time.time() - epoch_start 
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss, ))
            info[phase] = (epoch_loss, epoch_finish)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss_updated_on = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 100 == 0:
                    torch.save(model.state_dict(), os.path.join(models_path, f"{model_name if model_name is not '' else 'model'}_e{epoch}.pth"))

            if phase == 'val':
                val_loss_history.append(epoch_loss)

        # Save info on dataframe 
        train_loss = info['train'][0]
        train_time = info['train'][1]
        test_loss = info['val'][0]
        test_time = info['val'][1]

        df.loc[len(df.index)] = [epoch, train_loss, test_loss, train_time, test_time]

        if  epoch - best_loss_updated_on > limit_num_epochs:
            logging.info(f"Terminating training because accuracy has not improved in last {limit_num_epochs} epochs")
            break

        # make a step on the scheduler every epoch
        if scheduler is not None:
            scheduler.step()

        logging.info("")

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val loss: {:4f}'.format(best_loss))
    df.to_csv(os.path.join(save_path, 'training_info.csv'), index=False)
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(models_path, f"{model_name if model_name is not '' else 'model'}_final.pth"))
    return model, val_loss_history
