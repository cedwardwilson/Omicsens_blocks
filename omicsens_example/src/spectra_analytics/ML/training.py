""" Code to train models"""

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import os
import time
from math import ceil
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from visdom import Visdom


class RegressorTraining:
    """Class for regressor training."""

    def __init__(self, sampled_dataset, test_size, random_state, single_regressor):
        self.sampled_dataset = sampled_dataset
        self.spectra = sampled_dataset['spectra']
        self.weights = sampled_dataset['weights']
        self.weights_names = sampled_dataset['weight_columns']
        spectra_train, spectra_test, weights_train, weights_test = train_test_split(
            self.spectra, self.weights, test_size=test_size, random_state=random_state
        )
        self.spectra_train = spectra_train
        self.spectra_test = spectra_test
        self.weights_train = weights_train
        self.weights_test = weights_test

        multi_output_regressor = MultiOutputRegressor(single_regressor)
        self.multioutput_regressor = multi_output_regressor
        self.train_score = None
        self.test_score = None

    def run_training(self):
        """Run training for the model."""
        self.multioutput_regressor.fit(self.spectra_train, self.weights_train)
        self.multioutput_regressor.weights_names = self.weights_names
        self.train_score = self.multioutput_regressor.score(self.spectra_train, self.weights_train)
        self.test_score = self.multioutput_regressor.score(self.spectra_test, self.weights_test)


class DeepLearningTrainingData:
    """Class to generate consistent deep learning data."""

    def __init__(
        self,
        sampled_dataset,
        test_size,
        random_state,
        validation_size,
        batch_size,
        number_of_epochs,
        learning_rate,
        device=None,
    ):
        self.sampled_dataset = sampled_dataset
        self.spectra = sampled_dataset['spectra']
        self.weights = sampled_dataset['weights']
        self.spectra_tensor = torch.tensor(self.spectra, dtype=torch.float32)
        self.weight_tensor = torch.tensor(self.weights, dtype=torch.float32)

        spectra_train, spectra_test, weights_train, weights_test = train_test_split(
            self.spectra_tensor, self.weight_tensor, test_size=test_size, random_state=random_state
        )

        spectra_train, spectra_validation, weights_train, weights_validation = train_test_split(
            spectra_train,
            weights_train,
            test_size=validation_size,
            random_state=random_state,
        )
        self.spectra_test = spectra_test
        self.spectra_train = spectra_train
        self.spectra_validation = spectra_validation
        self.weights_test = weights_test
        self.weights_train = weights_train
        self.weights_validation = weights_validation
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        if device is None:
            device = 'cpu'
        assert device in ('cpu', 'cuda')
        if device == 'cuda':
            assert torch.cuda.is_available()
        self.device = device

        # load dataset with pytorch function
        train_dataset = TensorDataset(self.spectra_train, self.weights_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.epoch_train_size = ceil(len(train_dataset) / self.batch_size)

        self.validation_dataset = TensorDataset(self.spectra_validation, self.weights_validation)
        self.validation_loader = DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.epoch_validation_size = ceil(len(self.validation_dataset) / self.batch_size)


class DeepLearningTraining:
    """Class for deep learning training."""

    def __init__(self, dl_training_data, neural_network, criterion, optimizer):
        self.dl_training_data = dl_training_data
        self.neural_network = neural_network
        self.criterion = criterion
        self.optimizer = optimizer

    def run_training(self, filename_to_save):
        mini_validation_loss_value = 100000
        """run training for instantiated object."""
        training_start_time = time.time()
        # #to print in the window the loss
        vis = Visdom(port='8097', env=filename_to_save)
        vis.close()

        for epoch in range(
            self.dl_training_data.number_of_epochs
        ):  # loop over the dataset multiple times
            print('epoch : ', epoch)

            running_training_loss = 0.0
            running_validation_loss = 0.0
            self.neural_network.train()

            for batch_idx, data in enumerate(self.dl_training_data.train_loader, 0):
                print("batch idx", batch_idx)

                # get the inputs
                spectra, weights = data

                spectra = spectra.to(self.dl_training_data.device)
                weights = weights.to(self.dl_training_data.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                predictions = self.neural_network(spectra)
                print("weights : ", weights[0])
                print("predictions :", predictions[0])
                loss = self.criterion(predictions, weights)

                loss.backward()

                self.optimizer.step()

                running_training_loss += loss.item()  # to compute the loss for each epoch

            with torch.no_grad():
                self.neural_network.eval()

                for batch_idx, data in enumerate(self.dl_training_data.validation_loader, 0):
                    # print("validation batch idx :",batch_idx)

                    spectra, weights = data
                    spectra = spectra.to(self.dl_training_data.device)
                    weights = weights.to(self.dl_training_data.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    predictions = self.neural_network(spectra)
                    validation_loss = self.criterion(weights, predictions)

                    # # compute R2 score
                    # R2.update(predictions, weights)
                    # R2score_val = R2.compute()

                    running_validation_loss += validation_loss.item()
                    # running_validation_R2 += R2score_val.item()

            # #Compute of the losses
            train_loss_value = running_training_loss / self.dl_training_data.epoch_train_size
            validation_loss_value = (
                running_validation_loss / self.dl_training_data.epoch_validation_size
            )

            vis.line(
                Y=[train_loss_value],
                X=[epoch],
                update='append',
                win='loss',
                name='training loss',
                opts={'showlegend': True, 'title': "train loss"},
            )
            vis.line(
                Y=[validation_loss_value],
                X=[epoch],
                update='append',
                win='loss',
                name='validation loss',
                opts={'showlegend': True, 'title': "validation loss"},
            )


            checkpoint = {
                'config': self.neural_network.get_config(),
                "state_dict": self.neural_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            #save the smallest validation
            if validation_loss_value < mini_validation_loss_value:
                mini_validation_loss_value = validation_loss_value
                torch.save(checkpoint, os.path.join('trained_models',filename_to_save))


        print("training is done! The training time was", time.time() - training_start_time)
