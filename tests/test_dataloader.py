import torch.utils.data as data
from omegaconf import OmegaConf

import videoaugment.loader as VL

dataset_config = OmegaConf.load('./videoaugment/config/dataset.yaml')['dataset']
test_params = {'batch_size': dataset_config.dataloader.test.batch_size,
               'shuffle': dataset_config.dataloader.test.shuffle,
               'num_workers': dataset_config.dataloader.test.num_workers}
val_params = {'batch_size': dataset_config.dataloader.validation.batch_size,
              'shuffle': dataset_config.dataloader.validation.shuffle,
              'num_workers': dataset_config.dataloader.validation.num_workers,
              'pin_memory': True}

train_params = {'batch_size': dataset_config.dataloader.train.batch_size,
                'shuffle': dataset_config.dataloader.train.shuffle,
                'num_workers': dataset_config.dataloader.train.num_workers,
                'pin_memory': True}

train_prefix = "train"
dev_prefix = "dev"
test_prefix = "test"

training_set = VL.VideoDataset(config=dataset_config, mode=train_prefix, classes=classes)
training_generator = data.DataLoader(training_set, **train_params)
validation_set = VL.VideoDataset(config=dataset_config, mode=dev_prefix, classes=classes)
validation_generator = data.DataLoader(validation_set, **test_params)
test_set = VL.VideoDataset(config=dataset_config, mode=test_prefix, classes=classes)
test_generator = data.DataLoader(test_set, **test_params)
