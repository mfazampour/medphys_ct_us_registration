##  TO LAUNCH POLYAXON GROUP
- Before launching on polyaxon check:
1. limit_train_batches is commented
```python
trainer = Trainer(
    ...,
    # limit_train_batches=0.04,  # use 0.2 for Polyaxon, use 0.03 to avoid memory error on Anna's computer
    # limit_val_batches=0.06,  # use 0.4 for Polyaxon, use 0.05 to avoid memory error on Anna's computer
)
```

2. num training epochs is correct
3. num_workers is correct

```yaml
min_epochs : 50
max_epochs : 150
num_workers : 6
```

### 1. Launching script for evaluating network loss/positive weigths usage:
```polyaxon run -f polyaxonfile.yaml -u -l```

### 2. Launching script for evaluating network different augmentations usage
```polyaxon run -f polyaxonfile_evaluate_probs.yaml -u -l```

### 3. Launch pix2pix net
```polyaxon run -f polyaxonfile_pix2pix.yaml -u -l```



## TO LAUNCH TENSORBOARD ON POLYAXON ############

For group: 
```polyaxon tensorboard -g <group_id> start -f polyaxon_tensorboard.yaml```

For experiment:
```polyaxon tensorboard -xp 50846 start -f polyaxon_tensorboard****.yaml```


## LAUNCHING TENSORBOARD ON REMOTE WORKSTATION ##########
```
ssh -L 16006:127.0.0.1:6006 mariatirindelli@nevarro.ifl
source /home/mariatirindelli/mariaenv/bin/activate

cd /tmp/pycharm_project_932
tensorboard --logdir .
```

Then, on your local machine simply open browser and paste
```http://127.0.0.1:16006/```


## TO LAUNCH script.sh on remote workstation ############

from cmd
```
cd C:\GitRepo\pyfl\remote_debugging
win_batch.bat

cd /mnt/data/mariatirindelli/pyfl/remote_debugging
sed -i -e 's/\r$//' script.sh
./script.sh 
```


## TO COPY TO REMOTE WS
```
scp D:\NAS\FacadesDataset.zip mariatirindelli@10.23.0.56:/mnt/data/mariatirindelli
```
scp E:\Chrissi\US_augmentation\public_datasets\BoneClassification.zip tirindellim@campdata.in.tum.de:/home/public_html/files/VertebraeUltrasound



scp mariatirindelli@10.23.0.56:/mnt/data/mariatirindelli/output/Pix2Pix/myres.zip D:\NAS

scp mariatirindelli@10.23.0.54:/home/mariatirindelli/model_traning/results/210513-142855_GanModule_MixedDb_pix2pix_Pix2PixModel/pix2pix_o.zip D:\NAS
scp mariatirindelli@10.23.0.54:/home/mariatirindelli/model_traning/results/210513-143217_GanSemiUnpairedModule_MixedDb_pix2pix_Pix2PixModel/mixed_o.zip D:\NAS


scp mariatirindelli@10.23.0.54:/home/mariatirindelli/cut_model/A.mhd E:\A.mhd

scp C:\Users\maria\Downloads\CampDb.zip mariatirindelli@10.23.0.54:/home/mariatirindelli/model_training

## TO CREATE POLYAXON PROJECT
```
polyaxon project create --name CIFAR10_Example
polyaxon init CIFAR10_Example  --polyaxonfile

## PORT SPECIFIC TUNNEL SSH
ssh -L18097:localhost:8097 mariatirindelli@nevarro.ifl
```


## to run groups on polyaxon: 
hptuning:
  matrix:
    ct_weight:
      values: [0.5, 0.2, 0.35, 0.7]

wandb login
