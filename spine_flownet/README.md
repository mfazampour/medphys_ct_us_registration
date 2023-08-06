# Thesis
Thesis research

For this work, we used as base the [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch) and as brief guidence from [Fu et al.](https://pubmed.ncbi.nlm.nih.gov/33129147/) for tweeks and preparation of data.
In [Preprocessing.ipynb](https://github.com/lameski123/prethesis/blob/main/Preprocessing.ipynb) a pipeline of data creation and pre-processing.
In [model.py](https://github.com/lameski123/prethesis/blob/main/model.py) is the tweek of the original Flownet3D. 
In [data.py](https://github.com/lameski123/prethesis/blob/main/data.py) is the data loader. 
In [util.py](https://github.com/lameski123/prethesis/blob/main/util.py) the implementation of functions needed to make the model work properly (these functions are CUDA dependent for more information I would recomend to check [Flownet3D in pytorch](https://github.com/hyangwinter/flownet3d_pytorch)).
In [test_model.py](https://github.com/lameski123/prethesis/blob/main/distError.py) we produced test results on unseen data during training.

# Installation

*tested with python==3.8*

```
git clone https://github.com/lameski123/thesis
git checkout thesis
pip install -r requirements.txt
pip install chamferdist
cd pointnet2
python setup.py install
```

# Train
Training is done using the train.py file. you will need a WandB account to see the results. there are a couple of 
options for loss. see the z`options.py` file for that. 

`python train.py --dataset_path %DATASET_PATH% --loss rigidity`

# Test

Test is done using the test.py file. two arguments are mandatory, data path and the save model path. 

`python test.py --dataset_path %DATASET_PATH% ----model_path %MODEL_PATH%`

# Running on the cluster

we need the cuda exec. to compile the libraries. so we should pull a `devel` image not `runtime`

the code is tested on `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel`.

## After building the image to run locally or on the cluster:

First run the container:

`docker run -i -t --mount src=/home/farid/spine_flownet,target=/src,type=bind --mount src=/mnt/polyaxon/,target=/mnt/polyaxon/,type=bind  --shm-size=10g --entrypoint bash -u 0 mfazampour/spine_flownet`

then install the missing libraries:

`pip install matplotlib pyquaternion`

go to source code:

`cd /src/`

and finally:

`python train.py --wandb_key d3424a60f5e39087781fde8ff973ee92dd6da70d --epoch 60 --use_raycasted_data
--batch_size 35 --gpu_id 1 --loss mse,biomechanical --test_id 1 --dataset_path /mnt/polyaxon/data1/Spine_Flownet/new_data_raycasted_cpd_initialized_real_paired --coeff_rigidity 5.0 --num_points 2048 --no_augmentation --coeff_bio 0.5`

# Data Generation

The data are generated from the VerseDb Segmentation, according to the following pipeline: 

### 1. Run imfusion_workspace/volume_segmentation2pc.iws (example batch file in imfusion_workspace/volume_segmentation2pc.txt)
   
The workspace extractes point clouds Generation from Spine segmentations. 
The placeholders in the .iws file are 

- INPUTFILE = The path to the .mhd file containing a verseDb spine segmentation

- OLDVALUES = The values of the lumbar spine to be set to zeros. Specifically, for each lumbar spine, 5 point clouds
are generated, one for each vertebra. That is, the imfusion_workspace/volume_segmentation2pc.iws should be launched
5 times for each spine. At each iteration, all the lumbar vertebrae labels (OLDVALUES) are set to 0 except for one, 
corresponding to the vertebra from which the point cloud is extracted.

- OUTPUTSTL = The path where the .stl for each vertebrae will be saves

- OUTPUTPC = The path where the .txt point cloud for each vertebrae will be saves

The example batch file in imfusion_workspace/volume_segmentation2pc.txt generates the output .stl
and .txt point clouds for 5 lumbar vertebrae, extracted the from verse075_seg.nii.gz. 

The example batch file reads segmentation data in PCSpineRegistration\source_data\Spine<k>\verse<id>_seg.nii.gz and 
generates the stl models: 
PCSpineRegistration\source_data\Spine<k>\v1.stl, PCSpineRegistration\source_data\Spine<k>\v2.stl, 
PCSpineRegistration\source_data\Spine<k>\v3.stl, PCSpineRegistration\source_data\Spine<k>\v4.stl, 
PCSpineRegistration\source_data\Spine<k>\v5.stl 
and the .txt point clouds models: 
PCSpineRegistration\source_data\Spine<k>\v1.txt, PCSpineRegistration\source_data\Spine<k>\v2.txt, 
PCSpineRegistration\source_data\Spine<k>\v3.txt, PCSpineRegistration\source_data\Spine<k>\v4.txt, 
PCSpineRegistration\source_data\Spine<k>\v5.txt for all the input spines <k>. 

The ImFusion workspace imfusion_workspace/volume_segmentation2pc.iws performs the following operations:
1. Thresholded in a way to only leave the lumbar vertebrae 
2. Converted to a mesh
3. Rescaled of a factor 0.001 as sofa expects point clouds to be passed in meters and not mm

### 2. Generate the sofa xml scenes
    
   @todo: add process to generate the sofa scenes (1. generate springs connection + copy point clouds coordinates
   and spring connections coordinates in the xml sofa framework + launch the simulation)
   Save the scenes in PCSpineRegistration\sofa_scenes

### 3. Launch the sofa scenes
The sofa scenes will generate the output .vtu. The files are manually ordered in the directory 
PCSpineRegistration\vtu_data\Spine<k> for each k-th spine. The PCSpineRegistration\vtu_data\spine<k> will contain the
following files: 
1. The undeformed spines, saved as PCSpineRegistration\vtu_data\spine<k>\spine1_vert<vert_id>0.vtu 
   with <vert_id> = (1, 2, 3, 4, 5), (example: PCSpineRegistration\vtu_data\spine<k>\spine1_vert10.vtu, 
   PCSpineRegistration\vtu_data\spine<k>\spine1_vert20.vtu, PCSpineRegistration\vtu_data\spine<k>\spine1_vert30.vtu, ...)

2. The deformed spines save as PCSpineRegistration\vtu_data\spine<k>\spine1_vert<vert_id>1_<t_>_<sofa_id_>.vtu with
   - <vert_id> = (1, 2, 3, 4, 5) = the vertebral level
   - <t_> = range(0, 35) = the timestamp of the simulation
   - <sofa_id_> = (0, 1) a sofa id that gets updated every 20 iterations

### 4. Launch the "XML to txt" in the Preprocessing.ipynb notebook to convert the .vtu files into .txt
   The script reads the vtu files and generate source and target .txt point clouds, which are saved in: 
   PCSpineRegistration\source_target_pc\Spine<k>\spine1_vert<vert_id>1_<t_>_<id_>.txt and 
   PCSpineRegistration\vtu_data\spine<k>\spine1_vert<vert_id>0.vtuwith
- <vert_id> = (1, 2, 3, 4, 5) = the vertebral level
- <t_> = range(0, 35) = the timestamp of the simulation
- <id_> = (0, 2)

### 5. Launch the "XML to obj" in the Preprocessing.ipynb notebook to convert the .vtu files into .obj
   The script reads the vtu files and generate source and target .obj files, which are saved in: 
   PCSpineRegistration\source_target_pc\Spine<k>\spine1_vert<vert_id>1_<t_>_<id_>.obj with
- <vert_id> = (1, 2, 3, 4, 5) = the vertebral level
- <t_> = range(0, 35) = the timestamp of the simulation
- <id_> = (0, 2)


### 7. Launch 



