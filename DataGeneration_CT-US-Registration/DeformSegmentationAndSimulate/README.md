# Segmentation of verse DB

### get the extra tissues using total segmentator

- open the result using torchio so that it is in the same coordinate system as the image itself


## cropping
- find the index where we are out of the body just above the bed
  - definition of inside of body we can get from the segmentation map
  - this is the lateral bound
- frontal bound can be just 10 cm in front of the sacrum or for now the upper limit of the image
- the lowest part of the sacrum plus a margin would be the lower bound
- and in the other direction it would be after L1 plus some margin

- To apply this we can use torchio library, with the crop function, 
  just the center and the bounding box would be needed


## segment the soft tissue

## create the batch file to run ImFusionConsole


## to run the pipeline:
```shell
python 00_segment_and_simulate_pipeline.py --root_path_spines /mnt/HDD1/farid/spine_verse/data_generation/spines/01_training/ --root_path_vertebrae /mnt/HDD1/farid/spine_verse/data_generation/vertebrae/01_training/ --list_file_names all_spines.txt --nr_deform_per_spine 10 --verse_path /mnt/projects/deepSpine/VERSE20/training_data/  --segmentation_folder /mnt/HDD1/farid/spine_verse/data_generation/spines/segmentations/
```
## Failures

507 and 525 failed