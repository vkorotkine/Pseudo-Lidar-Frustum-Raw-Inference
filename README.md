## Pseudo-Lidar and Frustum PointNets for 3D Object Detection from RGB-D Data
This repository, fork of frustum-pointnets combines the following repos, 
https://github.com/charlesq34/frustum-pointnets , 
https://github.com/mileyan/pseudo_lidar , 
and https://github.com/JiaRenChang/PSMNet .

The original [pseudo-lidar paper code](https://github.com/mileyan/pseudo_lidar) 
had the following problems when I tried to run it on my machine, which this repo aims to resolve: 
1. PSMnet did not give good disparity results on my installation
3. No inference on the raw dataset. 
3. Having to run many bash scripts from different repos with different options was a bit annoying. 

This repo does just inference and nice visualization with the provided pretrained models. No training, no evaluation metrics. 

## Data and Models
For disparity prediction, download the [pretrained model](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view) (as per the PSMNet repo) and put it in the log folder in the base directory. 
For the Frustum model, download the pretrained [Frustum_V1 model](https://drive.google.com/file/d/1qhCxw6uHqQ4SAkxIuBi-QCKqLmTGiNhP/view), 
pretrained by the pseudo-lidar folks.  Also put it in the log folder in the base directory. 

Download the KITTI dataset, from the official site. The KITTI dataset is assumed
to lie in the base directory of the repo. You can symlink it to your KITTI location. 
Assumed folder structure: 

```
frustum-pointnets
├── dataset
│   ├── KITTI
│   │   │── object
│   │   |   │── training
│   │   │   │   ├──calib & velodyne & label_2 & image_2 & image_3 
│   │   │   │── testing
│   │   │   │   ├──calib & velodyne & image_2 & image_3 
│   │   │   │── 2d_detections
│   │   │── raw
│   │   |   │── 2011_09_26
│   │   │   │   ├──2011_09_26_drive_0001_sync & calib_cam_to_cam.txt & calib_imu_to_velo.txt & calib_velo_to_cam.txt 
│   │   │   │   │   │ ── image_02 & image_03 
│   │   │   │   │   │       │ ── data    │ ── data
│   │   │   │── 2d_detections

```
Notice that you need the 2d_detections which you can obtain using your detector of choice. I used detectron2
but any one will do. Just make sure they are in the right folder and format.

## Installation
Tested on Python 3.6.8, Ubuntu 20.04, CUDA 11.3. Recommend using pyenv to manage python version. 
```
pip install -r requirements.txt
```
to install dependencies, and 
```
cd fr_kitti_inference_lib
pip install -e . 
```
to install the "inference" lib in editable mode. You also need [pykitti](https://github.com/utiasSTARS/pykitti),
which you get by initializing submodules and installing it, again in editable mode: 
 ```
git submodule update --init
cd pykitti
pip install -e . 
```
You might also get errors related to top-level folders missing. Just create them if that happens and it should run fine. 

## Usage
To run a whole sequence, run the scripts in either the raw or object folder in order. 
To run a single image, run the full pipeline script. 
To visualize/compare pointclouds, run the Jupyter notebooks. 

## Citation
If you use this repo, please cite the papers corresponding to the different components: 

        @article{qi2017frustum,
          title={Frustum PointNets for 3D Object Detection from RGB-D Data},
          author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1711.08488},
          year={2017}
        }

        @inproceedings{chang2018pyramid,
        title={Pyramid Stereo Matching Network},
        author={Chang, Jia-Ren and Chen, Yong-Sheng},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        pages={5410--5418},
        year={2018}
        }

        @article{qi2017frustum,
        title={Frustum PointNets for 3D Object Detection from RGB-D Data},
        author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
        journal={arXiv preprint arXiv:1711.08488},
        year={2017}
        }

        @InProceedings{Qi_2017_CVPR,
        author = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
        title = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {July},
        year = {2017}
        }



## License
Code is released under the Apache 2.0 license (see LICENSE file for details).
Same license as Frustum-Pointnets. 

## References
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data: <a href="https://github.com/charlesq34/pointnet">here</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data: <a href="https://github.com/charlesq34/pointnet2">here</a>.

### Todo

- Add a demo script to run inference of Frustum PointNets based on raw input data.
- Add related scripts for SUNRGBD dataset
