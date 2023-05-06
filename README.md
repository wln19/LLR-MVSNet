# LLR-MVSNet
A ligweight network for low-texture reconstruction

## How to Use

### Environment

* python
* pytorch

## Training
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $MVS_TRANING  folder.

* in train.sh, set MVS_TRAINING as your training data path
* Train LLR-MVSNet: scripts/train.sh
##Testing
* Download our pre-processed dataset: [DTU's testing set](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) ,[Tanks & Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing) and
  [ETH3D benchmark](https://polybox.ethz.ch/index.php/s/pmTGWobErOnhEg0). Each dataset is already organized as follows:
```
root_directory
├──scan1 (scene_name1)
├──scan2 (scene_name2) 
      ├── images                 
      │   ├── 00000000.jpg       
      │   ├── 00000001.jpg       
      │   └── ...                
      ├── cams                   
      │   ├── 00000000_cam.txt   
      │   ├── 00000001_cam.txt   
      │   └── ...                
      └── pair.txt  
```
* In scripts/test.sh, set DTU_TESTPATH as $DTU_TESTPATH.
* The DTU_CKPT_FILE is automatically set as your pretrained checkpoint file, you also can download my [pretrained model]().
* Test on GPU by running scripts/test.sh. The code includes depth map estimation and depth fusion. The outputs are the point clouds in ply format.
* For quantitative evaluation on DTU dataset, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and
  [Points](http://roboimagedata.compute.dtu.dk/?page_id=36). Unzip them and place `Points` folder in `SampleSet/MVS Data/`.
  The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```
In `evaluations/dtu/BaseEvalMain_web.m`, set `dataPath` as path to `SampleSet/MVS Data/`, `plyPath` as directory that
stores the reconstructed point clouds and `resultsPath` as directory to store the evaluation results. Then run
`evaluations/dtu/BaseEvalMain_web.m` in matlab.
*For quantitative evaluation on [Tanks & Temples](https://www.tanksandtemples.org/) and [ETH3D benchmark](https://www.eth3d.net/), please submit to the website.
