#!/bin/bash
cd ${WORKING_PATH}
export LD_LIBRARY_PATH=${WORKING_PATH}/ffmpeg/lib/:$LD_LIBRARY_PATH
cd data
cd ILSVRC2015
rm Annotations
rm Data
ln -s /running_root/users/han.shen/data/ILSVRC2015/Annotations .
mkdir Data
cd Data
ln -s /running_root/users/han.shen/data/ILSVRC2015/Data/DET .
mkdir VID
cd VID
ln -s /running_root/users/han.shen/data/ILSVRC2015/Data/VID/* .
ln -s /running_root/users/zhaojin.huang/common/dataset/vid/ILSVRC2015/Data/VID/mpeg4_snippets .
cd ../..
cd ImageSets
rm DET
rm VID
ln -s /running_root/users/han.shen/data/ILSVRC2015/ImageSets/VID .
ln -s /running_root/users/han.shen/data/ILSVRC2015/ImageSets/DET .
cd ../../..

cd model
rm pretrained_model
hadoop fs -get  hdfs://hobot-bigdata/user/zhaojin.huang/models/vid/pretrained_model
cd ..
cd output
rm -r *
cd ..

cp -r dff_rfcn /job_data/
cp -r experiments /job_data/
cp -r lib /job_data/
cp train_cluster.sh /job_data/
CMD="python experiments/dff_rfcn/dff_rfcn_end2end_train_test.py --cfg experiments/dff_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml"
echo Running ${CMD}
${CMD}
CMD="python experiments/dff_rfcn/dff_rfcn_test.py --cfg experiments/dff_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml"
echo Running ${CMD}
${CMD}
mkdir $PBS_JOBNAME
model_prefix=$PBS_JOBNAME
mv output $PBS_JOBNAME
hadoop fs -put $model_prefix hdfs://hobot-bigdata/user/zhaojin.huang/new_cluster_output_models/vid/
