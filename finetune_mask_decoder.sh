#!/bin/bash
#SBATCH --ntasks=1                 # Number of tasks (see below)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=0-00:10             # Runtime in D-HH:MM
#SBATCH --partition gpu-2080ti
#SBATCH --gres=gpu:1               # optionally type and number of gpus
#SBATCH --mem=50G                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/akata/aoq833/logs/hostname_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/akata/aoq833/logs/hostname_%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=stefan.wezel@maddox.ai   # Email to which notifications will be sent

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akata/aoq833/ad/lib
source /home/akata/aoq833/ad/bin/activate

# DEBUG
datapath=/mnt/qb/work/akata/aoq833/vision_data/ datasets=('Cable' 'Capacitor' 'Casting' 'Console' 'Cylinder' 'Electronics' 'Groove' 'Hemisphere' 'Lens' 'PCB_1' 'PCB_2' 'Ring' 'Screw' 'Wood')


dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

# make patchcore module importable
export PYTHONPATH=../src



python ../bin/run_patchcore.py --gpu 0 --seed 0 --save_vision_submission --save_segmentation_images \
--log_group vision --log_project sandbox /mnt/qb/work/akata/aoq833/results  \
patch_core -b efficientnet_b0 -le features.5 -le features.6 --faiss_on_gpu \
--pretrain_embed_dimension -1 --target_embed_dimension -1 --anomaly_scorer_num_nn 1 --patchsize 1 \
memorybank_sampler --nominal_memorybank_downsampling_factor 0.3 --nok_memorybank_downsampling_factor 0.75 \
--use_ok_patches_in_nok_imgs --num_flow_nodes 1 --flow_training_epochs 3  --training_method flow \
--feature_sampler PCA --feature_downsampling_factor 0.1 approx_greedy_coreset \
dataset --train_val_split -1 --image_downsampling_factor 0.3 "${dataset_flags[@]}" augmented_coco_ad_dataset $datapath








echo "Done"