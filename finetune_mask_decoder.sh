#!/bin/bash
#SBATCH --ntasks=1                 # Number of tasks (see below)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=0-40:00             # Runtime in D-HH:MM
#SBATCH --partition gpu-2080ti
#SBATCH --gres=gpu:1               # optionally type and number of gpus
#SBATCH --mem=100G                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/akata/aoq833/sam_logs/hostname_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/akata/aoq833/sam_logs/hostname_%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=stefan.wezel@maddox.ai   # Email to which notifications will be sent

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akata/aoq833/ad/lib
source /home/akata/aoq833/ad/bin/activate

# # DEBUG
# datapath=/mnt/qb/work/akata/aoq833/vision_data/ datasets=('Cable' 'Capacitor' 'Casting' 'Console' 'Cylinder' 'Electronics' 'Groove' 'Hemisphere' 'Lens' 'PCB_1' 'PCB_2' 'Ring' 'Screw' 'Wood')


# dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

# # make patchcore module importable
# export PYTHONPATH=../src



python finetune_mask_decoder.py --lr 0.0001 --checkpoint_loadname sam_full.pth --checkpoint_savename sam_full2.pth






echo "Done"