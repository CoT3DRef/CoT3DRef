#!/bin/bash
#!/bin/bash
#SBATCH --job-name=run_single_meta
#SBATCH -N 1
#SBATCH -o run_single_meta.out
#SBATCH -e run_single_meta.err
#SBATCH --mail-user=mohamed.mohamed.2@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=40:00:00
#SBATCH --mem=32G
source /home/mohama0e/miniconda3/bin/activate
conda activate 3DCoMPat
cd /home/mohama0e/3D_Codes/CoT3D_VG/automatic_loc_module/extract_objects
python3 extract_obj_info.py \
--api-key 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 \
--num-samples 200 \
--offset 400 \
--csv-path ScanEnts3D_Nr3D.csv \
--output-file outputs/output_map_0.json \
--in-context \
--in-context-prompt prompts/meta_prompt_no_distractor 
