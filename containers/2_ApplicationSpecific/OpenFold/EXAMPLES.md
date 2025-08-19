# OpenFold Examples

## OpenFold example from the GitHub sources

The following example is from the [OpenFold Inference docs](https://openfold.readthedocs.io/en/latest/Inference.html#running-alphafold-model-inference)

Start an interactive job with a GPU e.g.
NOTE: OpenFold Inference only uses one GPU

```
salloc --cluster=ub-hpc --partition=general-compute --qos=general-compute \
 --account="[SlurmAccountName]" --mem=128GB --nodes=1 --cpus-per-task=1 \
 --tasks-per-node=12 --gpus-per-node=1 --time=02:00:00
```

Change to your OpenFold directory

```
cd /projects/academic/[YourGroupName]/OpenFold
```

Create a top level output directory

```
mkdir -p ./output
```

Start the container, with the "./output" directory a the top level output
 directory "/output" inside the contianer.

```
apptainer shell \
 -B /projects:/projects,/scratch:/scratch,/util:/util,/vscratch:/vscratch \
 -B /util/software/data/OpenFold:/data \
 -B /util/software/data/alphafold:/database \
 -B /util/software/data/OpenFold/openfold_params:/opt/openfold/openfold/resources/openfold_params \
 -B /util/software/data/alphafold/params:/opt/openfold/openfold/resources/params \
 -B $(pwd)/output:/output \
 --nv \
 OpenFold-$(arch).sif
```

expected output:

> ```
> Apptainer> 
> ```

All the following commands are run from the "Apptainer>" prompt.

The following example uses the [OpenFold model params](Download_model_parameters.md), which have
already been downloaded at CCR and are avaiable in the directory:
/util/software/data/OpenFold/openfold_params
This directory is mounted on /util/software/data/OpenFold/openfold_params
inside the contaner when using the "apptainer" command given above

# Get the example from the OpenFold GitHub repo

```
git clone https://github.com/aqlaboratory/openfold.git
mv openfold//examples/ ./examples/
rm -rf openfold
```

You should now have the monomer example in the ./examples/monomer/ directory

```
ls -l ./examples/monomer/
```

Sample output:

> ```
> total 1
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 21 15:45 alignments
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 21 15:45 fasta_dir
> -rwxrwxr-x 1 [CCRusername] nogroup  530 Aug 21 15:45 inference.sh
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 21 15:45 sample_predictions
> ```

## Run model inference

### Model inference with pre-computed alignments

Note: this example uses "/output/PDB_6KWC/pre-computed_alignments" as the
output directory; outside the container this is the directory:
"./output/PDB_6KWC/pre-computed_alignments"


```
export TRITON_CACHE_DIR="${SLURMTMPDIR}"
mkdir -p /output/PDB_6KWC/pre-computed_alignments
python3 "${OF_DIR}/run_pretrained_openfold.py" \
 --hhblits_binary_path "/opt/conda/bin/hhblits" \
 --hmmsearch_binary_path "/opt/conda/bin/hhsearch" \
 --hmmbuild_binary_path "/opt/conda/bin/hmmbuild" \
 --kalign_binary_path "/opt/conda/bin/kalign" \
 --model_device cuda \
 --data_random_seed $(((RANDOM<<15)|(RANDOM + 1))) \
 --use_precomputed_alignments "./examples/monomer/alignments" \
 --output_dir "/output/PDB_6KWC/pre-computed_alignments" \
 --config_preset model_1_ptm \
 --jax_param_path "${OF_DIR}/openfold/resources/params/params_model_1_ptm.npz" \
 "./examples/monomer/fasta_dir" \
 "/data/pdb_data/mmcif_files"
```

Sample output:

> ```
> [2025-08-25 14:49:35,606] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> INFO:/opt/openfold/openfold/utils/script_utils.py:Successfully loaded JAX parameters at /opt/openfold/openfold/resources/params/params_model_1_ptm.npz...
> INFO:/opt/openfold/run_pretrained_openfold.py:Using precomputed alignments for 6KWC_1 at ./examples/monomer/alignments...
> INFO:/opt/openfold/openfold/utils/script_utils.py:Running inference for 6KWC_1...
> INFO:/opt/openfold/openfold/utils/script_utils.py:Inference time: 56.51587692601606
> INFO:/opt/openfold/run_pretrained_openfold.py:Output written to /output/PDB_6KWC/pre-computed_alignments/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb...
> INFO:/opt/openfold/run_pretrained_openfold.py:Running relaxation on /output/PDB_6KWC/pre-computed_alignments/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb...
> INFO:/opt/openfold/openfold/utils/script_utils.py:Relaxation time: 10.501414217054844
> INFO:/opt/openfold/openfold/utils/script_utils.py:Relaxed output written to /output/PDB_6KWC/pre-computed_alignments/predictions/6KWC_1_model_1_ptm_relaxed.pdb...
> ```

The output for the run is in the PDB_6KWC/pre-computed_alignments directory tree

```
ls -laR /output/PDB_6KWC/pre-computed_alignments
```

Sample output:

> ```
> /output/PDB_6KWC/pre-computed_alignments:
> total 1
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 25 14:50 .
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 25 14:49 ..
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 25 14:51 predictions
> -rw-rw-r-- 1 [CCRusername] nogroup   44 Aug 25 14:50 timings.json
> 
> /output/PDB_6KWC/pre-computed_alignments/predictions:
> total 344
> drwxrwsr-x 2 [CCRusername] nogroup   4096 Aug 25 14:51 .
> drwxrwsr-x 2 [CCRusername] nogroup   4096 Aug 25 14:50 ..
> -rw-rw-r-- 1 [CCRusername] nogroup 230149 Aug 25 14:51 6KWC_1_model_1_ptm_relaxed.pdb
> -rw-rw-r-- 1 [CCRusername] nogroup 120528 Aug 25 14:50 6KWC_1_model_1_ptm_unrelaxed.pdb
> -rw-rw-r-- 1 [CCRusername] nogroup     34 Aug 25 14:51 timings.json
> ```


### Model inference without pre-computed alignments

Note: jackhmmer and nhmmer don't scale beyond 8 cores, henec the "--cpu" option
is set to 8 rather than $(nproc)

```
export TRITON_CACHE_DIR="${SLURMTMPDIR}"
mkdir -p /output/PDB_6KWC/without_pre-computed_alignments
python3 "${OF_DIR}/run_pretrained_openfold.py" \
 --hhblits_binary_path "/opt/conda/bin/hhblits" \
 --hmmsearch_binary_path "/opt/conda/bin/hhsearch" \
 --hmmbuild_binary_path "/opt/conda/bin/hmmbuild" \
 --kalign_binary_path "/opt/conda/bin/kalign" \
 --uniref90_database_path "/database/uniref90/uniref90.fasta" \
 --mgnify_database_path "/database/mgnify/mgy_clusters_2022_05.fa" \
 --pdb70_database_path "/database/pdb70/pdb70" \
 --uniclust30_database_path "/database/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
 --bfd_database_path "/database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
 --cpus 8 \
 --model_device cuda \
 --data_random_seed $(((RANDOM<<15)|(RANDOM + 1))) \
 --output_dir "/output/PDB_6KWC/without_pre-computed_alignments" \
 --config_preset model_1_ptm \
 --jax_param_path "${OF_DIR}/openfold/resources/params/params_model_1_ptm.npz" \
 "./examples/monomer/fasta_dir" \
 "/data/pdb_data/mmcif_files"
```

Sample output:

> ```
> [2025-08-26 09:33:00,593] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> INFO:/opt/openfold/openfold/utils/script_utils.py:Successfully loaded JAX parameters at /opt/openfold/openfold/resources/params/params_model_1_ptm.npz...
> INFO:/opt/openfold/run_pretrained_openfold.py:Generating alignments for 6KWC_1...
> INFO:/opt/openfold/openfold/utils/script_utils.py:Running inference for 6KWC_1...
> INFO:/opt/openfold/openfold/utils/script_utils.py:Inference time: 58.297142078052275
> INFO:/opt/openfold/run_pretrained_openfold.py:Output written to /output/PDB_6KWC/without_pre-computed_alignments/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb...
> INFO:/opt/openfold/run_pretrained_openfold.py:Running relaxation on /output/PDB_6KWC/without_pre-computed_alignments/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb...
> INFO:/opt/openfold/openfold/utils/script_utils.py:Relaxation time: 10.48616563109681
> INFO:/opt/openfold/openfold/utils/script_utils.py:Relaxed output written to /output/PDB_6KWC/without_pre-computed_alignments/predictions/6KWC_1_model_1_ptm_relaxed.pdb...
> ```

The output for the run is in the PDB_6KWC/without_pre-computed_alignments directory tree

```
ls -laR /output/PDB_6KWC/without_pre-computed_alignments
```

Sample output:

> ```
> /output/PDB_6KWC/without_pre-computed_alignments:
> total 1
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:52 .
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:35 ..
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:33 alignments
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:53 predictions
> -rw-rw-r-- 1 [CCRusername] nogroup   45 Aug 26 09:52 timings.json
> 
> /output/PDB_6KWC/without_pre-computed_alignments/alignments:
> total 0
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:33 .
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:52 ..
> drwxrwsr-x 2 [CCRusername] nogroup 4096 Aug 26 09:51 6KWC_1
> 
> /output/PDB_6KWC/without_pre-computed_alignments/alignments/6KWC_1:
> total 7028
> drwxrwsr-x 2 [CCRusername] nogroup    4096 Aug 26 09:51 .
> drwxrwsr-x 2 [CCRusername] nogroup    4096 Aug 26 09:33 ..
> -rw-rw-r-- 1 [CCRusername] nogroup  397302 Aug 26 09:51 bfd_uniclust_hits.a3m
> -rw-rw-r-- 1 [CCRusername] nogroup  136021 Aug 26 09:38 hhsearch_output.hhr
> -rw-rw-r-- 1 [CCRusername] nogroup 1972569 Aug 26 09:48 mgnify_hits.sto
> -rw-rw-r-- 1 [CCRusername] nogroup 4689644 Aug 26 09:38 uniref90_hits.sto
> 
> /output/PDB_6KWC/without_pre-computed_alignments/predictions:
> total 344
> drwxrwsr-x 2 [CCRusername] nogroup   4096 Aug 26 09:53 .
> drwxrwsr-x 2 [CCRusername] nogroup   4096 Aug 26 09:52 ..
> -rw-rw-r-- 1 [CCRusername] nogroup 230149 Aug 26 09:53 6KWC_1_model_1_ptm_relaxed.pdb
> -rw-rw-r-- 1 [CCRusername] nogroup 120528 Aug 26 09:52 6KWC_1_model_1_ptm_unrelaxed.pdb
> -rw-rw-r-- 1 [CCRusername] nogroup     33 Aug 26 09:53 timings.json
> ```


Note: Other possible options for "run_pretrained_openfold.py"

> ```
> --pdb_seqres_database_path "/database/pdb_seqres/pdb_seqres.txt" \
> --uniref30_database_path "/database/uniref30/UniRef30_2021_03" \
> --uniprot_database_path "/database/uniprot/uniprot.fasta" \
> --max_template_date MAX_TEMPLATE_DATE \
> --obsolete_pdbs_path OBSOLETE_PDBS_PATH \
> --model_device MODEL_DEVICE \
> --config_preset CONFIG_PRESET \
> --openfold_checkpoint_path OPENFOLD_CHECKPOINT_PATH \
> --save_outputs \
> --preset {reduced_dbs,full_dbs} \
> --output_postfix OUTPUT_POSTFIX \
> --skip_relaxation \
> --multimer_ri_gap MULTIMER_RI_GAP \
> --trace_model \
> --subtract_plddt \
> --long_sequence_inference \
> --cif_output \
> --experiment_config_json EXPERIMENT_CONFIG_JSON \
> --use_deepspeed_evoformer_attention \
> --release_dates_path RELEASE_DATES_PATH \
> ```


# Multi GPU example using the OpenFold PDB training set from RODA

Start an interactive job with more than one GPU e.g.

```
salloc --cluster=ub-hpc --account="[SlurmAccountName]" \
 --partition=industry-dgx --qos=industry-dgx --mem=128GB --nodes=1 \
 --gpus-per-node=8 --mem=0 --exclusive --time=3-00:00:00
```

sample outout:

> ```
> salloc: Pending job allocation 21070582
> salloc: job 21070582 queued and waiting for resources
> salloc: job 21070582 has been allocated resources
> salloc: Granted job allocation 21070582
> salloc: Nodes cpn-i09-04 are ready for job
> CCRusername@cpn-i09-04:~$ 
> ```

In this case the node allocated has eight H100 GPUs with 80GB RAM each

```
nvidia-smi -L
```

output:

> ````
> GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-e5f404f3-cc2a-cf0c-219f-dcf1a4e223f2)
> GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-96601a91-e977-7a71-a188-8df4aff2fbcc)
> GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-c4a62918-26ce-f10c-a009-dd3b2e069ac2)
> GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-7b286e42-7f9d-a8e8-501c-14b0663b8440)
> GPU 4: NVIDIA H100 80GB HBM3 (UUID: GPU-a9038bc9-da63-7f95-edb6-9857e428acbc)
> GPU 5: NVIDIA H100 80GB HBM3 (UUID: GPU-347ee5de-5ad5-fdea-3c1b-41ba332b066e)
> GPU 6: NVIDIA H100 80GB HBM3 (UUID: GPU-558a69d7-ed47-fd4c-be72-4308fefe6876)
> GPU 7: NVIDIA H100 80GB HBM3 (UUID: GPU-f65a6ec2-ce6f-ba6c-d4c2-8ca414d6e709)
> ````

Change to your OpenFold directory

```
cd /projects/academic/[YourGroupName]/OpenFold
```

Create an output directory

```
mkdir -p ./output
```

Start the container, with the "./output" directory as the output directory.
Note: You can change the /output mount: "-B $(pwd)/output:/output" to use an
alternate output directory

```
apptainer shell \
 -B /projects:/projects,/scratch:/scratch,/util:/util,/vscratch:/vscratch \
 -B /util/software/data/OpenFold:/data \
 -B /util/software/data/alphafold:/database \
 -B /util/software/data/OpenFold/openfold_params:/opt/openfold/openfold/resources/openfold_params \
 -B /util/software/data/alphafold/params:/opt/openfold/openfold/resources/params \
 -B $(pwd)/output:/output \
 --nv \
 OpenFold-$(arch).sif
```

expected output:

> ```
> Apptainer> 
> ```

All the following commands are run from the "Apptainer>" prompt.

The following example uses the OpenFold PDB training set from RODA, which was
downloaded and processed for use already, and is available at CCR in the
directory: /util/software/data/OpenFold/
This directory is about 1.5TB in size

The process to download & process this data is documented in in the
[Download_OpenFold_PDB_training_set.md](Download_OpenFold_PDB_training_set.md) file.
This process takes several days to complete.  Do NOT follow the instrctions
therein unless the CCR copy does not work for your use case, and you have
sufficient storage space for the files.


NOTE: The "--seed" option below uses a random number utilizing the ${RANDOM}
bash variable to generate an integer in the 1 to 2^32 range.  You should
expect to generate different loss values for the same parameters and data,
with multiple runs.  If you use the same seed for multiple runs you should
generate the same loss values (this can be used for reproducibility.)

```
export TRITON_CACHE_DIR="${SLURMTMPDIR}"
mkdir -p /output/PDB/2021-10-10/
python3 "${OF_DIR}/train_openfold.py" \
 --train_chain_data_cache_path "/data/pdb_data/data_caches/chain_data_cache.json" \
 --template_release_dates_cache_path "/data/pdb_data/data_caches/mmcif_cache.json" \
 --obsolete_pdbs_file_path "/data/pdb_data/obsolete.dat" \
 --config_preset initial_training \
 --seed $(((RANDOM<<15)|(RANDOM + 1))) \
 --num_nodes ${SLURM_NNODES} \
 --gpus $(expr ${SLURM_GPUS_ON_NODE} \* ${SLURM_NNODES}) \
 --max_epochs 1000 \
 --checkpoint_every_epoch \
 "/data/pdb_data/mmcif_files" \
 "/data/alignment_data/alignments" \
 "/data/pdb_data/mmcif_files" \
 "/output/PDB/2021-10-10" \
 "2021-10-10"
```

Sample output:

> ```
> [2025-08-26 11:53:29,143] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> [rank: 0] Seed set to 504588624
> /opt/conda/lib/python3.10/site-packages/lightning_fabric/connector.py:571: `precision=bf16` is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
> Using bfloat16 Automatic Mixed Precision (AMP)
> GPU available: True (cuda), used: True
> TPU available: False, using: 0 TPU cores
> HPU available: False, using: 0 HPUs
> You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
> Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8
> [2025-08-26 11:54:03,699] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 11:54:03,705] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 11:54:03,705] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 11:54:03,706] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 11:54:03,710] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 11:54:03,710] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 11:54:03,715] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> [rank: 1] Seed set to 504588624
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> [rank: 2] Seed set to 504588624
> [rank: 6] Seed set to 504588624
> [rank: 4] Seed set to 504588624
> [rank: 7] Seed set to 504588624
> [rank: 5] Seed set to 504588624
> [rank: 3] Seed set to 504588624
> Initializing distributed: GLOBAL_RANK: 4, MEMBER: 5/8
> Initializing distributed: GLOBAL_RANK: 6, MEMBER: 7/8
> Initializing distributed: GLOBAL_RANK: 7, MEMBER: 8/8
> Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/8
> Initializing distributed: GLOBAL_RANK: 5, MEMBER: 6/8
> Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/8
> Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/8
> ----------------------------------------------------------------------------------------------------
> distributed_backend=nccl
> All distributed processes registered. Starting with 8 processes
> ----------------------------------------------------------------------------------------------------
> 
> LOCAL_RANK: 7 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 6 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 4 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 5 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/utilities/model_summary/model_summary.py:231: Precision bf16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
> 
>   | Name  | Type          | Params | Mode 
> ------------------------------------------------
> 0 | model | AlphaFold     | 93.2 M | train
> 1 | loss  | AlphaFoldLoss | 0      | train
> ------------------------------------------------
> 93.2 M    Trainable params
> 0         Non-trainable params
> 93.2 M    Total params
> 372.916   Total estimated model params size (MB)
> 4451      Modules in train mode
> 0         Modules in eval mode
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py:106: Total length of `list` across ranks is zero. Please make sure this was your intention.
> Epoch 0:   0%|                                                                                                                            | 0/1250 [00:00<?, ?it/s]/opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/distogram', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/distogram_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py:79: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/experimentally_resolved', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/experimentally_resolved_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/fape', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/fape_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/plddt_loss', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/plddt_loss_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/masked_msa', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/masked_msa_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/supervised_chi', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/supervised_chi_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/violation', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/violation_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/unscaled_loss', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/unscaled_loss_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/loss', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/loss_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/lddt_ca', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/drmsd_ca', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> WARNING:root:The exact sequence TTADIVVFDEISMATNYDLSVVNARLRAKHYVYIGDPAQLPAPRTLLTKGTLEPEYFNSVCRLMKTIGPDMFLGTCRRCPAEIVDTVSALVYDNKLKAHKDKSAQCFKMFYKGVITHDVSSAINRPQIGVVREFLTRNPAWRKAVFISPYNSQNAVASKILGLPTQTVDSSQGSEYDYVIFTQTTETAHSCNVNRFNVAITRAKVGILCIMSDR was not found in 7o7y_BK. Realigning the template to the actual sequence.
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> Epoch 0:   1%|█▎                                                                                             | 17/1250 [01:45<2:07:09,  0.16it/s, train/loss=90.00]
> [...]
> Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [1:31:49<00:00,  0.23it/s, train/loss=55.40]
> ```

Note: This example will fail with an odd "strategy=None" error if run on a
node with only one GPU

In the above example, I stopped the training after Ephoch 0 which created the
following checkpoint file:

```
ls -l /output/PDB/2021-10-10/checkpoints
```

> ```
> total 1464717
> -rw-rw-r-- 1 tkewtest nogroup 1499869690 Aug 26 13:26 /output/PDB/2021-10-10/checkpoints/0-1250.ckpt
> ```

Restarted the training from the checkpoint:

```
export TRITON_CACHE_DIR="${SLURMTMPDIR}"
python3 "${OF_DIR}/train_openfold.py" \
 --train_chain_data_cache_path "/data/pdb_data/data_caches/chain_data_cache.json" \
 --template_release_dates_cache_path "/data/pdb_data/data_caches/mmcif_cache.json" \
 --obsolete_pdbs_file_path "/data/pdb_data/obsolete.dat" \
 --config_preset initial_training \
 --seed $(((RANDOM<<15)|(RANDOM + 1))) \
 --num_nodes ${SLURM_NNODES} \
 --gpus $(expr ${SLURM_GPUS_ON_NODE} \* ${SLURM_NNODES}) \
 --max_epochs 1000 \
 --checkpoint_every_epoch \
 --resume_from_ckpt /output/PDB/2021-10-10/checkpoints/0-1250.ckpt \
 "/data/pdb_data/mmcif_files" \
 "/data/alignment_data/alignments" \
 "/data/pdb_data/mmcif_files" \
 "/output/PDB/2021-10-10" \
 "2021-10-10"
```

Sample output:

```
> [2025-08-26 15:33:33,529] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> [rank: 0] Seed set to 741092476
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> /opt/conda/lib/python3.10/site-packages/lightning_fabric/connector.py:571: `precision=bf16` is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
> Using bfloat16 Automatic Mixed Precision (AMP)
> GPU available: True (cuda), used: True
> TPU available: False, using: 0 TPU cores
> HPU available: False, using: 0 HPUs
> You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
> Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8
> [2025-08-26 15:33:57,402] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 15:33:57,418] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 15:33:57,426] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 15:33:57,431] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 15:33:57,443] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 15:33:57,445] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> [2025-08-26 15:33:57,448] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> Warning: The default cache directory for DeepSpeed Triton autotune, /user/tkewtest/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> [rank: 2] Seed set to 741092476
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:49: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
>   def forward(ctx, input, weight, bias=None):
> /opt/conda/lib/python3.10/site-packages/deepspeed/runtime/zero/linear.py:67: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
>   def backward(ctx, grad_output):
> [rank: 3] Seed set to 741092476
> [rank: 1] Seed set to 741092476
> [rank: 5] Seed set to 741092476
> [rank: 4] Seed set to 741092476
> [rank: 6] Seed set to 741092476
> [rank: 7] Seed set to 741092476
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/8
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> /opt/openfold/train_openfold.py:328: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>   sd = torch.load(args.resume_from_ckpt)
> Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/8
> Initializing distributed: GLOBAL_RANK: 7, MEMBER: 8/8
> Initializing distributed: GLOBAL_RANK: 4, MEMBER: 5/8
> Initializing distributed: GLOBAL_RANK: 6, MEMBER: 7/8
> Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/8
> Initializing distributed: GLOBAL_RANK: 5, MEMBER: 6/8
> ----------------------------------------------------------------------------------------------------
> distributed_backend=nccl
> All distributed processes registered. Starting with 8 processes
> ----------------------------------------------------------------------------------------------------
> 
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:701: Checkpoint directory /output/PDB/2021-10-10/checkpoints exists and is not empty.
> Restoring states from the checkpoint path at /output/PDB/2021-10-10/checkpoints/0-1250.ckpt
> LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 6 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 7 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 5 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> LOCAL_RANK: 4 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/utilities/model_summary/model_summary.py:231: Precision bf16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> /opt/conda/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
>   warnings.warn(
> 
>   | Name  | Type          | Params | Mode 
> ------------------------------------------------
> 0 | model | AlphaFold     | 93.2 M | train
> 1 | loss  | AlphaFoldLoss | 0      | train
> ------------------------------------------------
> 93.2 M    Trainable params
> 0         Non-trainable params
> 93.2 M    Total params
> 372.916   Total estimated model params size (MB)
> 4451      Modules in train mode
> 0         Modules in eval mode
> Restored all states from the checkpoint at /output/PDB/2021-10-10/checkpoints/0-1250.ckpt
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py:106: Total length of `list` across ranks is zero. Please make sure this was your intention.
> Training: |                                                                                                                                  | 0/? [00:00<?, ?it/s]WARNING:root:The exact sequence LDARLDTVYDAIVLGGGMGGLSAAIYLARYGLKCLVVEKGRGRSFWMQDLRNYVGLDPDTPGRDIITHSTQQALHWGADLLRGYVEDVTDEGDTLAVKVKVGKKDSLYPIFRTKYVIAATGIIDNLPQLEDMQNVYDYAGYTLHVCMICDGFDMWDQKAVLIAGTEGQINAAFVLNWFTPYITVLTHGLCTVGDEMKAKLADHGYPLHEAAITKFLGEDHKMSGVELVDGTVMEATTGLINMGSVYHNHYLKGIEGLEWDGENLVTNDMAQTSHPRIFALGDLKKGLNQVSVAVADGTLAATQIWRNIRRASEPRK was not found in 5k0a_D. Realigning the template to the actual sequence.
> WARNING:root:The exact sequence ATLREQSFDEAWLFHRGDIAEGEKQSLDDSQWRQINLPHDWSIEDIPGTNSPFTADAATEVAGGFTVGGTGWYRKHFYIDAAEKGKAIAVSFDGIYMNADIWVNDRHVANHVYGYTAFELDITDYVRFGAENLIAVRVKNEGMNCRWYTGSGIYRHTFLKITNPLHFETWGTFVTTPVATADKAEVHVQSVLANTEKVTGKVILETRIVDKNNHTVARKEQLVTLDNKEKTEVGHALEVLAPQLWSIDNPYLYQVVNRLLQDDKVIDEEYISIGIRNIAFSAENGFQLNGKSMKLKGGCIHHDNGLLGAKAFDRAEERKIELLKAAGFNALRLSHNPPSIALLNACDRLGMLVIDEAFDMWRYGHYQYDYAQYFDKLWKEDLHSMVARDRNHPSVIMWSIGNEIKNKETAEIVDICRELTGFVKTLDTTRPVTAGVNSIVDATDDFLAPLDVCGYNYALNRYESDAKRHPDRIIYASESYASQAYDYWKGVEDHSWVIGDFIWTAFDYIGEASIGWCGYPLDKRIFPWNHANCGDLNLSGERRPQSYLRETLWSDAPVSHIVVTPPVPSFPLNPDKADWSVWDFPDVVDHWNFPGYEGKKMTVSVYSNCEQVELFLNGESLGKQENTADKKNTLVWEVPYAHGILKAVSYNKGGEVGTATLESAGKVEKIRLSADRTEIVADGNDLSYITLELVDSKGIRNQLAEELVAFSIEGDATIEGVGNANPMSIESFVANSRKTWRGSNLLVVRSGKSSGRIIVTAKVKALPVASITIT was not found in 6b6l_B. Realigning the template to the actual sequence.
> WARNING:root:The exact sequence VRKRVLIGLKDAPNFVMRLFTVEPGGLIDRASHPWEHEIFVLKGKLTVLKEQGEETVEEGFYIFVEPNEIHGFRNDTDSEVEFLA was not found in 6l2e_B. Realigning the template to the actual sequence.
> WARNING:root:The exact sequence MVILEVANPQEAARVLNENLLVGYFLPCKLVVYQENGTTKIGMPK was not found in 1q9u_B. Realigning the template to the actual sequence.
> WARNING:root:The exact sequence GRLGVTRNKIMTAQYECYQKIMQDPIQQAEGVYCQRTWDGWLCWNDVAAGTESMQLCPDYFQDFDPSEKVTKICDQDGNWFRHPASQRTWTNYTQCNVNT was not found in 6zho_A. Realigning the template to the actual sequence.
> WARNING:root:The exact sequence SSVPMTQNRNILWIMCDQLRFDYLSCYGHERLNTPNIDKLAKRGVRFTNAYVQATVXGPSRMSAYTGRYVRSHGSTQNGIPLRVGEPTLGDHLRDVGMRNVLIGKTHMRPDLDGMKRLGIDPDSEIGARVGEGGFDAFDRDDGVHPTGYRKKEPAYNDYLRHAGFQAENPWEFWANSAEGKGGENQSGWLLTHADKPARVPEEHSETAYMTRRAMEFMEAAEKDGRPWCAHLSYIKPHWPYIVPAPYHDMFGPDDVKPAVRSDEELKAAHPLFKAMTEEVYSRNFARDEVREKVIPAYMGLIKQIDDQLGQLFAFMQERGLDENTMIVFTADHGDYLGDHWMGEKYLFYEAAAKVPLIIYDPSDKADATRGTVSDALVEMIDLAPTFVDYAGGVPPMHILEGKSLLPLLHDDDSSWDRQYVFSELDYSNLPARLKLGRDIQDCRATMVFDGRYKLVEVMGFAPILFDLEVDPDELKDLGRDPSAEEVRQRLTSALDAWHRNTRQR was not found in 4upi_A. Realigning the template to the actual sequence.
> WARNING:root:The exact sequence PRGSHMASIKKPNVLILLFDDMRFDTFSYRNGPVSTPNIDALANEGTRFDQAMTSTGLASPSRAAMFTGRWGHKTGLDDNVGLYHSRLSELSLSEGSVIKRATSIGYDVSYVGKWHLGAQGPALRGANFMWGHDKDEERNGRPFTPYQTQKNVARMNAGERDKNGEKHDYYKTLPGTYADTVTAKEVNEGKLMLQNAAKSDKPFFGIVSFEQPHPPYRVPEPYASMYDYKDIKLPKNFGIKRKHKPMAQDDIWWPWHDVSHMSETDWRKAHSFYYGAIAMIDHAVGELINTAKEEGLYDDLHIILVGDQGSMLGEHNLYDKGPYAYDELMRMPLIIRDPSLEPKIINRQVSMLDIAPTLRQWMTLPLDGDEDGRSLLPLMKQGDSADAGKDDISLYAYEWYNGGWFGIRAIRTPEMKFVWNPGDSRDELYDLKNDPYEITNQIDNPKYKKQLTDLVHKMAGELNRIDDPSLTKF was not found in 6pt4_B. Realigning the template to the actual sequence.
> Epoch 1:   0%|                                                                                                                            | 0/1250 [00:00<?, ?it/s]/opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/distogram', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/distogram_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py:79: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/experimentally_resolved', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/experimentally_resolved_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/fape', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/fape_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/plddt_loss', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/plddt_loss_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/masked_msa', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/masked_msa_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/supervised_chi', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/supervised_chi_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/violation', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/violation_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/unscaled_loss', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/unscaled_loss_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/loss', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/loss_epoch', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/lddt_ca', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/conda/lib/python3.10/site-packages/pytorch_lightning/core/module.py:520: You called `self.log('train/drmsd_ca', ..., logger=True)` but have no logger configured. You can enable one by doing `Trainer(logger=ALogger(...))`
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:202: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/openfold/openfold/model/primitives.py:226: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> /opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:632: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
>   return fn(*args, **kwargs)
> /opt/openfold/openfold/model/primitives.py:258: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
>   with torch.cuda.amp.autocast(enabled=False):
> Epoch 1:   2%|█▌                                                                                             | 21/1250 [01:56<1:53:48,  0.18it/s, train/loss=50.10]
> [...]
```

You can monitor the GPU utilization which running the training as following,
using the Slurm job id:

e.g. from vortex:

```
srun --jobid="21070582" --export=HOME,TERM,SHELL --pty /bin/bash --login
```

Sample output:

> ```
> CCRusername@cpn-i09-04:~$
> ```

Show the GPUs available in the Slurm job:

```
nvidia-smi -L
```

Sample output:

> ```
> GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-e5f404f3-cc2a-cf0c-219f-dcf1a4e223f2)
> GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-96601a91-e977-7a71-a188-8df4aff2fbcc)
> GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-c4a62918-26ce-f10c-a009-dd3b2e069ac2)
> GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-7b286e42-7f9d-a8e8-501c-14b0663b8440)
> GPU 4: NVIDIA H100 80GB HBM3 (UUID: GPU-a9038bc9-da63-7f95-edb6-9857e428acbc)
> GPU 5: NVIDIA H100 80GB HBM3 (UUID: GPU-347ee5de-5ad5-fdea-3c1b-41ba332b066e)
> GPU 6: NVIDIA H100 80GB HBM3 (UUID: GPU-558a69d7-ed47-fd4c-be72-4308fefe6876)
> GPU 7: NVIDIA H100 80GB HBM3 (UUID: GPU-f65a6ec2-ce6f-ba6c-d4c2-8ca414d6e709)
> ```

Monitor the GPU activity:

```
nvidia-smi -l
```

Sample output:

> ```
> Tue Aug 26 15:49:52 2025       
> +-----------------------------------------------------------------------------------------+
> | NVIDIA-SMI 570.133.20             Driver Version: 570.133.20     CUDA Version: 12.8     |
> |-----------------------------------------+------------------------+----------------------+
> | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
> | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
> |                                         |                        |               MIG M. |
> |=========================================+========================+======================|
> |   0  NVIDIA H100 80GB HBM3          On  |   00000000:19:00.0 Off |                    0 |
> | N/A   41C    P0            362W /  700W |   26980MiB /  81559MiB |    100%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   1  NVIDIA H100 80GB HBM3          On  |   00000000:3B:00.0 Off |                    0 |
> | N/A   37C    P0            346W /  700W |   11751MiB /  81559MiB |    100%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   2  NVIDIA H100 80GB HBM3          On  |   00000000:4C:00.0 Off |                    0 |
> | N/A   36C    P0            330W /  700W |   11751MiB /  81559MiB |    100%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   3  NVIDIA H100 80GB HBM3          On  |   00000000:5D:00.0 Off |                    0 |
> | N/A   38C    P0            349W /  700W |   11751MiB /  81559MiB |    100%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   4  NVIDIA H100 80GB HBM3          On  |   00000000:9B:00.0 Off |                    0 |
> | N/A   39C    P0            334W /  700W |   11751MiB /  81559MiB |     95%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   5  NVIDIA H100 80GB HBM3          On  |   00000000:BB:00.0 Off |                    0 |
> | N/A   36C    P0            335W /  700W |   11751MiB /  81559MiB |     73%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   6  NVIDIA H100 80GB HBM3          On  |   00000000:CB:00.0 Off |                    0 |
> | N/A   86C    P0            226W /  700W |   11751MiB /  81559MiB |    100%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
> |   7  NVIDIA H100 80GB HBM3          On  |   00000000:DB:00.0 Off |                    0 |
> | N/A   37C    P0            353W /  700W |   11511MiB /  81559MiB |     89%      Default |
> |                                         |                        |             Disabled |
> +-----------------------------------------+------------------------+----------------------+
>                                                                                          
> +-----------------------------------------------------------------------------------------+
> | Processes:                                                                              |
> |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
> |        ID   ID                                                               Usage      |
> |=========================================================================================|
> |    0   N/A  N/A         2085037      C   python3                               13266MiB |
> |    0   N/A  N/A         2085212      C   /opt/conda/bin/python3                 1952MiB |
> |    0   N/A  N/A         2085213      C   /opt/conda/bin/python3                 1952MiB |
> |    0   N/A  N/A         2085214      C   /opt/conda/bin/python3                 1952MiB |
> |    0   N/A  N/A         2085215      C   /opt/conda/bin/python3                 1952MiB |
> |    0   N/A  N/A         2085216      C   /opt/conda/bin/python3                 1952MiB |
> |    0   N/A  N/A         2085217      C   /opt/conda/bin/python3                 1952MiB |
> |    0   N/A  N/A         2085218      C   /opt/conda/bin/python3                 1952MiB |
> |    1   N/A  N/A         2085212      C   /opt/conda/bin/python3                11742MiB |
> |    2   N/A  N/A         2085213      C   /opt/conda/bin/python3                11742MiB |
> |    3   N/A  N/A         2085214      C   /opt/conda/bin/python3                11742MiB |
> |    4   N/A  N/A         2085215      C   /opt/conda/bin/python3                11742MiB |
> |    5   N/A  N/A         2085216      C   /opt/conda/bin/python3                11742MiB |
> |    6   N/A  N/A         2085217      C   /opt/conda/bin/python3                11742MiB |
> |    7   N/A  N/A         2085218      C   /opt/conda/bin/python3                11502MiB |
> +-----------------------------------------------------------------------------------------+
> ```

