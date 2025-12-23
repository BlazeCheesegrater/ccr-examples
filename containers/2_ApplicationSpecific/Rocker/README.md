# Example Rocker Container

## Building the container

Please refer to CCR's [container documentation](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/) for more detailed information on building and using Apptainer.

A brief guide to building an Rocker container for CCR's HPC follows:

1. Start an interactive job

> [!NOTE]
> Apptainer is not available on the CCR login nodes and the compile nodes may not provide enough resources for you to build a container. We recommend requesting an interactive job on a compute node to conduct this build process. 
> See CCR docs for more info on [submitting an interactive job](https://docs.ccr.buffalo.edu/en/latest/hpc/jobs/#interactive-job-submission).

Request a job allocation from a login node:
```
salloc --cluster=ub-hpc --partition=debug --qos=debug --exclusive --mem=8GB --time=00:30:00
```
Sample output:
```
salloc: Pending job allocation [JobID]
salloc: job [JobID] queued and waiting for resources
salloc: job [JobID] has been allocated resources
salloc: Granted job allocation [JobID]
salloc: Waiting for resource configuration
salloc: Nodes [NodeID] are ready for job
```

2. Navigate to your build directory and use the Slurm job local temporary directory for cache.
```
cd /projects/academic/[YourGroupName]/[CCRusername]/Rocker
export APPTAINER_CACHEDIR="${SLURMTMPDIR}"
```

Download the Rocker build file, `Rocker.def` to this directory.
```
curl -L -o Rocker.def https://raw.githubusercontent.com/Blaze-Cheesegrater/ccr-examples/containers/2_ApplicationSpecific/Rocker/Rocker.def
```

3. Build the container
```
apptainer build rocker-$(arch).sif rocker.def
```
Sample output:
```
INFO:    User not listed in /etc/subuid, trying root-mapped namespace
INFO:    The %post section will be run under the fakeroot command
INFO:    Starting build...
INFO:    Fetching OCI image...
..............
INFO:    Creating SIF file...
[===============================================================] 100 % 0s
INFO:    Build complete: rocker.sif
```

## Run the container

1. Start an interactive job

> [!NOTE]
> It may be necessary to change the requested resources based on the R program you want to run. For this example, we will be using minimal resources to check if our container runs. See CCR docs for more info on [submitting an interactive job](https://docs.ccr.buffalo.edu/en/latest/hpc/jobs/#interactive-job-submission).

Request a job allocation from a login node:
```
salloc --cluster=ub-hpc --partition=debug --qos=debug --exclusive --mem=8GB --time=00:30:00
```

Sample output:
```
salloc: Pending job allocation [JobID]
salloc: job [JobID] queued and waiting for resources
salloc: job [JobID] has been allocated resources
salloc: Granted job allocation [JobID]
salloc: Waiting for resource configuration
salloc: Nodes [NodeID] are ready for job
```

2. Run the container:
```
apptainer run rocker-x86_64.sif
```
You should see:
```
Launching R...

R version 4.5.2 (2025-10-31) -- "[Not] Part in a Rumble"
Copyright (C) 2025 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

>
```

## Editing the definition file
Test adding BiocGenerics and rmarkdown - also neofetch - use library(package) to test and maybe search()


See CCR docs for more info on [Building Images with Apptainer](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/#building-images-with-apptainer)