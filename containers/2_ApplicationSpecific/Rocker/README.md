# Example Rocker Container

## Building the container
The build process follows the same steps detailed in the [introductory container example](../../0_Introductory/README.md#pulling-the-container), which you can use as a guide. Please refer to CCR's [container documentation](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/) for more information on using Apptainer.

On a compute node, navigate to your build directory and use the Slurm job local temporary directory for cache:
```
cd /projects/academic/[YourGroupName]/[CCRusername]/Rocker
export APPTAINER_CACHEDIR="${SLURMTMPDIR}"
```

Download the Rocker build file, `Rocker.def` to this directory.
```
curl -L -o Rocker.def https://raw.githubusercontent.com/BlazeCheesegrater/ccr-examples/refs/heads/main/containers/2_ApplicationSpecific/Rocker/Rocker.def
```

Once ready, build the container:
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

To edit the definition file, we are going to add a package from apt (a Linux package manager), CRAN (a general R package manager), and BioConductor (a more specific use case R package manager).

1. Navigate to your build directory.

```
cd /projects/academic/[CCRUsername]/Rocker
```

2. Add a Linux package to the definition file

We are going to add `neofetch` to the definition file first. This is a small program that shows useful information about the operating system. We can do this by using the simple text editor, `nano`.

Open the file with `nano`

```
nano Rocker.def
```

Add `neofetch \` on a new line between `apt-get update && apt-get install -y \` and `&& apt-get clean`. It should look like

```
apt-get update && apt-get install -y \
    neofetch \
    && apt-get clean
```

3. Add R packages to the definition file

We are going to add `rmarkdown` and `BiocGenerics` to our definition file. These are R files from two different package managers, so we will have to install them with their respective package managers. To do this, we will first uncomment `## Rscript -e "install.packages(c('[packages]'))"` and replace `[packages]` with `rmarkdown`. It should look like

```
## Install packages in R (from CRAN)
Rscript -e "install.packages(c('rmarkdown'))"
```

To install `BiocGenerics` we need to use the `Bioconductor` package manager. To do this we will first uncomment `## Rscript -e "[PackageManager]::install(c('[packages]'))"` and replace `[PackageManager]` with `BiocManager` and `[packages]` with `BiocGenerics`. It should look like

```
## Install packages from other maintainers in R
Rscript -e "BiocManager::install(c('BiocGenerics'))"
```

To save the file, type `ctrl + x`, then `y`, then hit the `enter` key.

4. Test the container

Ensure you are on a compute node. Run the container with:
```
apptainer shell rocker-x86_64.sif
```

You should see this
```
Apptainer>
```

Now we want to run the Linux program

```
neofetch
```

You should see something like
```
            .-/+oossssoo+/-.               [CCRUsername]@[NodeID].core.ccr.buffalo.edu 
        `:+ssssssssssssssssss+:`           ---------------------------------------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 24.04.3 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: PowerEdge R660xs 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 6.8.0-87-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 29 days, 42 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 516 (dpkg) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.2.21 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Resolution: 1024x768 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Terminal: /dev/pts/40 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: Intel Xeon Gold 6448Y (64) @ 4.100GHz 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   GPU: 03:00.0 Matrox Electronics Systems Ltd. Integrated Matrox G200eW3 Graphics Controller 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Memory: 56281MiB / 515478MiB 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/
  +sssssssssdmydMMMMMMMMddddyssssssss+                             
   /ssssssssssshdmNNNNmyNMMMMhssssss/                              
    .ossssssssssssssssssdMMMNysssso.
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`
            .-/+oossssoo+/-.
```

Now we will run R

```
R
```

You should see something like

```
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

We will now test if the R packages were installed

```
library(rmarkdown)
library(BiocGenerics)
```

If there is any output that looks like
```
Error in library([package]) : there is no package called ‘[package]’
```
there was an issue with the installation. Otherwise, everything is working as it should.


See CCR docs for more info on [Building Images with Apptainer](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/#building-images-with-apptainer)

See R-project docs for more info on [Installing R Packages](https://cran.r-project.org/doc/manuals/r-release/R-admin.html#Installing-packages)

## Additional Information

- The [Placeholders](../../../README.md#placeholders) section lists the available options for each placeholder used in the example scripts.
- For more info on accessing shared project and global scratch directories, resource options, and other important container topics, please refer to the CCR [container documentation](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/) 