# Example Open Force Field toolkit container

Please Note: the graphical part of openff-toolkit does not work at CCR because "nglview" must be run in a browser

## Building the container

A brief guide to building the openff-toolkit container follows:<br/>
Please refer to CCR's [container documentation](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/) for more detailed information on building and using Apptainer.

1. Start an interactive job

Apptainer is not available on the CCR login nodes and the compile nodes may not provide enough resources for you to build a container.  We recommend requesting an interactive job on a compute node to conduct this build process.<br/>
See CCR docs for more info on [running jobs](https://docs.ccr.buffalo.edu/en/latest/hpc/jobs/#interactive-job-submission)

```
$ salloc --cluster=ub-hpc --partition=debug --qos=debug --mem=64GB --time=01:00:00
salloc: Pending job allocation 19781052
salloc: job 19781052 queued and waiting for resources
salloc: job 19781052 has been allocated resources
salloc: Granted job allocation 19781052
salloc: Nodes cpn-i14-39 are ready for job
CCRusername@cpn-i14-39:~$ 

```

2. Navigate to your build directory & set a temp directory for cache

You should now be on the compute node allocated to you.  In this example we're using our project directory for our build directory.  Ensure you've placed your `openff-toolkit.def` and `environment.yml` file in your build directory

```
CCRusername@cpn-i14-39:~$ cd /projects/academic/[YourGroupName]/[CCRusername]
CCRusername@cpn-i14-39:~$ mkdir cache
CCRusername@cpn-i14-39:~$ export APPTAINER_CACHEDIR=/projects/academic/[YourGroupName]/[CCRusername]/cache

```

3. Build your container

```
CCRusername@cpn-i14-39:~$ apptainer build openff-toolkit-$(arch).sif openff-toolkit.def
...
...
INFO:    Adding environment to container
INFO:    Creating SIF file...
INFO:    Build complete: openff-toolkit-x86_64.sif
```

## Running the container

Start an interactive job e.g.

```
$ salloc --cluster=ub-hpc --partition=general-compute --qos=general-compute --mem=128GB --nodes=1 --cpus-per-task=1 --tasks-per-node=32 --time=05:00:00
```

...then run the container


```
CCRusername@cpn-q06-35-02:~$ cd /projects/academic/[YourGroupName]/[CCRusername]
CCRusername@cpn-q06-35-02$ apptainer exec -B /scratch:/scratch,/projects:/projects openff-toolkit-$(arch).sif python
Python 3.12.10 | packaged by conda-forge | (main, Apr 10 2025, 22:21:13) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Alternatively, you can first get shell access into the container and then run Python, as shown here:

```
CCRusername@@cpn-q06-35-02:~$ apptainer shell -B /scratch:/scratch,/projects:/projects openff-toolkit-$(arch).sif
Apptainer> python
Python 3.12.10 | packaged by conda-forge | (main, Apr 10 2025, 22:21:13) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

In either case, once you are in python you can use the Open Force Field toolkit<br/>
For example:

```
>>> from openff.toolkit import Molecule, Topology
>>> from openff.toolkit.utils import get_data_file_path
>>> m = Molecule.from_smiles("CCCCOCC")
>>> m.generate_conformers()
>>> 
```

Unfortunately "nglview", the Python widget to interactively view molecular structures and trajectories, does not work at CCR; hence the following produces an empty output:

```
>>> import nglview
>>> m.visualize(backend="nglview")
NGLWidget()
>>> 
```

We hope to address this in the future.<br/>

See the [Open Force Field toolkit](https://docs.openforcefield.org/projects/toolkit/en/stable) website and the [Open Force Field](https://openforcefield.org) website for more info on Open Force Field

