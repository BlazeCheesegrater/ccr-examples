# Example Script for Slurm Job Arrays

This directory includes an [example script](./job-array-example.bash) that shows how to run a Job Array on CCR's clusters. Be sure to modify parts of the script to suit your specific needs.

## What Are Job Arrays
Job Arrays let you submit multiple similar jobs using a single Slurm batch script. Each job in the array shares the same requested resources specified in your batch script.

## Specifying Arrays
The resource flag `--array` sets a job array with specified index values to be submitted. The formatting for this is as follows:  
```
#SBATCH --array=0-3      #  Submits a job array between index values 0 and 3.  
#SBATCH --array=1,3,5,7  #  Submits a job array with the index values of 1, 3, 5, and 7.
```

Note: The max amount of tasks that can run at a time depends on the partition it's being run on. For example, only 4 tasks can run simutaneously per user using the debug partition, but on the general-compute partition, up to 1000 tasks can be ran simutaneously per user. For job arrays larger than these, we can limit the amount of tasks running at once with the `%` operator. For example, if we submit `#SBATCH --array=0-100%4`, Slurm will run the first four tasks. When one finishes, the next task in the array will start automatically. For more information on partition limits, go to the second table [here](https://docs.ccr.buffalo.edu/en/latest/hpc/jobs/#slurm-directives-partitions-qos).
## Using Array IDs
As shown in [job-array-example.bash](./job-array-example.bash), the environment variables $SLURM_ARRAY_JOB_ID and $SLURM_ARRAY_TASK_ID can be used in the body of the batch script and passed to other commands such as `srun` or in the case of this example, `echo`. However, environment variables can NOT be used in `#SBATCH` directives such as `--output` and `--error`. Instead, you must use these format symbols to specify unique output file names:

- `%A` is the job array master job ID ($SLURM_ARRAY_JOB_ID)
- `%a` is the individual job index number ($SLURM_ARRAY_TASK_ID)
```
#SBATCH --output="slurm-%A_%a.out"  #  Generates files with names: "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
```

## Example:
```
##  Specify what jobs in the array will run
#SBATCH --array=0-3

##  Output files format
#SBATCH --output="slurm-%A_%a.out"
```
If this job array was run as job 12345678 this formatting would generate the output files:
- `"slurm-12345678_0.out"`
- `"slurm-12345678_1.out"`
- `"slurm-12345678_2.out"`
- `"slurm-12345678_3.out"`

An example of running this job, then checking the file outputs:
```
$ sbatch job-array-example.bash
Submitted batch job 22244918

$ cat slurm-22244918_0.out
Hello World from job 22244918, task 0 on node:cpn-h23-36.core.ccr.buffalo.edu

$ cat slurm-22244918_1.out
Hello World from job 22244918, task 1 on node:cpn-h23-36.core.ccr.buffalo.edu

$ cat slurm-22244918_2.out
Hello World from job 22244918, task 2 on node:cpn-h23-36.core.ccr.buffalo.edu

$ cat slurm-22244918_3.out
Hello World from job 22244918, task 3 on node:cpn-h23-36.core.ccr.buffalo.edu
```


  ## Learn More
- [Our Documentation](https://docs.ccr.buffalo.edu/en/latest/hpc/jobs/#job-arrays)
- [Slurm Documentation](https://slurm.schedmd.com/job_array.html)
