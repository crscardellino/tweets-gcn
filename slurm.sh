#!/bin/bash
### Lines "#SBATCH" configure the job resources
### (even though they look like bash comments)

### Job queue to use (options: batch)
#SBATCH --partition=batch

### Amount of nodes to use
#SBATCH --nodes=1

### Processes per node
#SBATCH --ntasks-per-node=1

### Available cores per node 
#SBATCH --cpus-per-task=12

### execution time. Format: days-hours:minutes:seconds -- Max: three days
#SBATCH --time 3-00:00:00

### Check invocation line
if [ $# -ne 1 ] || [ ! -f $1 ]
then
        echo "[ERROR] Must invoke through \"enqueue_job.sh\""
        exit 1
fi

## ## Load environment modules
## module load compilers/gcc/4.9

### Enqueue job
mkdir -p logs
srun -o logs/%j.out -e logs/%j.err /bin/bash $1

exit 0
