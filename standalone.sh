#!/bin/bash
#SBATCH --job-name=spark_standalone
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=spark-%j.out
#SBATCH --error=spark-%j.err
#SBATCH --mail-user=rvb5977@psu.edu
#SBATCH --mail-type=BEGIN,END

# -----------------------------
# Load modules and Python environment
# -----------------------------
module load anaconda3
module load jdk
source activate ds410
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
source .venv/bin/activate
#conda activate ds41

#python3 -m pip install optuna seaborn matplotlib plotly wordcloud -y

export PYSPARK_PYTHON=$PWD/.venv/bin/python3
export PYSPARK_DRIVER_PYTHON=$PWD/.venv/bin/python3

echo "modules loaded"

# -----------------------------
# Spark logs in current directory
# -----------------------------

# export SPARK_LOCAL_DIRS=$PWD/spark_local
# mkdir -p $SPARK_LOCAL_DIRS
export SPARK_WORKER_DIR=$PWD/spark_work
mkdir -p $SPARK_WORKER_DIR

# export SPARK_LOG_DIR=$PWD/spark-logs-$SLURM_JOB_ID
# mkdir -p $SPARK_LOG_DIR
# echo "Spark logs will be written to: $SPARK_LOG_DIR"

echo $SLURM_NNODES
echo $SLURN_NTASKS_PER_NODE
echo $SLURM_CPUS_PER_TASK

TEC=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE * $SLURM_CPUS_PER_TASK))
EM=$(($SLURM_MEM_PER_NODE / $SLURM_NTASKS_PER_NODE / 1024))
echo "total-executor-cores=${TEC}"
echo "executor-memory=${EM}"

MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_URL="spark://$MASTER_NODE:7077"

srun --nodes=1 --ntasks=1 --exclusive \
     spark-class org.apache.spark.deploy.master.Master \
     --host $MASTER_NODE --port 7077 &

sleep 15

for node in $(scontrol show hostnames $SLURM_NODELIST); do
    if [ "$node" != "$MASTER_NODE" ]; then
        srun --nodes=1 --ntasks=1 --nodelist=$node --exclusive \
             spark-class org.apache.spark.deploy.worker.Worker $MASTER_URL &
    fi
done

sleep 15

echo "Master URL: $MASTER_URL"

# Record the start time
start_time=$(date +%s)
$SPARK_HOME/bin/spark-submit --master $MASTER_URL --driver-memory 50G --executor-memory 60G --executor-cores 8 --conf spark.executor.memoryOverhead=8G main.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
