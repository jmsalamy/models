#!/bin/bash

#log into all vms, set up reqired scripts (work ot how this is easiest)

#add to python path (will be forgotten on log in, regardless of sourcing apparently
export PYTHONPATH="$PYTHONPATH:/afs/csail.mit.edu/u/j/jsalamy/horovod/models"

#parallel tcpdump

#open bashrc
source ~/.bashrc

#open venv
conda activate /home/jsalamy/gpu-tf

#run test script here
mpirun --allow-run-as-root --tag-output -np 4 -H localhost,jsalamy@roodabeh.csail.mit.edu,jsalamy@gordafarid.csail.mit.edu,jsalamy@tahmineh.csail.mit.edu \
-bind-to none -map-by slot -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include eno1 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eno1 -x CONDA_SHLVL \
-x LD_LIBRARY_PATH -x CONDA_EXE -x SSH_CONNECTION -x LANG -x SWT_GTK3 -x CONDA_PREFIX -x _CE_M -x XDG_SESSION_ID -x USER -x XILINXD_LICENSE_FILE -x PWD -x HOME \
-x SSH_CLIENT -x KRB5CCNAME -x _CE_CONDA -x CONDA_PROMPT_MODIFIER -x SSH_TTY -x MAIL -x TERM -x SHELL -x SHLVL -x LANGUAGE -x LOGNAME -x  DBUS_SESSION_BUS_ADDRESS \
-x PATH -x CONDA_DEFAULT_ENV --verbose \
python ~/horovod/models/official/resnet/imagenet_main.py --batch_size 64 --data_dir /usr/data/imagenet/processed/combined/ --resnet_size 50 --train_epochs 1 --max_train_steps 500 --benchmark_logger_type BenchmarkFileLogger --benchmark_log_dir resnet50_drop0_benchmark --model_dir resnet_model_dir

#exit successfully if you've go there
exit 0