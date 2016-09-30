#!/usr/bin/env bash 
export CUDA_AUTO_BOOST=0
echo "clock,bandwidth" > $1
for clock in 875 862 849 836 823 810 797 784 771 758 745 732 719 705 692 679 666 653 640 627 614 601 588 575 562; do #
	srun nvidia-smi -ac 2505,$clock -i 0 
	echo -n "$clock," >> $1
	srun ./stream.bin | cut -d " " -f 6 >> $1
	srun nvidia-smi -q -d CLOCK -i 0 | grep "Applications Clocks" -A 2
done
