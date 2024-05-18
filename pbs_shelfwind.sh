#!/bin/bash -l
#PBS -A UCLA0052
#PBS -N refreflo
#PBS -k eod
#PBS -o logs/DNSkerr.out
#PBS -e logs/DNSkerr.err
#PBS -l walltime=12:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l gpu_type=v100
#PBS -M zhazorken@gmail.com
#PBS -m abe

# Clear the environment from any previously loaded modules
module purge
module load ncarenv/23.10
module load julia/1.9.2 cuda

module load peak-memusage
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

#export JULIA_DEPOT_PATH="/glade/work/tomasc/.julia_bkp"
export JULIA_DEPOT_PATH="/glade/work/kenzhao/.julia"

peak_memusage julia --project shelfwind.jl --simname=DNSgayen3casper --factor=2 2>&1 | tee out/DNSgayen1.out
