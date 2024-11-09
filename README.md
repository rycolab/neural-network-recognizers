# Training Neural Networks as Recognizers of Formal Languages

This repository contains the code for the paper "Training Neural Networks as
Recognizers of Formal Languages" (Butoi et al., 2024). It includes all of the
code necessary to reproduce the experiments and figures in the paper, as well
as a Docker image definition that can be used to replicate the software
environment it was developed in.

The FLaRe benchmark, which contains the datasets for all formal languages used
in our experiments, can be found [here](https://github.com/rycolab/flare).

Our experiments our fast enough that they can be run quite quickly even without
GPUs. It takes only a couple hours or so to run all experiments on a scientific
computing cluster in CPU mode.

## Directory Structure

* `experiments/`: Contains high-level scripts for reproducing all of the
  experiments and figures in the paper.
* `scripts/`: Contains helper scripts for setting up the software environment,
  building container images, running containers, installing Python packages,
  etc. Instructions for using these scripts are below.
* `src/`: Contains source code for training neural networks,
  generating data, etc.
  * `recognizers/`: Contains source code for training neural networks,
    generating data, etc.
    * `analysis/`: Code for generating plots, analyzing predictions, etc.
    * `automata/`: Data structures and algorithms for automata.
    * `hand_picked_languages/`: Implementations of each language.
    * `neural_networks/`: Code for training and evaluating neural networks.
    * `string_sampling/`: Code for sampling positive and negative strings.
  * `rayuela/`: Helper library for automata.
* `tests/`: Contains pytest unit tests for the code under `src/`.

## Installation and Setup

In order to foster reproducibility, the code for this paper was developed and
run inside of a [Docker](https://www.docker.com/) container defined in the file
[`Dockerfile-dev`](Dockerfile-dev). To run this code, you can build the
Docker image yourself and run it using Docker. Or, if you don't feel like
installing Docker, you can simply use `Dockerfile-dev` as a reference for
setting up the software environment on your own system. You can also build
an equivalent [Singularity](https://sylabs.io/docs/#singularity) image which
can be used on an HPC cluster, where it is likely that Docker is not available
but Singularity is. There is a script that automatically sets up the Docker
container and opens a shell in it (instructions below).

### Using Docker

In order to use the Docker image, you must first
[install Docker](https://www.docker.com/get-started).
If you intend to run any experiments on a GPU, you must also ensure that your
NVIDIA driver is set up properly and install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Our experiments are quite fast even on CPU.

In order to automatically build the Docker image, start the container, and open
up a bash shell inside of it, run

    $ bash scripts/docker_shell.bash --build

After you have built the image once, there is no need to do so again, so
afterwards you can simply run

    $ bash scripts/docker_shell.bash

By default, this script starts the container in GPU mode, which will fail if
you are not running on a machine with a GPU. If you only want to run things in
CPU mode, you can run

    $ bash scripts/docker_shell.bash --cpu

### Using Singularity

Singularity is an alternative container runtime that is more suitable for
shared computing environments. Note: Singularity also goes by the name
Apptainer; they refer to the same thing.

In order to run the code in a Singularity container, you must first obtain the
Docker image and then convert it to a `.sif` (Singularity image) file on a
machine where you have root access (for example, your personal computer or
workstation). This requires installing both Docker and
[Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html)
on that machine. Assuming you have already built the Docker image according to
the instructions above, you can use the following to create the `.sif` file:

    $ bash scripts/build_singularity_image.bash

This will create the file `neural-network-recognizers.sif`. It is normal for
this to take several minutes. Afterwards, you can upload the `.sif` file to
your HPC cluster and use it there.

You can open a shell in the Singularity container using

    $ bash scripts/singularity_shell.bash

This will work on machines that do and do not have an NVIDIA GPU, although it
will output a warning if there is no GPU.

You can find a more general tutorial on Singularity
[here](https://github.com/bdusell/singularity-tutorial).

### Additional Setup

Whatever method you use to run the code (whether in a Docker container,
Singularity container, or no container), you must run this script once (*inside
the container shell*):

    $ bash scripts/setup.bash

Specifically, this script installs the Python packages required by our code,
which will be stored in the local directory rather than system-wide.

## Running Code

All files under `src/` should be run using `poetry` so they have access to the
Python packages provided by the Poetry package manager. This means you should
either prefix all of your commands with `poetry run` or run `poetry shell`
beforehand to enter a shell with Poetry's virtualenv enabled all the time. You
should run both Python and Bash scripts with Poetry, because the Bash scripts
might call out to Python scripts. All Bash scripts under `src/` should be run
with `src/` as the current working directory.

All scripts under `scripts/` and `experiments/` should be run with the
top-level directory as the current working directory.

## Running Experiments

The [`experiments/`](experiments) directory contains scripts for reproducing
all of the experiments and plots presented in the paper. Some of these scripts
are intended to be used to submit jobs to a computing cluster. They should be
run outside of the container. You will need to edit the file
[`experiments/submit_job.bash`](experiments/submit_job.bash)
to tailor it to your specific computing cluster. Other scripts are for plotting
or printing tables and should be run inside the container.

### Dataset Generation

Scripts for generating all of our datasets from scratch can be found under
`experiments/dataset_generation/`. All datasets are sampled using a fixed
random seed, so the results are deterministic.

For convenience, we have also included plaintext versions of all of our formal
language datasets in the [FLaRe repository](https://github.com/rycolab/flare).

Note that the plaintext files still need to be "prepared" (converted to
integers in .pt files) before being used to train neural networks using our
code.

* `submit_prepare_automaton_language_jobs.bash`: Generate and prepare all
  datasets for all regular languages.
* `submit_prepare_hand_coded_language_jobs.bash`: Generate and prepare all
  datasets for all non-regular languages.
* `submit_prepare_test_edit_distance_jobs.bash`: Generate and prepare the
  datasets used for the edit distance plots. Must be run after
  `submit_prepare_automaton_language_jobs.bash`.

Dataset generation consists of the following steps:

1. (For regular languages only) Write the DFA for the language to a .pt file.
2. (For regular languages only) Run weight pushing on the DFA so it can be used
   for sampling.
3. Randomly sample positive and negative examples for each split, and save the
   results as plaintext files.
4. Prepare the plaintext files by converting all symbols to integers and saving
   them in .pt files.

### Training Neural Networks

The relevant scripts are under `experiments/training/`. They should be run
after datasets are generated and prepared.

* `submit_train_and_evaluate_jobs.bash`: Train and evaluate all models on all
  languages.
* `submit_rerun_evaluate_jobs.bash`: Rerun just the model evaluation without
  retraining. Useful if there was an error during evaluation that does not
  require retraining the model.

### Analysis

The relevant scripts are under `experiments/analysis/`. They should be run
after models are trained and evaluated.

* `print_full_tables.bash`: Generate the tables of unabridged results for all
  languages.
* `print_main_tables.bash`: Generate the table summarizing all inductive bias
  and expressivity experiments, and the table showing the best loss functions.
* `plot_cross_entropy_vs_edit_distance.bash`: Generate the plots of
  cross-entropy vs. edit distance.
* `print_hardest_examples.bash`: Sort examples in the test set from highest to
  lowest cross-entropy for a certain language and architecture.
