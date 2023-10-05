# ⚙️$`\mathbb{R}\mathrm{e}\mathbf{pilot}`$🛠️

<p align="left">
    <a href="https://arxiv.org/abs/2309.00608"><img src="https://img.shields.io/badge/arXiv-2309.00608-b31b1b.svg?style=for-the-badge">
    <a href="https://doi.org/10.5281/zenodo.8281250"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.8281250-blue?style=for-the-badge">
    <a href="https://hub.docker.com/r/universefly/repilot/tags"><img src="https://img.shields.io/badge/docker-universefly%2Frepilot-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"></a>
</p>

Welcome to the source code repo of **Repilot**, a patch generation tool introduced in our ESEC/FSE'23 paper "Copiloting the Copilot: Fusing Large Language Models with Completion Engines for Automated Program Repair"!

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/assets/Repilot-Demo-Light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="/assets/Repilot-Demo-Dark.svg">
  <img alt="Repilot Demo" src="/assets/Repilot-Demo-Light.svg">
</picture>

Repilot leverages the synergy between a semantics-based code completion engine and an auto-regressive large language model for more efficient valid patch generation.

> [!IMPORTANT]
> Repilot is implemented for Java patch generation as a complex hybrid system combining a [Modified Eclipse JDT Language Server](https://github.com/UniverseFly/eclipse.jdt.ls) and Python's [huggingface/transformers](https://github.com/huggingface/transformers) interface for manipulating large language models. Correctly setting up the dependencies and configurations of Repilot can be non-trivial. Therefore, **we highly recommend directly using our out-of-the-box Docker image**.

## 🚀 Quick start with Repilot's Docker image

```bash
# Pull the image and run a container.
# This may take some time...
docker run -it --name repilot universefly/repilot:latest
# Now you will get into a "virtual environment" provided by Docker
# Enter the `Repilot` directory
cd /root/Repilot
# This is important because Repilot relies on a `meta_config.json` file to work properly
cat meta_config.json
# Generate patches with the full Repilot approach using CodeT5
ACTIVE=1 python -m repilot.cli.main repair -b "Chart-9" --method pruned-mem -d chart-9-repilot -n 5
# You will see logs about the patch generation and which tokens are accepted/rejected.

# Validate the patch generation
python -m repilot.cli.main validate -d chart-9-repilot

# Print a table of the evaluation results
python -m repilot.cli.main evaluate -d chart-9-repilot
# You'll see something like this:
#                                              Repilot Evaluation Results                                              
# ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
# ┃ Tag             ┃ Average Gen Time ┃ %Compilable Patches ┃ %Plausible Patches ┃ #Plausible Fixes ┃ #Correct Fixes ┃
# ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
# │ chart-9-repilot │ 1.33s            │ 100.0%              │ 0.000%             │ 0                │ -              │
# └─────────────────┴──────────────────┴─────────────────────┴────────────────────┴──────────────────┴────────────────┘
```

## ️⭐️ Artifact️

For more comprehensive guidance on how to use Repilot and how to reproduce the results in our paper, we greatly encourage you to check out our [artifact documentation](/README-Artifact.md).


## ⚠️ How to build and use Repilot from source?

> [!WARNING]
>
> Building Repilot from source is **NOT** recommended since there are many complex dependencies and configurations to handle. It is only for advanced users who want to extend Repilot. If you want to build from source, we also encourage you to check out our [Dockerfile](https://github.com/ise-uiuc/Repilot/blob/main/Dockerfile) for more details.

> [!IMPORTANT]
> **Environment requirements**
> 
> - Python 3.10 and [Git LFS](https://git-lfs.com) are required.
> - **All three versions of Java 8, 11, and 18** are required. For convenient management of multiple Java versions, we recommend [coursier](https://get-coursier.io/docs/cli-java).
> - (Optional) It's recommended to have an NVIDIA GPU with >6G memory for running Repilot with CodeT5 and >30G memory for Incoder-6.7B.

<details><summary>Download and build the modified Eclipse JDT Language Server</summary>

Follow the instructions in [the repo](https://github.com/UniverseFly/eclipse.jdt.ls) to build the modified Eclipse JDT Language Server. Note you will need Java 11:

```bash
git clone https://github.com/UniverseFly/eclipse.jdt.ls
cd eclipse.jdt.ls
JAVA_HOME=/path/to/java/11 ./mvnw clean verify -DskipTests=true
```

**Adjust** the following command according to your build to dry run the language server:

```bash
java \
	-Declipse.application=org.eclipse.jdt.ls.core.id1 \
	-Dosgi.bundles.defaultStartLevel=4 \
	-Declipse.product=org.eclipse.jdt.ls.core.product \
	-Dlog.level=ALL \
	-noverify \
	-Xmx1G \
	--add-modules=ALL-SYSTEM \
	--add-opens java.base/java.util=ALL-UNNAMED \
	--add-opens java.base/java.lang=ALL-UNNAMED \
	-jar ./plugins/org.eclipse.equinox.launcher_1.5.200.v20180922-1751.jar \
	-configuration ./config_linux \
	-data /path/to/data
```

If everything goes well, you can move on to the next step.
</details>

<details><summary>Download and install Repilot as a Python package including its dependencies</summary>

```bash
git clone https://github.com/UniverseFly/Repilot && cd Repilot
# Do an editable install
pip install -e .
# Consider upgrading pip if you encounter any errors, also make sure you are using Python 3.10
# This command should also install all the dependencies of Repilot
```
</details>


<details><summary>Install the Defects4j datasets</summary>

Repilot evaluates on the [Defects4j](https://github.com/rjust/defects4j) dataset. Please checkout to its [v2.0.0 release](https://github.com/rjust/defects4j/releases/tag/v2.0.0) and follow its instructions to install the dataset.

> [!WARNING]
> If you directly download the release instead of doing a checkout you may encounter errors when running Repilot, as Repilot will dump the metadata by collecting the meta information of these projects as Git repos. If they are not Git repos, Repilot may fail.

You can check the installation by running `/path/to/defects4j info -p Chart`.
</details>


<details><summary>Prepare the runtime environment of Repilot</summary>

We need to prepare a `meta_config.json` file for Repilot to work properly. The file should be placed in the root directory of Repilot. Please **modify** the following template according to your environment and save the file in the root directory of Repilot:

```json
{
  "d4j_home": "/home/yuxiang/Developer/defects4j",
  "d4j_checkout_root": "/home/yuxiang/Developer/d4j-checkout",
  "jdt_ls_repo": "/home/yuxiang/Developer/eclipse.jdt.ls",
  "java8_home": "/home/yuxiang/.cache/coursier/arc/https/github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u181-b13/OpenJDK8U-jdk_x64_linux_hotspot_8u181b13.tar.gz/jdk8u181-b13",
  "language_server_cmd": [
    "/home/yuxiang/.cache/coursier/arc/https/github.com/adoptium/temurin18-binaries/releases/download/jdk-18.0.2%252B9/OpenJDK18U-jdk_x64_linux_hotspot_18.0.2_9.tar.gz/jdk-18.0.2+9/bin/java",
    "-Declipse.application=org.eclipse.jdt.ls.core.id1",
    "-Dosgi.bundles.defaultStartLevel=4",
    "-Declipse.product=org.eclipse.jdt.ls.core.product",
    "-Dlog.level=ERROR",
    "-noverify",
    "-Xmx1G",
    "--add-modules=ALL-SYSTEM",
    "--add-opens",
    "java.base/java.util=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.lang=ALL-UNNAMED",
    "-jar",
    "/home/yuxiang/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/plugins/org.eclipse.equinox.launcher_1.6.400.v20210924-0641.jar",
    "-configuration",
    "/home/yuxiang/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/config_linux"
  ],
  "seed": 0
}
```

Now let's `cd` back to the root directory of Repilot, and run the following command to checkout all the Defects4J bugs:

```bash
python -m repilot.cli.init
```
</details>


<details><summary>Do an example run</summary>

```bash
# Generate patches with the full Repilot approach using CodeT5
ACTIVE=1 python -m repilot.cli.main repair -b "Chart-9" --method pruned-mem -d chart-9-repilot -n 5 # You will see logs about the patch generation and which tokens are accepted/rejected.

# Validate the patch generation
python -m repilot.cli.main validate -d chart-9-repilot

# Print a table of the evaluation results
python -m repilot.cli.main evaluate -d chart-9-repilot
```

You will see a table of evaluation results if everything goes well.
</details>

<details><summary>(Optional) Unpack the pre-generated patches</summary>

The GitHub repo also contains pre-generated patches for the experiments in our paper. You can unpack if you would like to check them. First, make sure you `cd` to the root directory of Repilot. Then run the following command:

```bash
tar -xvf ./data/large.tar.xz
```

Then you will see the `data/large` directory is populated with the pre-generated patches.

</details>

**🔥🔥Congratulations! You have successfully built and used Repilot from source!🔥🔥**
