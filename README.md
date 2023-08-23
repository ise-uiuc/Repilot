# ⚙️$`\mathbb{R}\mathrm{ectify}`$🛠️

Welcome to the source code repo of **Rectify**, a patch generation tool introduced in our ESEC/FSE'23 paper "Copiloting the Copilot: Fusing Large Language Models with Completion Engines for Automated Program Repair"!

![A Demo of Rectify](/assets/Rectify-Demo.svg)

Rectify leverages the synergy between a semantics-based code completion engine and an auto-regressive large language model for more efficient valid patch generation.

> [!IMPORTANT]
> Rectify is implemented for Java patch generation as a complex hybrid system combining a [Modified Eclipse JDT Language Server](https://github.com/UniverseFly/eclipse.jdt.ls) and Python's [huggingface/transformers](https://github.com/huggingface/transformers) interface for manipulating large language models. Correctly setting up the dependencies and configurations of Rectify can be non-trivial. Therefore, **we highly recommend directly using our out-of-the-box Docker image**.

## 🚀 Quick start with Rectify's Docker image

```bash
# Pull the image and run a container.
# This may take some time...
docker run -it --name rectify universefly/rectify-fse23
# Now you will get into a "virtual environment" provided by Docker
# Enter the `rectify` directory
cd /rectify
# This is important because Rectify relies on a `meta_config.json` file to work properly
cat meta_config.json
# Generate patches with the full Rectify approach using CodeT5
ACTIVE=1 python -m rectify.cli.main repair -b "Chart-9" --method pruned-mem -d chart-9-rectify -n 5
# You will see logs about the patch generation and which tokens are accepted/rejected.

# Validate the patch generation
python -m rectify.cli.main validate -d chart-9-rectify

# Print a table of the evaluation results
python -m rectify.cli.main evaluate -d chart-9-rectify
# You'll see something like this:
#                                              Rectify Evaluation Results                                              
# ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
# ┃ Tag             ┃ Average Gen Time ┃ %Compilable Patches ┃ %Plausible Patches ┃ #Plausible Fixes ┃ #Correct Fixes ┃
# ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
# │ chart-9-rectify │ 1.33s            │ 100.0%              │ 0.000%             │ 0                │ -              │
# └─────────────────┴──────────────────┴─────────────────────┴────────────────────┴──────────────────┴────────────────┘
```

## ️⭐️ Artifact️

For a more comprehensive guidance on how to use Rectify and how to reproduce the results in our paper, we greatly encourage you to check our artifact at https://github.com/UniverseFly/Rectify-Artifact.


## ⚠️ How to build and use Rectify from source?

> [!WARNING]
> Building Rectify from source is **NOT** recommended since there are many complex dependencies and configurations to handle. It is only for advanced users who want to extend Rectify.

> [!IMPORTANT]
> **Environment requirements**
> 
> - Python 3.10 and [Git LFS](https://git-lfs.com) are required.
> - **All three versions of Java 8, 11, and 18** are required. For convenient management of multiple Java versions, we recommend [coursier](https://get-coursier.io/docs/cli-java).
> - (Optional) It's recommended to have an NVIDIA GPU with >6G memory for running Rectify with CodeT5 and >30G memory for Incoder-6.7B.

<details><summary><b>Download and build the modified Eclipse JDT Language Server</b></summary>

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

<details><summary><b>Download and install Rectify as a Python package including its dependencies</b></summary>

```bash
git clone https://github.com/UniverseFly/Rectify && cd Rectify
# Do an editable install
pip install -e .
# Consider upgrading pip if you encounter any errors, also make sure you are using Python 3.10
# This command should also install all the dependencies of Rectify
```
</details>

<details><summary><b>Prepare the runtime environment of Rectify</b></summary>

We need to prepare a `meta_config.json` file for Rectify to work properly. The file should be placed in the root directory of Rectify. Please **modify** the following template according to your environment and save the file in the root directory of Rectify:

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
</details>

<details><summary><b>Install the Defects4j datasets</b></summary>

Rectify evaluates on the [Defects4j](https://github.com/rjust/defects4j) dataset. Please checkout to its [v2.0.0 release](https://github.com/rjust/defects4j/releases/tag/v2.0.0) and follow its instructions to install the dataset.

> [!WARNING]
> If you directly download the release instead of doing a checkout you may encounter errors when running Rectify, as Rectify will dump the metadata by collecting the meta information of these projects as Git repos. If they are not Git repos, Rectify may fail.

After the installation, you should have a `defects4j` command in your `$PATH`. You can check it by running `defects4j info -p Chart`.

> [!IMPORTANT]
> The `defects4j` command must be in your `$PATH` for Rectify to work properly.

Now let's `cd` back to the root directory of Rectify, and run the following command to checkout all the bugs:

```bash
python -m rectify.cli.init
```

</details>


<details><summary><b>Do an example run</b></summary>

```bash
# Generate patches with the full Rectify approach using CodeT5
ACTIVE=1 python -m rectify.cli.main repair -b "Chart-9" --method pruned-mem -d chart-9-rectify -n 5
# You will see logs about the patch generation and which tokens are accepted/rejected.

# Validate the patch generation
python -m rectify.cli.main validate -d chart-9-rectify

# Print a table of the evaluation results
python -m rectify.cli.main evaluate -d chart-9-rectify
```

You will see a table of evaluation results if everything goes well.
</details>

<details><summary><b>(Optional) Unpack the pre-generated patches</b></summary>
The GitHub repo also contains pre-generated patches the experiments in our paper. You can unpack if you would like to check them. First make sure you `cd` to the root directory of Rectify. Then run the following command:

```bash
tar -xvf ./data/large.tar.xz
```

Then you will see the `data/large` directory is populated with the pre-generated patches.

</details>

🔥🔥**Congratulations! You have successfully built and used Rectify from source!**🔥🔥
