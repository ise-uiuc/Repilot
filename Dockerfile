FROM ubuntu:22.04

# Install dependencies
RUN apt-get -y update \
    && apt-get -y install software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    python-is-python3 \
    subversion \
    curl \
    cpanminus \
    unzip

# Set the working directory to /root

# Install coursier
WORKDIR /root

RUN mkdir bin \
    && cd bin \
    && curl -fL "https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz" | gzip -d > cs \
    && chmod +x cs \
    && yes n | ./cs setup

ENV PATH="/root/bin:${PATH}"

RUN cs java --jvm 8 --env \
    && cs java --jvm 11 --env \
    && cs java --jvm 18 --env

# Install the modified version of the Eclipse JDT Language Server
RUN git clone https://github.com/UniverseFly/eclipse.jdt.ls --depth 1 \
    && cd eclipse.jdt.ls \
    && eval $(cs java --jvm 11 --env) \
    && ./mvnw clean verify -DskipTests

# Copy the current directory contents into the container
COPY . Repilot
RUN cd Repilot \
    && pip install --upgrade pip \
    && pip install -e . \
    && git config --global user.email "repilot@example.com" \
    && git config --global user.name "Repilot" \
    && git config --global init.defaultBranch main \
    && git init \
    # Repilot requires at least one commit message as of now
    && git add README.md \
    && git commit -m "Welcome to Repilot!"

# Install Defects4J
RUN git clone https://github.com/rjust/defects4j \
    && cd defects4j \
    && git checkout v2.0.0 -b v2.0.0 \
    && cpanm --installdeps . \
    && ./init.sh

# Generate the meta_config.json and checkout all Defects4J bugs
WORKDIR /root/Repilot
RUN python generate_meta_config_in_docker.py \
    && python -m repilot.cli.init
