# A dockerfile describing a container with all necessary dependencies
# to use the full cbadc feature suit.

# Use the official python image as a parent image
FROM ubuntu:latest

LABEL maintainer="Hampus Malmberg <hampus.malmberg88@gmail.com>"
ARG DEV_USER="cbadc"
ARG DEV_UID="1000"
ARG DEV_GID="100"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    locales \
    sudo \
    htop \
    wget \
    python3.10 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV SHELL=/bin/bash \
    DEV_USER="${DEV_USER}" \
    DEV_UID=${DEV_UID} \
    DEV_GID=${DEV_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME="/home/${DEV_USER}"

# Copy a script that we will use to correct permissions after running certain commands
COPY docker/dev/fix-permissions.sh /usr/local/bin/fix-permissions.sh
RUN chmod a+rx /usr/local/bin/fix-permissions.sh

# Create DEV_USER with name cbadc user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -l -m -s /bin/bash -N -u "${DEV_UID}" "${DEV_USER}" && \
    chmod g+w /etc/passwd && \
    fix-permissions.sh "${HOME}"

# install ngspice
WORKDIR /tmp
COPY docker/install_ngspice.sh /tmp/install_ngspice.sh
RUN bash /tmp/install_ngspice.sh

USER ${DEV_UID}

RUN python3.10 -m pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install -r /tmp/requirements.txt
RUN python3.10 -m pip install bandit black flake8 pytest pytest-cov pytest-benchmark pytest-mock pytest-xdist tox
# RUN export '${PATH}=${PATH}:/home/cbadc/.local/bin'
