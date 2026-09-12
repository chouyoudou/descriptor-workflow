ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PROFILE_DIGEST
ARG LOCK_SHA256
ARG BASE_IMAGE
LABEL org.opencontainers.image.source="https://github.com/chouyoudou/descriptor-workflow" \
      io.descriptor.runtime.profile="local-environment-cpu-v1" \
      io.descriptor.runtime.profile-digest="${PROFILE_DIGEST}" \
      io.descriptor.runtime.lock-sha256="${LOCK_SHA256}" \
      io.descriptor.runtime.base-image="${BASE_IMAGE}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
COPY wheelhouse /wheelhouse
COPY resolved.lock /tmp/resolved.lock
RUN python3 -m pip install --no-index --find-links=/wheelhouse \
      -r /tmp/resolved.lock --quiet \
    && rm -rf /wheelhouse /root/.cache/pip
