############################################# BASE
FROM python:3.12-slim-bookworm AS aroa-etl-base

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates curl nano git

# Ensure the installed binary is on the `PATH`
RUN useradd -ms /bin/bash user
WORKDIR /home/user
USER user

# Download the latest installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin/:$PATH"

# install aroa processing package
WORKDIR /home/user/aroa_etl

COPY --chown=user:user ./src/ ./src
COPY --chown=user:user ./pyproject.toml ./
COPY --chown=user:user ./uv.lock ./
COPY --chown=user:user ./tests/ ./tests
COPY --chown=user:user ./README.md ./
COPY --chown=user:user .python-version ./

RUN uv sync
RUN uv build

############################################## MATCHING

FROM aroa-etl-base AS aroa-elt-matching

WORKDIR /home/user/workdir

COPY --chown=user:user ./scripts/matching-container/README.md ./
COPY --chown=user:user ./scripts/matching-container/pyproject.toml ./
COPY --chown=user:user ./scripts/matching-container/uv.lock ./
COPY --chown=user:user ./scripts/matching-container/.python-version ./

RUN uv sync

COPY --chown=user:user ./scripts/dbquery-container/queries.py ./ 
COPY --chown=user:user ./scripts/matching-container/update-persdata.py ./
COPY --chown=user:user ./scripts/matching-container/run-matching.py ./
COPY --chown=user:user ./.db_credentials ./
COPY --chown=user:user ./scripts/dbquery-container/loadcredentials.py ./

RUN uv add ../aroa_etl
# podman build -t aroa-etl-matching --target=aroa-elt-matching -f Dockerfile .
# podman run -it -v persdata:/persdata -v /path/to/matching1/:/home/user/workdir/code --network=host aroa-etl-matchingg:latest /bin/bash

############################################## Clustering

FROM aroa-etl-base AS aroa-person-clustering

WORKDIR /home/user/workdir

COPY --chown=user:user ./scripts/person-clustering-container/README.md ./
COPY --chown=user:user ./scripts/person-clustering-container/pyproject.toml ./
COPY --chown=user:user ./scripts/person-clustering-container/uv.lock ./
COPY --chown=user:user ./scripts/person-clustering-container/.python-version ./

RUN uv sync

COPY --chown=user:user ./scripts/person-clustering-container/run_clustering.py ./

RUN uv add ../aroa_etl

ENTRYPOINT  ["uv", "run", "run_clustering.py"] 

CMD ["datadir/data.csv"]

# docker build -t aroa-person-clustering --target=aroa-person-clustering .
# docker run -it -v $(pwd)/datadir:/home/user/workdir/datadir aroa-person-clustering:latest datadir/data.csv

############################################## Clustering
FROM aroa-etl-base AS aroa-dbquery

WORKDIR /home/user/workdir

COPY --chown=user:user ./scripts/dbquery-container/README.md ./
COPY --chown=user:user ./scripts/dbquery-container/pyproject.toml ./
COPY --chown=user:user ./scripts/dbquery-container/uv.lock ./
COPY --chown=user:user ./scripts/dbquery-container/.python-version ./
COPY --chown=user:user ./scripts/dbquery-container/load_data.ipynb ./
COPY --chown=user:user ./scripts/dbquery-container/persons_by_docid.ipynb ./
COPY --chown=user:user ./scripts/dbquery-container/queries.py ./
COPY --chown=user:user ./scripts/dbquery-container/loadcredentials.py ./
# TODO First create a db_credentials file with loadcredentials.py
# or remove this line and generate the credentials while executing the code. 
# You will be prompted for them
COPY --chown=user:user ./scripts/dbquery-container/db_credentials ./

RUN uv sync
# docker build -t dbquery --target=aroa-dbquery .
# docker run -it -v /path/to/query/data:/home/user/workdir/data dbquery:latest

############################################## Deduplication
FROM aroa-etl-base AS aroa-deduplication

WORKDIR /home/user/workdir

COPY --chown=user:user ./scripts/deduplication-container/README.md ./
COPY --chown=user:user ./scripts/deduplication-container/pyproject.toml ./
COPY --chown=user:user ./scripts/deduplication-container/uv.lock ./
COPY --chown=user:user ./scripts/deduplication-container/.python-version ./
COPY --chown=user:user ./scripts/deduplication-container/deduplication_template.py ./
RUN uv sync

# docker build -t aroa-deduplication --target=aroa-deduplication .
# docker run -it -v /path/to/deduplication/data:/home/user/workdir/data aroa-deduplication:latest

############################################## ETL for ENC
FROM aroa-etl-base AS aroa-etl

COPY --chown=user:user ./scripts/etl-container/README.md ./
COPY --chown=user:user ./scripts/etl-container/pyproject.toml ./
COPY --chown=user:user ./scripts/etl-container/uv.lock ./
COPY --chown=user:user ./scripts/etl-container/.python-version ./

RUN uv sync

COPY --chown=user:user ./scripts/etl-container/01_unpacking.ipynb ./
COPY --chown=user:user ./scripts/etl-container/02_processing.ipynb ./
COPY --chown=user:user ./scripts/etl-container/03_deduplicate.ipynb ./

RUN mkdir 00_raw_data
RUN mkdir 01_unpacked_data
RUN mkdir 02_processed_data
RUN mkdir 03_marked_for_qa
RUN mkdir 04_qa_reviewed
RUN mkdir 05_final_for_export

RUN uv add ../aroa_etl

CMD  ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888"] 

# docker build -t aroa-etl --target=aroa-etl .
# docker run -it -p PORT:8888 aroa-etl:latest


