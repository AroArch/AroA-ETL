# ETL Repository for Arolsen Archives [![Test ETL Pipeline](https://github.com/AroArch/AroA-ETL/actions/workflows/tests.yml/badge.svg)](https://github.com/AroArch/AroA-ETL/actions/workflows/tests.yml)

Workflows

- Person Entities: Cluster person data to person entities and assign unique identifiers
- Data Integration: Clean, standadize and transform data.

## Install Package
build wheel

```uv build```

install with pip

```pip install /path/to/repo```

## Run Tests
Run all tests 

```uv tool run pytest -v tests/```

or without uv (install pytest first)

```pytest -v tests/```

## Person Entities 
A work environment is setup via a docker container. 

First build the container:
```docker build -t aroa-clustering -f ./scripts/clustering-container/Dockerfile .```

Then run a clustering on a dataset that is in `datadir/data.csv`. 

Note: Change `datadir` to the path where the person data file is located and rename `data.csv` to the filename.

```docker run -it -v $(pwd)/datadir:/home/user/workdir/datadir aroa-clustering:latest datadir/data.csv```

## Data Integration
A work environment is setup via a docker container. Alternatively, the
notebooks in the workflow subfolder can be used in other python environemts.

First build the container:

```docker build -t aroa-etl -f ./scripts/etl-docker/Dockerfile .```

```podman build -t aroa-etl -f ./scripts/etl-docker/Dockerfile .```

Then run the container. Change `PORT` to a free port on your system, e.g. `8844`:

```docker run -it -p PORT:8888 aroa-etl:latest```

or with podman on wsl

```podman run -it -p PORT:8888 --network=host aroa-etl:latest```

or with persistent directory

```podman run -it -p PORT:8888 --network=host  -v /home/user/code/enc_notebooks:/home/user/workdir/notebook aroa-etl:latest```

## LICENCE
Copyright (c) 2024 Arolsen Archives

Distributed under the MIT Licence.
