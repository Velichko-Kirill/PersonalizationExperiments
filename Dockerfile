ARG POETRY_VERSION=1.8.2

FROM harbor.anb.lan/docker/python:3.11

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_PATH=/opt/poetry
#    VENV_PATH=/opt/venv \
#    POETRY_VIRTUALENVS_CREATE=false

#ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"

RUN #pip3 install pip --upgrade
ARG POETRY_VERSION
RUN pip install poetry==${POETRY_VERSION}
COPY poetry.lock pyproject.toml ./
RUN poetry install
#COPY --from=poetry $VENV_PATH $VENV_PATH

WORKDIR /biography_generator
COPY . ./
ENTRYPOINT ["python", "-m", "biography_generator"]

