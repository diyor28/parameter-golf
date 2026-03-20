FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

COPY requirements.txt /tmp/pgolf-requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/pgolf-requirements.txt && \
    rm -f /tmp/pgolf-requirements.txt
