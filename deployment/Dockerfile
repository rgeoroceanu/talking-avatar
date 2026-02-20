FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone MuseTalk (shallow clone for speed)
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

# Pin numpy first, then install MuseTalk Python dependencies
RUN pip install --no-cache-dir "numpy==1.23.5" && \
    pip install --no-cache-dir \
    "diffusers==0.30.2" \
    "accelerate==0.28.0" \
    "transformers==4.39.2" \
    "huggingface_hub==0.30.2" \
    "librosa==0.11.0" \
    "einops==0.8.1" \
    "omegaconf" \
    "ffmpeg-python" \
    "soundfile==0.12.1" \
    "opencv-python-headless" \
    "imageio[ffmpeg]"

# Install mmcv stack for DWPose face detection
RUN pip install --no-cache-dir openmim && \
    mim install mmengine "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.2.0"

# Install server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    python-multipart \
    edge-tts \
    && pip install --no-cache-dir "numpy==1.23.5"


# Download model checkpoints via huggingface-cli
# SD-VAE (ft-mse)
RUN huggingface-cli download stabilityai/sd-vae-ft-mse \
    --include "config.json" "diffusion_pytorch_model.bin" \
    --local-dir /app/models/sd-vae

# MuseTalk V1.5 UNet (3.4GB - force-download to handle partial/corrupt downloads)
RUN huggingface-cli download TMElyralab/MuseTalk \
    --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" \
    --local-dir /app/models --force-download

# Whisper-tiny
RUN huggingface-cli download openai/whisper-tiny \
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json" \
    --local-dir /app/models/whisper

# DWPose checkpoint
RUN huggingface-cli download yzd-v/DWPose \
    --include "dw-ll_ucoco_384.pth" \
    --local-dir /app/models/dwpose

# Face parsing BiSeNet
RUN mkdir -p /app/models/face-parse-bisent && \
    wget -q --show-progress \
    "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
    -O /app/models/face-parse-bisent/resnet18-5c106cde.pth && \
    pip install --no-cache-dir gdown && \
    gdown "154JgKpzCPW82qINcVieuPH3fZ2e0P812" \
    -O /app/models/face-parse-bisent/79999_iter.pth

# SFD face detection (downloaded at runtime by face_detection lib; pre-cache it)
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q --show-progress \
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
    -O /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth

# Symlink models so MuseTalk code finds them at ./models/ relative to workdir
RUN ln -sf /app/models /app/MuseTalk/models

# Bake avatar into image (no volume mount needed)
COPY examples/idle.mp4 /app/idle.mp4

# Copy application code (last layer for fast rebuilds)
COPY server.py /app/server.py
COPY engine/ /app/engine/
COPY static/ /app/static/
COPY deployment/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Symlink server.py into MuseTalk dir (start.sh runs uvicorn from there
# because MuseTalk preprocessing uses relative paths like ./musetalk/utils/dwpose/)
RUN ln -sf /app/server.py /app/MuseTalk/server.py

EXPOSE 8080

ENTRYPOINT ["/app/start.sh"]
