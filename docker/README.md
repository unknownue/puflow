
## Steps to start

```bash
git clone https://github.com/unknownue/PU-Flow.git
cd PU-Flow/docker
docker build -t pytorch/pu-flow -f Dockerflie .

cd ../
docker run -it --rm \
    -u $(id -u):$(id -g) \
    -e DISPLAY=unix$DISPLAY \
    -v $(pwd):/workspace/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /workspace \
    --name pu-gan-runtime \
    --gpus all \
    --shm-size 8G \
    pytorch/pu-flow
cd ../
python train_uflow.py
```
