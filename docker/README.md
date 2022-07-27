
## Steps to start

```bash
git clone https://github.com/unknownue/puflow.git
cd puflow/docker
docker build -t unknownue/nf -f Dockerflie .

cd ../../
docker run -id --rm \
   -e DISPLAY=unix$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $(pwd):/workspace \
   --device /dev/nvidia0 \
   --device /dev/nvidia-uvm \
   --device /dev/nvidia-uvm-tools \
   --device /dev/nvidiactl \
   --gpus all \                                                    
   --name nf \    
   -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,compute,utility \
   -w /workspace \
   --shm-size 8G \
   unknownue/nf

docker exec -it -w /workspace/ nf bash
```
