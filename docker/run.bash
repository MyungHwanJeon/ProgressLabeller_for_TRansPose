xhost +local:

docker run --gpus all -it --env="DISPLAY" --net=host --ipc=host --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /:/mydata --cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined jmong1994/transpose:ProgressLabeller
