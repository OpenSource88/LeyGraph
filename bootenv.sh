export CUDA_VISIBLE_DEVICES=1,2,3,4
#export CUDA_VISIBLE_DEVICES=1,2
echo $CUDA_VISIBLE_DEVICES

#export NCCL_DEBUG=info
#export NCCL_DEBUG=version
export NCCL_SOCKET_IFNAME=eno3 #change accoding to your machine
export NCCL_IB_DISABLE=1
