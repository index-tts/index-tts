CUDA_VISIBLE_DEVICES=1 python api.py \
    --verbose \
    --port 8002 \
    --host 0.0.0.0 \
    --fp16 \
    --cuda_kernel \
    --fa2 \
    --compile
