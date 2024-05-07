#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=geneformer_bert_test1
VOCAB_FILE=/workspace/bgi/geneidversion_vocal.txt
DATA_PATH=gene_genes_document


GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 4 \
    --hidden-size 512 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 8 \
    --global-batch-size 32 \
    --lr 0.0001 \
    --train-iters 100000 \
    --lr-decay-iters 990 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --attention-softmax-in-fp32 \
    --use-checkpoint-opt_param-scheduler \
    --bert-no-binary-head \
    --no-gradient-accumulation-fusion \
"
    # --freeze-wte

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --split 94,5,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS /workspace/megatron/pretrain_uce.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH