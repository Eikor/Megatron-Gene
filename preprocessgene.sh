python /workspace/megatron/tools/preprocess_data.py \
       --input /workspace/bgi10/geneidversion2.json \
       --output-prefix gene \
       --vocab-file /workspace/bgi10/geneidversion_vocal.txt \
       --tokenizer-type BertWordPieceCase \
       --workers 32 \
       --json-key genes \
       # --gene-vocab
