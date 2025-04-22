record_txt="./output/turl-cta.txt"
python3 inference_rel_extraction_col_type.py \
    --base_model osunlp/TableLlama \
    --context_size 8192 \
    --max_gen_len 128 \
    --input_data_file "./dataset/turl-cta.json" \
    --output_data_file "./output/turl-cta.json" \
    > "$record_txt"

record_txt="./output/turl-cpa.txt"
python3 inference_rel_extraction_col_type.py \
    --base_model osunlp/TableLlama \
    --context_size 8192 \
    --max_gen_len 128 \
    --input_data_file "./dataset/turl-cpa.json" \
    --output_data_file "./output/turl-cpa.json" \
    > "$record_txt"
