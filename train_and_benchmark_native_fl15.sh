



# train native fl15
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --save_checkpoint_path ./native_fl15_checkpoint --amp --epochs 2 --decay_start_step 112000 --embedding_compression_type native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed 1 | tee train_native_fl15.txt

# benchmark native fl15
python -m dlrm.scripts.main --mode inference_benchmark --dataset /data/fl15/binary_dataset/ --load_checkpoint_path ./native_fl15_checkpoint --amp --embedding_compression_type native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native | tee benchmark_native_fl15.txt




