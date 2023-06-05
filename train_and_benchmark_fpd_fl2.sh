



# train fpd fl2
#python -m dlrm.scripts.main --mode train --dataset /data/fl2/binary_dataset/ --save_checkpoint_path ./fpd8x1_fl2_checkpoint --amp --epochs 3 --decay_start_step 176000 --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 1,1,1,1,1 --seed 1 | tee train_fpd8x1_fl2_nr5.txt

# train fpd fl2
#python -m dlrm.scripts.main --mode train --dataset /data/fl2/binary_dataset/ --save_checkpoint_path ./fpd8x2_fl2_checkpoint --amp --epochs 3 --decay_start_step 176000 --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 2,2,2,2,2 --seed 1 | tee train_fpd8x2_fl2_nr5.txt

# train fpd fl2
python -m dlrm.scripts.main --mode train --dataset /data/fl2/binary_dataset/ --save_checkpoint_path ./fpd8x4_fl2_checkpoint --amp --epochs 3 --decay_start_step 176000 --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 4,4,4,4,4 --seed 1 | tee train_fpd8x4_fl2_nr5.txt




# test fpd fl2
#python -m dlrm.scripts.main --mode test --dataset /data/fl2/binary_dataset/ --load_checkpoint_path ./fpd8x1_fl2_checkpoint --amp --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 1,1,1,1,1 | tee test_fpd8x1_fl2_nr5.txt

# test fpd fl2
#python -m dlrm.scripts.main --mode test --dataset /data/fl2/binary_dataset/ --load_checkpoint_path ./fpd8x2_fl2_checkpoint --amp --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 2,2,2,2,2 | tee test_fpd8x2_fl2_nr5.txt

# test fpd fl2
python -m dlrm.scripts.main --mode test --dataset /data/fl2/binary_dataset/ --load_checkpoint_path ./fpd8x4_fl2_checkpoint --amp --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 4,4,4,4,4 | tee test_fpd8x4_fl2_nr5.txt





# benchmark fpd fl2
#python -m dlrm.scripts.main --mode inference_benchmark --dataset /data/fl2/binary_dataset/ --load_checkpoint_path ./fpd8x1_fl2_checkpoint --amp --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 1,1,1,1,1 | tee benchmark_fpd8x1_fl2_nr4.txt

# benchmark fpd fl2
#python -m dlrm.scripts.main --mode inference_benchmark --dataset /data/fl2/binary_dataset/ --load_checkpoint_path ./fpd8x2_fl2_checkpoint --amp --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 2,2,2,2,2 | tee benchmark_fpd8x2_fl2_nr5.txt

# benchmark fpd fl2
python -m dlrm.scripts.main --mode inference_benchmark --dataset /data/fl2/binary_dataset/ --load_checkpoint_path ./fpd8x4_fl2_checkpoint --amp --embedding_compression_type fpd,fpd,fpd,fpd,fpd,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank 8,8,8,8,8 --frobenius_blocks 4,4,4,4,4 | tee benchmark_fpd8x4_fl2_nr5.txt




