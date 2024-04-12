# Copyright (c) 2023 BLUEWALM. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# curve frobenius fl15
for SEED in 1;
do
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --amp --epochs 3 --decay_start_step 134000 --test_freq 1000 --embedding_compression_type frobenius,frobenius,frobenius,frobenius,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed ${SEED} | tee curve_frobenius_default_fl15_nr4.txt
sleep 30
done


# train frobenius fl15
for SEED in 12345;
do
for r in 8 16;
do
for p in 1 2 4 8;
do
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --save_checkpoint_path ./frobenius_r${r}p${p}_fl15_nr4_checkpoint --amp --epochs 2 --decay_start_step 112000 --embedding_compression_type frobenius,frobenius,frobenius,frobenius,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank ${r},${r},${r},${r} --frobenius_blocks ${p},${p},${p},${p} --seed ${SEED} | tee train_frobenius_r${r}_p${p}_fl15_nr4_seed${SEED}.txt
sleep 30
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --save_checkpoint_path ./frobenius_r${r}p${p}_fl15_nr5_checkpoint --amp --epochs 3 --decay_start_step 176000 --embedding_compression_type frobenius,frobenius,frobenius,frobenius,frobenius,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --frobenius_rank ${r},${r},${r},${r},${r} --frobenius_blocks ${p},${p},${p},${p},${p} --seed ${SEED} | tee train_frobenius_r${r}_p${p}_fl15_nr5_seed${SEED}.txt
sleep 30
done
done
done


# train frobenius fl15
for SEED in 1 2 3 4 5;
do
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --save_checkpoint_path ./frobenius_default_fl15_nr4_checkpoint --amp --epochs 2 --decay_start_step 112000 --embedding_compression_type frobenius,frobenius,frobenius,frobenius,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed ${SEED} | tee train_frobenius_default_fl15_nr4_seed${SEED}.txt
sleep 30
done


# benchmark frobenius fl15
for ONCHIP in true false;
do
for BATCH_SIZE in 2048 4096 8192 16384 24576 32768 49152 53248 57344;
do
python -m dlrm.scripts.deploy --onchip_memory=${ONCHIP} --batch_size ${BATCH_SIZE} --test_batch_size ${BATCH_SIZE} --dataset /data/fl15/binary_dataset/ --load_checkpoint_path ./frobenius_default_fl15_nr4_checkpoint --amp --embedding_compression_type frobenius,frobenius,frobenius,frobenius,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed 1 | tee benchmark_frobenius_default_fl15_nr4_batchsize${BATCH_SIZE}_onchip${ONCHIP}.txt
rm ./*.weight
rm ./*.onnx
rm ./*.engine
sleep 30
done
done



