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


# curve native
for SEED in 1;
do
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --amp --epochs 3 --decay_start_step 134000 --test_freq 1000 --embedding_compression_type native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed ${SEED} | tee curve_native_fl15.txt
sleep 30
done


# train native fl15
for SEED in 1 2 3 4 5;
do
python -m dlrm.scripts.main --mode train --dataset /data/fl15/binary_dataset/ --save_checkpoint_path ./native_fl15_checkpoint --epochs 2 --decay_start_step 112000 --embedding_compression_type native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed ${SEED} | tee train_native_fl15_seed${SEED}.txt
sleep 30
done


# benchmark native fl15
for BATCH_SIZE in 2048 4096 8192 16384 24576 32768 49152 53248 57344;
do
python -m dlrm.scripts.deploy --batch_size ${BATCH_SIZE} --test_batch_size ${BATCH_SIZE} --dataset /data/fl15/binary_dataset/ --load_checkpoint_path ./native_fl15_checkpoint --amp --embedding_compression_type native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native,native --seed 1 | tee benchmark_native_fl15_batchsize${BATCH_SIZE}.txt
rm ./*.weight
rm ./*.onnx
rm ./*.engine
sleep 30
done



