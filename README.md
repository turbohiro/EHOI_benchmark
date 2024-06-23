# EHoI: A Benchmark for Task-oriented Hand-Object Interaction Recognition Using Event Camera
<img src="figures/task_types-1.png" width="1000" border="1"/>

# Datasets
* It is stored in Onedriver `https://onedrive.live.com/?authkey=%21AOQ3qRCQh8A32hQ&id=392AC0752D520BEF%2110229&cid=392AC0752D520BEF`
  
# Training and Evaluate
* cd  `spikingjelly_pkg/spikingjelly/activation_based/examples` folder, Adjust the model type by change run `net` parameters, run `python -m spikingjelly.activation_based.examples.classify_hand_object -T 16 -device cuda:0 -b 2 -epochs 100  -data-dir /media/wchen/linHDD/wchen/hand_object/  -amp -cupy  -opt adam -lr 0.001 -j 12`
* Or cd `src` folder, run `.evaluate.sh`
