python srnn_class_scnn_enc.py -T 16 -device cuda:0 -b 1 -epochs 100 -data-dir ~/code/DVS_hand_object_detection/data/hand_object -amp -cupy -opt adam -lr 0.001 -j 12
