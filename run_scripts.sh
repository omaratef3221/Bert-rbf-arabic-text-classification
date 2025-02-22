python train.py --device cuda --EPOCHS 10 --BATCH_SIZE 32 --model_id UBC-NLP/MARBERT
python train.py --EPOCHS 10 --device cuda --BATCH_SIZE 32 --enable_rbf 1 --num_kernels 2 --kernel_name gaussian --model_id UBC-NLP/MARBERT
python train.py --EPOCHS 10 --device cuda --BATCH_SIZE 32 --enable_rbf 1 --num_kernels 2 --kernel_name  multiquadric_kernel --model_id UBC-NLP/MARBERT
python train.py --EPOCHS 10 --device cuda --BATCH_SIZE 32 --enable_rbf 1 --num_kernels 2 --kernel_name  linear --model_id UBC-NLP/MARBERT