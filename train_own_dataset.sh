

#python train.py --model_name supervised_M_640x416_1 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --num_epochs 20
#python evaluate_depth.py --model_name supervised_M_640x416_1 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/supervised_M_640x416_1/models/weights_19 --eval_mono
#python train.py --model_name supervised_M_640x416_2 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --num_epochs 20
#python evaluate_depth.py --model_name supervised_M_640x416_2 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/supervised_M_640x416_2/models/weights_19 --eval_mono
#python train.py --model_name supervised_M_640x416_3 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --num_epochs 20
#python evaluate_depth.py --model_name supervised_M_640x416_3 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/supervised_M_640x416_3/models/weights_19 --eval_mono
#python train.py --model_name supervised_M_640x416_4 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --num_epochs 20
#python evaluate_depth.py --model_name supervised_M_640x416_4 --height 416 --width 640 --png --dataset own_supervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/supervised_M_640x416_4/models/weights_19 --eval_mono

python evaluate_depth.py --model_name unsupervised_M_640x416_pil_load --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/unsupervised_M_640x416_pil_load/models/weights_19 --eval_mono
python train.py --model_name unsupervised_M_640x416_2 --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --num_epochs 20
python evaluate_depth.py --model_name unsupervised_M_640x416_2 --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/unsupervised_M_640x416_2/models/weights_19 --eval_mono
python train.py --model_name unsupervised_M_640x416_3 --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --num_epochs 20
python evaluate_depth.py --model_name unsupervised_M_640x416_3 --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/unsupervised_M_640x416_3/models/weights_19 --eval_mono
python train.py --model_name unsupervised_M_640x416_4 --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --num_epochs 20
python evaluate_depth.py --model_name unsupervised_M_640x416_4 --height 416 --width 640 --png --dataset own_unsupervised --log_dir Log_own_dataset --load_weights_folder Log_own_dataset/unsupervised_M_640x416_4/models/weights_19 --eval_mono
