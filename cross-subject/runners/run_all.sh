# python run_cca_experiments.py
# python run_dnn_cross_experiments.py
# python run_dnncca_cross_experiments.py
python run_eegnet_experiments.py
python run_eegnetcca_cross_experiments.py
# python run_eegnetfbcca_cross_experiments.py

# python run_smalldnn_experiments.py --use-cca --user-start 1 --user-end 35 --num-freqs 8 --window 1.0 --learning-rate 0.001 --batch-size 64 --weight-decay 0.0002 --dropout-rate 0.6 --n-filters 64 --results-subdir SMALLDNN_CCA_1sub
# python run_smalldnn_experiments.py --no-use-cca --user-start 1 --user-end 35 --num-freqs 8 --window 1.0 --learning-rate 6e-05 --batch-size 32 --weight-decay 3.6e-06 --dropout-rate 0.68 --n-filters 64 --results-subdir SMALLDNN_1sub