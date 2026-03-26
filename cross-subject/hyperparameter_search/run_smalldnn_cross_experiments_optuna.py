"""Run SmallDNN Optuna tuning without CCA preprocessing."""

from run_smalldnn_optuna import main


if __name__ == "__main__":
    main(use_cca_default=False)
