"""Runs CNN federated learning for MNIST dataset."""

'''
FedProx는 FedAvg 보다 수렴에서 강하다
'''

from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

#from flower.baselines.flwr_baselines.publications.fedprox_mnist import client, utils
import client, utils

DEVICE: str = torch.device("cpu")


@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    client_fn, testloader = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds,
        iid=cfg.iid,
        balance=cfg.balance,
        learning_rate=cfg.learning_rate,
        stragglers=cfg.stragglers_fraction,  # 논문에서의 감마 의미 (낙오자 비율 정하는 hyperparameter)
                                             # 감마가 계속 변하면서 round epoch을 수정함
    )
    
    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=int(cfg.num_clients * (1 - cfg.stragglers_fraction)),
        min_evaluate_clients=0,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=lambda curr_round: {"curr_round": curr_round},
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        proximal_mu=cfg.mu,     # proximal term 계산에 사용되는 hyperparameter (정의된 파일: docs/conf/config.yaml)
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    file_suffix: str = (
        f"{'_iid' if cfg.iid else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
        f"_stag={cfg.stragglers_fraction}"
    )

    np.save(
        Path(cfg.save_path) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        cfg.save_path,
        (file_suffix),
    )


if __name__ == "__main__":
    main()
