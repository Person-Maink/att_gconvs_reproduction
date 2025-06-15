import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np, random
from experiments.omniglot import dataset, trainer     # adapt paths if different
from experiments.run_util import *

@hydra.main(version_base=None, config_path="experiment_configs", config_name="omniglot.yaml")  # default YAML
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = "cuda:0" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model).to(device)

    model_directory(cfg)

    loaders, test_loader = dataset.get_dataset(
        batch_size=cfg.batch_size,
        num_workers=4,
        augmentation=cfg.augment,
        root=cfg.root
    )

    if not cfg.pretrained:
        trainer.train(model, loaders, cfg)           # your existing trainer
    else:
        model.load_state_dict(torch.load(cfg.path))

    acc, _, _ = test(model, test_loader, device)
    print(f"Test accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()
