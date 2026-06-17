import argparse
import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    name: str
    conv_scale: int = 1
    extra_conv_before_fc: bool = False
    extra_fc_after_fc1: bool = False
    pool_type: str = "max"
    dropout_conv: float = 0.25
    dropout_fc: float = 0.5
    use_batch_norm: bool = False
    optimizer_name: str = "adam"


class CNN(nn.Module):
    def __init__(self, config: ExperimentConfig, num_classes: int = 10):
        super().__init__()
        c1 = 32 * config.conv_scale
        c2 = 64 * config.conv_scale

        self.config = config
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1) if config.use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c1) if config.use_batch_norm else nn.Identity()

        pool_layer = nn.AvgPool2d if config.pool_type == "avg" else nn.MaxPool2d
        self.pooling1 = pool_layer(kernel_size=2)
        self.dropout1 = nn.Dropout(config.dropout_conv)

        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c2) if config.use_batch_norm else nn.Identity()
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c2) if config.use_batch_norm else nn.Identity()
        self.extra_conv = (
            nn.Conv2d(c2, c2, kernel_size=3, padding=1)
            if config.extra_conv_before_fc
            else nn.Identity()
        )
        self.bn_extra = (
            nn.BatchNorm2d(c2)
            if config.use_batch_norm and config.extra_conv_before_fc
            else nn.Identity()
        )

        self.pooling2 = pool_layer(kernel_size=2)
        self.dropout2 = nn.Dropout(config.dropout_conv)

        self.fc1 = nn.Linear(c2 * 8 * 8, 512)
        self.extra_fc = (
            nn.Linear(512, 512) if config.extra_fc_after_fc1 else nn.Identity()
        )
        self.dropout3 = nn.Dropout(config.dropout_fc)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn_extra(self.extra_conv(x)))
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        if self.config.extra_fc_after_fc1:
            x = F.relu(self.extra_fc(x))
        x = self.dropout3(x)
        return self.fc2(x)


def build_experiments():
    return [
        ExperimentConfig(name="baseline"),
        ExperimentConfig(name="double_conv_channels", conv_scale=2),
        ExperimentConfig(name="extra_conv_before_fc", extra_conv_before_fc=True),
        ExperimentConfig(name="extra_fc_after_fc1", extra_fc_after_fc1=True),
        ExperimentConfig(name="avg_pooling", pool_type="avg"),
        ExperimentConfig(name="dropout_0_50", dropout_conv=0.5, dropout_fc=0.5),
        ExperimentConfig(name="dropout_0_05", dropout_conv=0.05, dropout_fc=0.05),
        ExperimentConfig(name="batch_norm", use_batch_norm=True),
        ExperimentConfig(name="sgd_optimizer", optimizer_name="sgd"),
    ]


def get_dataloaders(data_dir, batch_size, train_subset=None, test_subset=None):
    transform = transforms.ToTensor()
    trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    if train_subset:
        trainset = Subset(trainset, range(min(train_subset, len(trainset))))
    if test_subset:
        testset = Subset(testset, range(min(test_subset, len(testset))))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainset, testset, trainloader, testloader


def visualize_dataset(trainset, output_dir):
    output_path = output_dir / "cifar10_examples.png"
    labels = np.array([trainset[i][1] for i in range(len(trainset))])
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))

    for class_index in range(10):
        indices = np.where(labels == class_index)[0]
        for image_index in range(10):
            ax = axes[class_index][image_index]
            if image_index < len(indices):
                image, _ = trainset[int(indices[image_index])]
                ax.imshow(image.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if image_index == 0:
                ax.set_ylabel(CLASS_NAMES[class_index], rotation=0, labelpad=35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def make_optimizer(model, config, learning_rate):
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def run_one_epoch(model, dataloader, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    desc = "train" if train else "test"
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / total_samples, total_correct / total_samples


def train_and_evaluate(config, trainloader, testloader, args, device):
    model = CNN(config).to(device)
    optimizer = make_optimizer(model, config, args.learning_rate)
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print(f"\n===== Experiment: {config.name} =====")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, trainloader, optimizer, device, train=True
        )
        test_loss, test_acc = run_one_epoch(
            model, testloader, optimizer, device, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    return model, history


def save_model(model, config, output_dir):
    output_path = output_dir / f"{config.name}.pth"
    torch.save(
        {
            "config": config.__dict__,
            "state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path, output_path.stat().st_size / 1024 / 1024


def load_saved_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = ExperimentConfig(**checkpoint["config"])
    model = CNN(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def plot_loss_curves(histories, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in histories.items():
        ax.plot(history["train_loss"], label=f"{name} train")
        ax.plot(history["test_loss"], linestyle="--", label=f"{name} test")

    ax.set_title("Loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross entropy loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = output_dir / "loss_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_summary(results, output_dir):
    output_path = output_dir / "experiment_summary.csv"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("experiment,final_train_loss,final_train_acc,final_test_loss,final_test_acc,model_size_mb\n")
        for result in results:
            history = result["history"]
            f.write(
                f"{result['name']},"
                f"{history['train_loss'][-1]:.6f},"
                f"{history['train_acc'][-1]:.6f},"
                f"{history['test_loss'][-1]:.6f},"
                f"{history['test_acc'][-1]:.6f},"
                f"{result['model_size_mb']:.6f}\n"
            )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR10 CNN experiments for machine learning experiment 2."
    )
    parser.add_argument("--data-dir", default="./cifar10")
    parser.add_argument("--output-dir", default="./outputs_experiment2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small subsets and 1 epoch for a fast code check.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip saving the CIFAR10 example image grid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.train_subset = args.train_subset or 512
        args.test_subset = args.test_subset or 256

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainset, testset, trainloader, testloader = get_dataloaders(
        args.data_dir,
        args.batch_size,
        args.train_subset,
        args.test_subset,
    )
    print(f"Train samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

    if not args.skip_visualization:
        image_grid_path = visualize_dataset(trainset, output_dir)
        print(f"Saved dataset visualization: {image_grid_path}")

    experiments = build_experiments()
    histories = {}
    results = []
    saved_paths = {}

    for config in experiments:
        set_seed(args.seed)
        model, history = train_and_evaluate(config, trainloader, testloader, args, device)
        model_path, model_size_mb = save_model(model, config, model_dir)
        histories[config.name] = history
        saved_paths[config.name] = model_path
        results.append(
            {
                "name": config.name,
                "history": history,
                "model_size_mb": model_size_mb,
            }
        )
        print(f"Saved model: {model_path} ({model_size_mb:.2f} MB)")

    summary_path = write_summary(results, output_dir)
    loss_curve_path = plot_loss_curves(histories, output_dir)

    loaded_model, loaded_config = load_saved_model(saved_paths["baseline"], device)
    original_state = torch.load(saved_paths["baseline"], map_location=device)["state_dict"]
    loaded_ok = all(
        torch.equal(loaded_model.state_dict()[key], value)
        for key, value in original_state.items()
    )
    print(f"Loaded baseline model config: {loaded_config.name}")
    print(f"Save/load check passed: {loaded_ok}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved loss curves: {loss_curve_path}")


if __name__ == "__main__":
    main()
