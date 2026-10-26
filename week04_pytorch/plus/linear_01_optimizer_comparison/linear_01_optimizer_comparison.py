
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch


## different optimizer:
# torch.optim.Adagrad
# torch.optim.Adam
# torch.optim.Adamax
# torch.optim.ASGD
# torch.optim.LBFGS
# torch.optim.RMSprop
# torch.optim.Rprop
# torch.optim.SGD

# Make the comparison reproducible.
torch.manual_seed(42)

# 1. Training data: the target relationship is y = 2x.
x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
y_data = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)


# 2. Define the model.
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Every optimizer starts from exactly the same initial weights and bias.
initial_model = LinearModel()
initial_state = deepcopy(initial_model.state_dict())

criterion = torch.nn.MSELoss(reduction="sum")

# Learning rates are optimizer-specific demonstration settings, not a formal benchmark.
# Different optimizers may need different hyperparameter tuning.
optimizer_factories = {
    "SGD": lambda params: torch.optim.SGD(params, lr=0.01),
    "Adagrad": lambda params: torch.optim.Adagrad(params, lr=0.2),
    "Adam": lambda params: torch.optim.Adam(params, lr=0.05),
    "Adamax": lambda params: torch.optim.Adamax(params, lr=0.1),
    "ASGD": lambda params: torch.optim.ASGD(params, lr=0.01),
    # LBFGS needs a closure that recomputes the loss and its gradients.
    "LBFGS": lambda params: torch.optim.LBFGS(
        params, lr=1.0, max_iter=1, line_search_fn="strong_wolfe"
    ),
    "RMSprop": lambda params: torch.optim.RMSprop(params, lr=0.01),
    "Rprop": lambda params: torch.optim.Rprop(params, lr=0.01),
}

epochs = 100
trained_models = {}
loss_histories = {}


# 3. Train each optimizer from the same initial state.
for name, optimizer_factory in optimizer_factories.items():
    model = LinearModel()
    model.load_state_dict(initial_state)
    optimizer = optimizer_factory(model.parameters())
    history = []

    for epoch in range(epochs):
        if name == "LBFGS":
            # LBFGS may evaluate this closure multiple times during one optimizer step.
            def closure():
                optimizer.zero_grad()
                predictions = model(x_data)
                loss = criterion(predictions, y_data)
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            predictions = model(x_data)
            loss = criterion(predictions, y_data)
            loss.backward()
            optimizer.step()

        # Record the loss after the parameter update, so curves use post-update losses.
        with torch.no_grad():
            current_loss = criterion(model(x_data), y_data).item()
        history.append(current_loss)

    trained_models[name] = model
    loss_histories[name] = history


# 4. Print each optimizer's final results.
print("Optimizer comparison (lower final loss is better for this training set):")
for name, model in trained_models.items():
    with torch.no_grad():
        final_loss = criterion(model(x_data), y_data).item()
        prediction_at_4 = model(torch.tensor([[4.0]])).item()

    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    print(
        f"{name:8s} | loss={final_loss:.8g} | "
        f"w={weight:.6f} | b={bias:.6f} | prediction at x=4: {prediction_at_4:.6f}"
    )


# Save figures alongside this script.
output_dir = Path(__file__).resolve().parent

# 5. Figure 1: training-loss curves.
plt.figure(figsize=(10, 6))
for name, history in loss_histories.items():
    # A tiny positive floor lets a log-scale axis display losses that approach zero.
    plot_history = [max(value, 1e-12) for value in history]
    plt.plot(range(1, epochs + 1), plot_history, label=name)

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training loss (sum of squared errors)")
plt.title("Training loss comparison across optimizers")
plt.grid(True, which="both")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(output_dir / "optimizer_loss_comparison.png", dpi=160)


# 6. Figure 2: final fitted lines against the training data.
x_plot = torch.linspace(0.0, 4.0, 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(
    x_data.numpy().reshape(-1),
    y_data.numpy().reshape(-1),
    label="Training data",
    marker="o",
)

for name, model in trained_models.items():
    with torch.no_grad():
        y_plot = model(x_plot).numpy().reshape(-1)
    plt.plot(x_plot.numpy().reshape(-1), y_plot, label=name)

plt.xlabel("x")
plt.ylabel("y / predicted y")
plt.title("Final fitted lines from different optimizers")
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(output_dir / "optimizer_fit_comparison.png", dpi=160)

# Show both figures when running locally in an environment with a plotting display.
plt.show()
