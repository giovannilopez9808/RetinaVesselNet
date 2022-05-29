from Modules.params import get_params, organization_files
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join

params = get_params("normal")
params = organization_files(params)

datasets_parameters = {
    "files": ["history_01.csv",
              "history_02.csv",
              "history_03.csv"],
    "loss": {
        "y lim": [0.15, 0.6],
        "title": "train loss"
    },
    "accuracy": {
        "y lim": [0.8, 0.92],
        "title": "train accuracy"},
    "val_loss": {
        "y lim": [0.15, 0.5],
        "title": "validation loss"
    },
    "val_accuracy": {
        "y lim": [0.8, 0.94],
        "title": "validation accuracy"
    },
}
colors = {"history_01.csv": {"color": "#03071e",
                             "title": "All"},
          "history_02.csv": {"color": "#9d0208",
                             "title": "Last conv"},
          "history_03.csv": {"color": "#f48c06",
                             "title": "None"},
          }
history = {}
for filename in datasets_parameters["files"]:
    file = join(params["path results"],
                filename)
    data = read_csv(file,
                    index_col=0)
    history[filename] = data.copy()

fig, axs = plt.subplots(2, 2,
                        figsize=(14, 8))
axs = axs.flatten()
for train_type in history:
    data = history[train_type]
    color = colors[train_type]["color"]
    title = colors[train_type]["title"]
    for ax, parameter in zip(axs, data.columns):
        dataset = datasets_parameters[parameter]
        parameter_data = data[parameter]
        epochs = list(data.index)
        ax.plot(epochs,
                parameter_data,
                label=title,
                color=color,
                ls="--",
                marker="o")
        ax.set_title(datasets_parameters[parameter]["title"])
        ax.set_xlabel("número de epoca")
        ax.set_xlim(0, 14)
        ax.set_ylim(dataset["y lim"][0],
                    dataset["y lim"][1])
        ax.grid(ls="--",
                color="#000000",
                alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles,
           labels,
           bbox_to_anchor=(0.62,
                           1.01),
           ncol=3,
           frameon=False
           )
plt.tight_layout()
filename = join(params["path graphics"],
                "history.png")
plt.savefig(filename,
            dpi=400)
