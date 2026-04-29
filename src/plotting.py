import matplotlib.pyplot as plt

def plot_dispatch(dispatch, load, period, title, savepath):
    ax = dispatch.loc[period].plot.area(figsize=(12,5))
    load.loc[period].plot(ax=ax, color="black")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()