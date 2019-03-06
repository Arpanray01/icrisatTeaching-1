import matplotlib.pyplot as plt

def plot_model_history(history,varToPlot='loss',nameFig='modelHistory.png'):
    """[summary]
    plots train and validation losses from a keras model
    Arguments:
        model_history {[type] keras model} -- [description] 
    """
    plt.plot(history.history[varToPlot],label="train")
    plt.plot(history.history['val_'+varToPlot],label="validation")
    plt.title('model '+varToPlot)
    plt.ylabel(varToPlot)
    plt.xlabel('epoch')
    plt.savefig(nameFig)