import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class Plotter():
    def __init__(self, *args):
        self.Data = args
        self.fig = plt.figure()
        self.fig.subplots_adjust(hspace=0.5)
        self.loclegend = 'upper right'
        sns.set(style='whitegrid', context='paper',font_scale=1.5, palette=sns.color_palette("muted"))

    def plotEnvelope(self, Regions):
        Time = self.Data[-1]
        ax1 = self.fig.add_subplot(2,1,1, title=f'Region {Regions[0]}')
        ax2 = self.fig.add_subplot(2,1,2, title=f'Region {Regions[1]}')
        ax1.plot(Time, self.Data[0], label='Reference Signal')
        ax1.plot(Time, self.Data[1], label='Reference Envelope')
        ax1.legend(loc=self.loclegend)

        ax2.plot(Time,self.Data[2], label='Signal')
        ax2.plot(Time, self.Data[3], label='Envelope')
        ax2.plot(Time, self.Data[4], label='Orthogonalized Envelope')
        ax2.legend(loc=self.loclegend)

        plt.show()
        pass

    def plotFC(self):
        gs = figure.add_gridspec(2, 6)
        a = figure.add_subplot(gs[:, :2])
        divider = make_axes_locatable(a)
        cax = divider.append_axes('right', size="5%", pad=0.05)
        im = a.imshow(data)
        figure.colorbar(im, cax=cax)
        c = figure.add_subplot(gs[:, 3:5])
        im = c.imshow(data * 2)
        figure.colorbar(im, ax=c)

        pass

    def plotMetastability(self):
        pass

    def plotGraphMeasures(self):
        pass

x = np.linspace(0,6,100)
Signal = np.sin(x)
Plotter(np.sin(x), np.sin(x+np.pi), np.cos(x), np.cos(x+np.pi), x).plotEnvelope([1,2])