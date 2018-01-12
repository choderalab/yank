"""
YANK Health Report Notebook formatter

This module handles all the figure formatting and processing to minimize the code shown in the Health Report Jupyter
Notebook. All data processing and analysis is handled by the main multistate.analyzers package,
mainly image formatting is passed here.
"""

import os
import yaml
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, NoNorm
from matplotlib import gridspec
from simtk import unit as units
from .. import version
from .. import analyze

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA


class HealthReportData(object):
    """
    Class which houses the data used for the notebook and the generation of all plots including formatting
    """
    def __init__(self, store_directory):
        """
        Initial data read in and object assignment

        Parameters
        ----------
        store_directory : string
            Location where the analysis.yaml file is and where the NetCDF files are
        """
        # Read in data
        analysis_script_path = os.path.join(store_directory, 'analysis.yaml')
        if not os.path.isfile(analysis_script_path):
            err_msg = 'Cannot find analysis.yaml script in {}'.format(store_directory)
            raise RuntimeError(err_msg)
        with open(analysis_script_path, 'r') as f:
            analysis = yaml.load(f)
        phases_names = []
        signs = {}
        analyzers = {}
        for phase, sign in analysis:
            phases_names.append(phase)
            signs[phase] = sign
            storage_path = os.path.join(store_directory, phase + '.nc')
            analyzers[phase] = analyze.get_analyzer(storage_path)
        self.phase_names = phases_names
        self.signs = signs
        self.analyzers = analyzers
        self.nphases = len(phases_names)
        # Assign flags for other sections along with their global variables
        # General Data
        self._general_run = False
        self.iterations = {}
        # Equilibration
        self._equilibration_run = False
        self.u_ns = {}
        self.nequils = {}
        self.g_ts = {}
        self.Neff_maxs = {}
        self._n_discarded = 0
        # Decorrelation break-down
        self._decorrelation_run = False
        # Mixing Run (state)
        self._mixing_run = False
        # Replica mixing
        self._replica_mixing_run = False
        self._free_energy_run = False

    def general_simulation_data(self):
        """
        General purpose simulation data on number of iterations, number of states, and number of atoms.
        This just prints out this data in a regular, formatted pattern.
        """
        iterations = {}
        nstates = {}
        natoms = {}
        for phase_name in self.phase_names:
            analyzer = self.analyzers[phase_name]
            try:
                positions = analyzer.reporter.read_sampler_states(0)[0].positions
                natoms[phase_name], _ = positions.shape
            except AttributeError:  # Trap unloaded checkpoint file
                natoms[phase_name] = 'No Cpt.'
            energies, _ = analyzer.reporter.read_energies()
            iterations[phase_name], nstates[phase_name], _ = energies.shape

        leniter = max(len('Iterations'), *[len(str(i)) for i in iterations.values()]) + 2
        lenstates = max(len('States'), *[len(str(i)) for i in nstates.values()]) + 2
        lennatoms = max(len('Num Atoms'), *[len(str(i)) for i in natoms.values()]) + 2
        lenleftcol = max(len('Phase'), *[len(phase) for phase in self.phase_names]) + 2

        lines = []
        headstring = ''
        headstring += ('{:^' + '{}'.format(lenleftcol) + '}').format('Phase') + '|'
        headstring += ('{:^' + '{}'.format(leniter) + '}').format('Iterations') + '|'
        headstring += ('{:^' + '{}'.format(lenstates) + '}').format('States') + '|'
        headstring += ('{:^' + '{}'.format(lennatoms) + '}').format('Num Atoms')
        lines.append(headstring)
        lenline = len(headstring)
        topdiv = '=' * lenline
        lines.append(topdiv)
        for phase in self.phase_names:
            phasestring = ''
            phasestring += ('{:^' + '{}'.format(lenleftcol) + '}').format(phase) + '|'
            phasestring += ('{:^' + '{}'.format(leniter) + '}').format(iterations[phase]) + '|'
            phasestring += ('{:^' + '{}'.format(lenstates) + '}').format(nstates[phase]) + '|'
            phasestring += ('{:^' + '{}'.format(lennatoms) + '}').format(natoms[phase])
            lines.append(phasestring)
            lines.append('-' * lenline)

        for line in lines:
            print(line)
        self.iterations = iterations
        self._general_run = True

    def generate_equilibration_plots(self, discard_from_start=1):
        """
        Create the equilibration scatter plots showing the trend lines, correlation time,
        and number of effective samples

        Returns
        -------
        equilibration_figure : matplotlib.figure
            Figure showing the equilibration between both phases

        """
        # Adjust figure size
        plt.rcParams['figure.figsize'] = 20, 6 * self.nphases * 2
        plot_grid = gridspec.GridSpec(self.nphases, 1)  # Vertical distribution
        equilibration_figure = plt.figure()
        # Add some space between the figures
        equilibration_figure.subplots_adjust(hspace=0.4)
        for i, phase_name in enumerate(self.phase_names):
            sub_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=plot_grid[i])
            analyzer = self.analyzers[phase_name]

            # Data crunching to get timeseries
            u_kln, _ = analyzer.get_states_energies()
            # TODO: Figure out how not to discard the first sample
            # Sample at index 0 is actually the minimized structure and NOT from the equilibrium distribution
            # This throws off all of the equilibrium data
            u_kln = u_kln[:, :, discard_from_start:]
            self._n_discarded = discard_from_start
            self.u_ns[phase_name] = analyzer.get_timeseries(u_kln)
            # Timeseries statistics
            g_t, Neff_t = analyze.analysis.get_equilibration_data_per_sample(self.u_ns[phase_name])
            self.Neff_maxs[phase_name] = Neff_t.max()
            self.nequils[phase_name] = Neff_t.argmax()
            self.g_ts[phase_name] = g_t[int(self.nequils[phase_name])]

            # FIRST SUBPLOT: energy scatter
            # Attach subplot to figure
            p = equilibration_figure.add_subplot(sub_grid[0])
            # Data assignment for plot generation
            y = self.u_ns[phase_name]
            N = y.size
            x = np.arange(N)
            # Scatter plot
            p.plot(x, y, 'k.')
            # Smoothed equilibrium, this is very crude but it works for large data
            tck = interpolate.splrep(x, y, k=5, s=N * 1E7)
            smoothed = interpolate.splev(x, tck, der=0)
            p.plot(x, smoothed, '-r', linewidth=4)
            # Nequil line
            ylim = p.get_ylim()
            p.vlines(self.nequils[phase_name], *ylim, colors='b', linewidth=4)
            p.set_ylim(*ylim)  # Reset limits in case vlines expanded them
            p.set_xlim([0, N])
            # Set text
            p.set_title(phase_name + " phase", fontsize=20)
            p.set_ylabel(r'$\Sigma_n u_n$ in kT', fontsize=20)

            # Extra info in text boxes
            subsample_string = 'Subsample Rate: {0:.2f}\nDecorelated Samples: {1:d}'.format(self.g_ts[phase_name], int(
                np.floor(self.Neff_maxs[phase_name])))
            if np.mean([0, N]) > self.nequils[phase_name]:
                txt_horz = 'right'
                txt_xcoord = 0.95
            else:
                txt_horz = 'left'
                txt_xcoord = 0.05
            smooth_index = {'right': -1, 'left': 0}  # condition y
            if np.mean(ylim) > smoothed[smooth_index[txt_horz]]:
                txt_vert = 'top'
                txt_ycoord = 0.95
            else:
                txt_vert = 'bottom'
                txt_ycoord = 0.05
            p.text(txt_xcoord, txt_ycoord,
                   subsample_string,
                   verticalalignment=txt_vert, horizontalalignment=txt_horz,
                   transform=p.transAxes,
                   fontsize=15,
                   bbox={'alpha': 1.0, 'facecolor': 'white'}
                   )

            # SECOND SUBPLOT: g_t trace
            g = equilibration_figure.add_subplot(sub_grid[1])
            g.plot(x[:-1], g_t)
            ylim = g.get_ylim()
            g.vlines(self.nequils[phase_name], *ylim, colors='b', linewidth=4)
            g.set_ylim(*ylim)  # Reset limits in case vlines expanded them
            g.set_xlim([0, N])
            g.set_ylabel(r'Decor. Time', fontsize=20)

            # THRID SUBPLOT: Neff trace
            ne = equilibration_figure.add_subplot(sub_grid[2])
            ne.plot(x[:-1], Neff_t)
            ylim = ne.get_ylim()
            ne.vlines(self.nequils[phase_name], *ylim, colors='b', linewidth=4)
            ne.set_ylim(*ylim)  # Reset limits in case vlines expanded them
            ne.set_xlim([0, N])
            ne.set_ylabel(r'Neff samples', fontsize=20)
            ne.set_xlabel(r'Iteration', fontsize=20)

        # Set class variables to be used elsewhere
        # Set flag
        self._equilibration_run = True
        return equilibration_figure

    def generate_decorrelation_plots(self, decorrelation_threshold=0.1):
        """

        Parameters
        ----------
        decorrelation_threshold : float, Optional
            When number of decorrelated samples is less than this percent of the total number of samples, raise a
            warning. Default: `0.1`.

        Returns
        -------
        decorrelation_figure : matplotlib.figure
            Figure showing the decorrelation pie chart data of how the samples are distributed between equilibration,
            correlation, and decorrelation.
        """
        if not self._general_run or not self._equilibration_run:
            raise RuntimeError("Cannot generate decorrelation data without general simulation data and equilibration "
                               "data first! Please run the corresponding functions/cells.")
        # Readjust figure output
        plt.rcParams['figure.figsize'] = 20, 8
        decorrelation_figure = plt.figure()
        decorrelation_figure.subplots_adjust(wspace=0.2)
        plotkeys = [100 + (10 * self.nphases) + (i + 1) for i in range(self.nphases)]  # Horizonal distribution
        for phase_name, plotid in zip(self.phase_names, plotkeys):
            # Create subplot
            analyzer = self.analyzers[phase_name]
            p = decorrelation_figure.add_subplot(plotid)
            # Determine toal number of iterations
            N = self.iterations[phase_name]
            labels = ['Decorrelated', 'Correlated', 'Equilibration']
            colors = ['#2c7bb6', '#abd0e0', '#fdae61']  # blue, light blue, and orange
            explode = [0, 0, 0.0]
            # Determine the wedges
            eq = self.nequils[phase_name] + self._n_discarded  # Make sure we include the discarded
            decor = int(np.floor(self.Neff_maxs[phase_name]))
            cor = N - eq - decor
            dat = np.array([decor, cor, eq]) / float(N)
            if dat[0] <= decorrelation_threshold:
                colors[0] = '#d7191c'  # Red for warning
            patch, txt, autotxt = p.pie(
                dat,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90 + 360 * dat[0] / 2,  # put center of decor at top
                counterclock=False,
                textprops={'fontsize': 14}
            )
            for tx in txt:  # This is the only way I have found to adjust the label font size
                tx.set_fontsize(18)
            p.axis('equal')
            p.set_title(phase_name + " phase", fontsize=20, y=1.05)
            # Generate warning if need be
            if dat[0] <= decorrelation_threshold:
                p.text(
                    0.5, -0.1,
                    "Warning! Fewer than {0:.1f}% samples are\nequilibrated and decorelated!".format(
                        decorrelation_threshold * 100),
                    verticalalignment='bottom', horizontalalignment='center',
                    transform=p.transAxes,
                    fontsize=20,
                    color='red',
                    bbox={'alpha': 1.0, 'facecolor': 'white', 'lw': 0, 'pad': 0}
                )
        # Set globals
        self._decorrelation_run = True
        return decorrelation_figure

    def generate_mixing_plot(self, mixing_cutoff=0.05, mixing_warning_threshold=0.90, cmap_override=None):
        """
        Generate the state diffusion mixing map as an image instead of array of number

        Parameters
        ----------
        mixing_cutoff : float
            Minimal level of mixing percent from state `i` to `j` that will be plotted.
            Domain: [0,1]
            Default: 0.05.
        mixing_warning_threshold : float
            Level of mixing where transition from state `i` to `j` generates a warning based on percent of total swaps.
            Domain (mixing_cutoff, 1)
            Default: `0.90`.
        cmap_override : None or string
            Override the custom colormap that is used for this figure in case the figure is too white or you wnat to
            do something besides the custom one here.

        Returns
        -------
        mixing_figure : matplotlib.figure
            Figure showing the state mixing as a color diffusion map instead of grid of numbers
        """
        # Set up image
        mixing_figure, subplots = plt.subplots(1, 2)
        # Create custom cmap goes from white to pure blue, goes red if the threshold is reached
        if mixing_cutoff is None:
            mixing_cutoff = 0
        if mixing_warning_threshold <= mixing_cutoff:
            raise ValueError("mixing_warning_threshold must be larger than mixing_cutoff")
        if (mixing_warning_threshold > 1 or mixing_cutoff > 1 or
                    mixing_warning_threshold < 0 or mixing_cutoff < 0):
            raise ValueError("mixing_warning_threshold and mixing_cutoff must be between [0,1]")
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (mixing_cutoff, 1.0, 1.0),
                         (mixing_warning_threshold, 0.0, 0.0),
                         (mixing_warning_threshold, 1.0, 1.0),
                         (1.0, 1.0, 1.0)),

                 'green': ((0.0, 1.0, 1.0),
                           (mixing_cutoff, 1.0, 1.0),
                           (mixing_warning_threshold, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),

                 'blue': ((0.0, 1.0, 1.0),
                          (mixing_cutoff, 1.0, 1.0),
                          (mixing_warning_threshold, 1.0, 1.0),
                          (mixing_warning_threshold, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        if cmap_override is not None:
            # Use this cmap instead if your results are too diffuse to see over the white
            cmap = plt.get_cmap("Blues")
        else:
            cmap = LinearSegmentedColormap('BlueWarnRed', cdict)

        # Plot a diffusing mixing map for each phase.
        for phase_name, subplot in zip(self.phase_names, subplots):
            # Generate mixing statistics.
            analyzer = self.analyzers[phase_name]
            mixing_statistics = analyzer.generate_mixing_statistics(
                number_equilibrated=self.nequils[phase_name])
            transition_matrix, eigenvalues, statistical_inefficiency = mixing_statistics

            # Without vmin/vmax, the image normalizes the values to mixing_data.max
            # which screws up the warning colormap.
            # Can also use norm=NoNorm(), but that makes the colorbar manipulation fail.
            output_image = subplot.imshow(transition_matrix, aspect='equal',
                                          cmap=cmap, vmin=0, vmax=1)
            # Add colorbar.
            decimal = 2  # Precision setting
            nticks = 11
            # The color bar has to be configured independently of the source image
            # or it cant be truncated to only show the data. i.e. it would instead
            # go 0-1 always.
            ubound = np.min([np.around(transition_matrix.max(), decimals=decimal) + 10 ** (-decimal), 1])
            lbound = np.max([np.around(transition_matrix.min(), decimals=decimal) - 10 ** (-decimal), 0])
            boundslice = np.linspace(lbound, ubound, 256)
            cbar = plt.colorbar(output_image, ax=subplot, orientation='vertical',
                                boundaries=boundslice,
                                values=boundslice[1:],
                                format='%.{}f'.format(decimal))
            # Update ticks.
            ticks = np.linspace(lbound, ubound, nticks)
            cbar.set_ticks(ticks)

            # Title: Perron eigenvalue, equilibration time and statistical inefficiency.
            perron_eigenvalue = eigenvalues[1]
            title_txt = (phase_name + ' phase\n'
                         'Perron eigenvalue: {}\n'
                         'State equilibration timescale: ~{} iterations\n')
            if perron_eigenvalue >= 1:
                title_txt = title_txt.format('1.0', '$\infty$')
            else:
                equilibration_timescale = 1.0 / (1.0 - perron_eigenvalue)
                title_txt = title_txt.format('{:.5f}', '{:.1f}')
                title_txt = title_txt.format(perron_eigenvalue, equilibration_timescale)
            title_txt += 'Replica state index statistical inefficiency: {:.3f}'.format(statistical_inefficiency)
            subplot.set_title(title_txt, fontsize=20, y=1.05)

            # Display Warning.
            if np.any(transition_matrix >= mixing_warning_threshold):
                subplot.text(
                    0.5, -0.2,
                    ("Warning!\nThere were states that less than {0:.2f}% swaps!\n"
                     "Consider adding more states!".format((1 - mixing_warning_threshold) * 100)),
                    verticalalignment='bottom', horizontalalignment='center',
                    transform=subplot.transAxes,
                    fontsize=20,
                    color='red',
                    bbox={'alpha': 1.0, 'facecolor': 'white', 'lw': 0, 'pad': 0}
                )
        self._mixing_run = True
        return mixing_figure

    def generate_replica_mixing_plot(self, phase_stacked_replica_plots=False):
        """
        Generate the replica trajectory mixing plots. Show the state of each replica as a function of simulation time

        Parameters
        ----------
        phase_stacked_replica_plots : boolean, Default: False
            Determine if the phases should be shown side by side, or one on top of the other. If True, the two phases
            will be shown with phase 1 on top and phase 2 on bottom.

        Returns
        -------
        replica_figure : matplotlib.figure
            Figure showing the replica state trajectories for both phases

        """
        # Create Parent Gridspec
        if phase_stacked_replica_plots:
            plot_grid = gridspec.GridSpec(2, 1)
            plt.rcParams['figure.figsize'] = 20, 8 * 6
        else:
            plot_grid = gridspec.GridSpec(1, 2)
            plt.rcParams['figure.figsize'] = 20, 8 * 3
        replica_figure = plt.figure()
        for i, phase_name in enumerate(self.phase_names):
            # Gather state NK
            reporter = self.analyzers[phase_name].reporter
            state_nk = reporter.read_replica_thermodynamic_states()[:, :]
            N, K = state_nk.shape
            # Create subgrid
            sub_grid = gridspec.GridSpecFromSubplotSpec(K, 1, subplot_spec=plot_grid[i])
            # Loop through all states
            for k in range(K):
                # Add plot
                plot = replica_figure.add_subplot(sub_grid[k])
                # Actually plot
                plot.plot(state_nk[:, k], 'k.')
                # Format plot
                plot.set_yticks([])
                plot.set_xlim([0, N])
                plot.set_ylim([0, K])
                if k < K - 1:
                    plot.set_xticks([])
                plot.set_ylabel('{}'.format(k))
                if k == 0:  # Title
                    plot.set_title('{} phase'.format(phase_name), fontsize=20)
        self._replica_mixing_run = True
        return replica_figure

    def generate_free_energy(self):
        if not self._equilibration_run:
            raise RuntimeError("Cannot run free energy without first running the equilibration. Please run the "
                               "corresponding function/cell first!")
        data = dict()
        for phase_name in self.phase_names:
            analyzer = self.analyzers[phase_name]
            data[phase_name] = analyzer.analyze_phase()
            kT = analyzer.kT

        # Compute free energy and enthalpy
        DeltaF = 0.0
        dDeltaF = 0.0
        DeltaH = 0.0
        dDeltaH = 0.0
        for phase_name in self.phase_names:
            sign = self.signs[phase_name]
            DeltaF -= sign * (data[phase_name]['DeltaF'] + data[phase_name]['DeltaF_standard_state_correction'])
            dDeltaF += data[phase_name]['dDeltaF'] ** 2
            DeltaH -= sign * (data[phase_name]['DeltaH'] + data[phase_name]['DeltaF_standard_state_correction'])
            dDeltaH += data[phase_name]['dDeltaH'] ** 2
        dDeltaF = np.sqrt(dDeltaF)
        dDeltaH = np.sqrt(dDeltaH)

        # Attempt to guess type of calculation
        calculation_type = ''
        for phase in self.phase_names:
            if 'complex' in phase:
                calculation_type = ' of binding'
            elif 'solvent1' in phase:
                calculation_type = ' of solvation'

        print('Free energy{:<13}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
            calculation_type, DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole,
                                               dDeltaF * kT / units.kilocalories_per_mole))

        for phase in self.phase_names:
            print('DeltaG {:<17}: {:9.3f} +- {:.3f} kT'.format(phase, data[phase]['DeltaF'],
                                                               data[phase]['dDeltaF']))
            if data[phase]['DeltaF_standard_state_correction'] != 0.0:
                print('DeltaG {:<17}: {:18.3f} kT'.format('standard state correction',
                                                          data[phase]['DeltaF_standard_state_correction']))
        print('')
        print('Enthalpy{:<16}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
            calculation_type, DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole,
                                               dDeltaH * kT / units.kilocalories_per_mole))
        self._free_energy_run = True

    @staticmethod
    def report_version():
        print("Rendered with YANK Version {}".format(version.version))

