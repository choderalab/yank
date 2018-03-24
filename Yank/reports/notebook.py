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
from pymbar import MBAR

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
        self._serialized_data = {}

    def general_simulation_data(self):
        """
        General purpose simulation data on number of iterations, number of states, and number of atoms.
        This just prints out this data in a regular, formatted pattern.
        """
        iterations = {}
        nstates = {}
        natoms = {}
        nreplicas = {}
        for phase_name in self.phase_names:
            if phase_name not in self._serialized_data:
                self._serialized_data[phase_name] = {}
            self._serialized_data[phase_name]['general'] = {}
            analyzer = self.analyzers[phase_name]
            try:
                positions = analyzer.reporter.read_sampler_states(0)[0].positions
                natoms[phase_name], _ = positions.shape
            except AttributeError:  # Trap unloaded checkpoint file
                natoms[phase_name] = 'No Cpt.'
            energies, _, _, = analyzer.reporter.read_energies()
            iterations[phase_name], nreplicas[phase_name], nstates[phase_name] = energies.shape

        leniter = max(len('Iterations'), *[len(str(i)) for i in iterations.values()]) + 2
        lenreplica = max(len('Replicas'), *[len(str(i)) for i in nreplicas.values()]) + 2
        lenstates = max(len('States'), *[len(str(i)) for i in nstates.values()]) + 2
        lennatoms = max(len('Num Atoms'), *[len(str(i)) for i in natoms.values()]) + 2
        lenleftcol = max(len('Phase'), *[len(phase) for phase in self.phase_names]) + 2

        lines = []
        headstring = ''
        headstring += ('{:^' + '{}'.format(lenleftcol) + '}').format('Phase') + '|'
        headstring += ('{:^' + '{}'.format(leniter) + '}').format('Iterations') + '|'
        headstring += ('{:^' + '{}'.format(lenreplica) + '}').format('Replicas') + '|'
        headstring += ('{:^' + '{}'.format(lenstates) + '}').format('States') + '|'
        headstring += ('{:^' + '{}'.format(lennatoms) + '}').format('Num Atoms')
        lines.append(headstring)
        lenline = len(headstring)
        topdiv = '=' * lenline
        lines.append(topdiv)
        for phase in self.phase_names:
            phasestring = ''
            serial = self._serialized_data[phase]['general']
            phasestring += ('{:^' + '{}'.format(lenleftcol) + '}').format(phase) + '|'
            phase_iter = iterations[phase]
            serial['iterations'] = phase_iter
            phasestring += ('{:^' + '{}'.format(leniter) + '}').format(iterations[phase]) + '|'
            phase_states = nstates[phase]
            serial['states'] = phase_states
            phasestring += ('{:^' + '{}'.format(lenreplica) + '}').format(nreplicas[phase]) + '|'
            phase_atoms = natoms[phase]
            serial['natoms'] = phase_atoms
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
            if phase_name not in self._serialized_data:
                self._serialized_data[phase_name] = {}
            self._serialized_data[phase_name]['equilibration'] = {}
            serial = self._serialized_data[phase_name]['equilibration']
            sub_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=plot_grid[i])
            analyzer = self.analyzers[phase_name]
            # Data crunching to get timeseries
            # TODO: Figure out how not to discard the first sample
            # Sample at index 0 is actually the minimized structure and NOT from the equilibrium distribution
            # This throws off all of the equilibrium data
            self._n_discarded = discard_from_start
            self.u_ns[phase_name] = analyzer.get_effective_energy_timeseries()[discard_from_start:]
            # Timeseries statistics
            g_t, Neff_t = analyze.multistate.get_equilibration_data_per_sample(self.u_ns[phase_name])
            self.Neff_maxs[phase_name] = Neff_t.max()
            self.nequils[phase_name] = Neff_t.argmax()
            self.g_ts[phase_name] = g_t[int(self.nequils[phase_name])]
            serial['discarded_from_start'] = int(discard_from_start)
            serial['effective_samples'] = float(self.Neff_maxs[phase_name])
            serial['equilibration_samples'] = int(self.nequils[phase_name])
            serial['subsample_rate'] = float(self.g_ts[phase_name])

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

    def compute_rmsds(self):
        return NotImplementedError("This function is still a prototype and has segfault issues, please disable for now")
        # """Compute the RMSD of the ligand and the receptor by state"""
        # if not self._equilibration_run:
        #     raise RuntimeError("Cannot run RMSD without first running the equilibration. Please run the "
        #                        "corresponding function/cell first!")
        # plt.rcParams['figure.figsize'] = 20, 6 * self.nphases * 2
        # rmsd_figure, subplots = plt.subplots(2, 1)
        # for i, phase_name in enumerate(self.phase_names):
        #     if phase_name not in self._serialized_data:
        #         self._serialized_data[phase_name] = {}
        #     self._serialized_data[phase_name]['rmsd'] = {}
        #     serial = self._serialized_data[phase_name]['rmsd']
        #     analyzer = self.analyzers[phase_name]
        #     reporter = analyzer.reporter
        #     metadata = reporter.read_dict('metadata')
        #     topography = mmtools.utils.deserialize(metadata['topography'])
        #     topology = topography.topology
        #     test_positions = reporter.read_sampler_states(0, analysis_particles_only=True)[0]
        #     atoms_analysis = test_positions.positions.shape[0]
        #     topology = topology.subset(range(atoms_analysis))
        #     iterations = self.iterations[phase_name]
        #     positions = np.zeros([iterations, atoms_analysis, 3])
        #     for j in range(iterations):
        #         sampler_states = reporter.read_sampler_states(j, analysis_particles_only=True)
        #         # Deconvolute
        #         thermo_states = reporter.read_replica_thermodynamic_states(iteration=j)
        #         sampler = sampler_states[thermo_states[0]]
        #         positions[j, :, :] = sampler.positions
        #     trajectory = md.Trajectory(positions, topology)
        #     rmsd_ligand = md.rmsd(trajectory, trajectory, frame=0, atom_indices=topography.ligand_atoms)
        #     rmsd_recpetor = md.rmsd(trajectory, trajectory, frame=0, atom_indices=topography.receptor_atoms)
        #     serial['ligand'] = rmsd_ligand.tolist()
        #     serial['receptor'] = rmsd_recpetor.tolist()
        #     p = subplots[i]
        #     x = range(iterations)
        #     p.set_title(phase_name + " phase", fontsize=20)
        #     p.plot(x, rmsd_ligand, label='Ligand RMSD')
        #     p.plot(x, rmsd_recpetor, label='Receptor RMSD')
        #     p.legend()
        #     p.set_xlim([0, iterations])
        #     ylim = p.get_ylim()
        #     p.set_ylim([0, ylim[-1]])
        #     p.set_ylabel(r'RMSD (nm)', fontsize=20)
        #     p.set_xlabel(r'Iteration', fontsize=20)
        # return rmsd_figure

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
            serial = self._serialized_data[phase_name]['equilibration']  # This will exist because of _equilibration_run
            # Create subplot
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
            serial['count_total_equilibration_samples'] = int(eq)
            serial['count_decorrelated_samples'] = int(decor)
            serial['count_correlated_samples'] = int(cor)
            serial['percent_total_equilibration_samples'] = float(dat[2])
            serial['percent_decorrelated_samples'] = float(dat[0])
            serial['percent_correlated_samples'] = float(dat[1])
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
            if phase_name not in self._serialized_data:
                self._serialized_data[phase_name] = {}
            self._serialized_data[phase_name]['mixing'] = {}
            serial = self._serialized_data[phase_name]['mixing']
            # Generate mixing statistics.
            analyzer = self.analyzers[phase_name]
            mixing_statistics = analyzer.generate_mixing_statistics(
                number_equilibrated=self.nequils[phase_name])
            transition_matrix, eigenvalues, statistical_inefficiency = mixing_statistics
            serial['transitions'] = transition_matrix.tolist()
            serial['eigenvalues'] = eigenvalues.tolist()
            serial['stat_inefficiency'] = float(statistical_inefficiency)

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

    def free_energy_trace(self, discard_from_start=1, n_trace=10):
        """
        Trace the free energy by keeping fewer and fewer samples in both forward and reverse direction

        Returns
        -------
        free_energy_trace_figure : matplotlib.figure
            Figure showing the equilibration between both phases

        """

        trace_spacing = 1.0/n_trace

        def format_trace_plot(plot: plt.Axes, trace_forward: np.ndarray, trace_reverse: np.ndarray):
            x = np.arange(n_trace + 1)[1:] * trace_spacing * 100
            plot.errorbar(x, trace_forward[:, 0], yerr=2 * trace_forward[:, 1], ecolor='b',
                          elinewidth=0, mec='none', mew=0, linestyle='None',
                          zorder=10)
            plot.plot(x, trace_forward[:, 0], 'b-', marker='o', mec='b', mfc='w', label='Forward', zorder=20,)
            plot.errorbar(x, trace_reverse[:, 0], yerr=2 * trace_reverse[:, 1], ecolor='r',
                          elinewidth=0, mec='none', mew=0, linestyle='None',
                          zorder=10)
            plot.plot(x, trace_reverse[:, 0], 'r-', marker='o', mec='r', mfc='w', label='Reverse', zorder=20)
            y_fill_upper = [trace_forward[-1, 0] + 2 * trace_forward[-1, 1]] * 2
            y_fill_lower = [trace_forward[-1, 0] - 2 * trace_forward[-1, 1]] * 2
            xlim = [0, 100]
            plot.fill_between(xlim, y_fill_lower, y_fill_upper, color='orchid', zorder=5)
            plot.set_xlim(xlim)
            plot.legend()
            plot.set_xlabel("% Samples Analyzed", fontsize=20)
            plot.set_ylabel(r"$\Delta G$ in kcal/mol", fontsize=20)
        # Adjust figure size
        plt.rcParams['figure.figsize'] = 15, 6 * (self.nphases + 1) * 2
        plot_grid = gridspec.GridSpec(self.nphases + 1, 1)  # Vertical distribution
        free_energy_trace_figure = plt.figure()
        # Add some space between the figures
        free_energy_trace_figure.subplots_adjust(hspace=0.4)
        traces = {}
        for i, phase_name in enumerate(self.phase_names):
            traces[phase_name] = {}
            if phase_name not in self._serialized_data:
                self._serialized_data[phase_name] = {}
            serial = self._serialized_data[phase_name]
            if "free_energy" not in serial:
                serial["free_energy"] = {}
            serial = serial["free_energy"]
            free_energy_trace_f = np.zeros([n_trace, 2], dtype=float)
            free_energy_trace_r = np.zeros([n_trace, 2], dtype=float)
            p = free_energy_trace_figure.add_subplot(plot_grid[i])
            analyzer = self.analyzers[phase_name]
            kcal = analyzer.kT / units.kilocalorie_per_mole
            # Data crunching to get timeseries
            sampled_energies, _, _, states = analyzer.read_energies()
            n_replica, n_states, _ = sampled_energies.shape
            # Sample at index 0 is actually the minimized structure and NOT from the equilibrium distribution
            # This throws off all of the equilibrium data
            sampled_energies = sampled_energies[:, :, discard_from_start:]
            states = states[:, discard_from_start:]
            total_iterations = sampled_energies.shape[-1]
            for trace_factor in range(n_trace, 0, -1):  # Reverse order tracing
                trace_percent = trace_spacing*trace_factor
                j = trace_factor - 1  # Indexing
                kept_iterations = int(np.ceil(trace_percent*total_iterations))
                u_forward = sampled_energies[:, :, :kept_iterations]
                s_forward = states[:, :kept_iterations]
                u_reverse = sampled_energies[:, :, -1:-kept_iterations-1:-1]
                s_reverse = states[:, -1:-kept_iterations - 1:-1]
                for energy_sub, state_sub, storage in [
                        (u_forward, s_forward, free_energy_trace_f), (u_reverse, s_reverse, free_energy_trace_r)]:
                    u_n = analyzer.get_effective_energy_timeseries(energies=energy_sub, states=state_sub)
                    number_equilibrated, g_t, neff_max = analyze.multistate.utils.get_equilibration_data(u_n)
                    energy_sub = analyze.multistate.utils.remove_unequilibrated_data(energy_sub,
                                                                                     number_equilibrated,
                                                                                     -1)
                    state_sub = analyze.multistate.utils.remove_unequilibrated_data(state_sub, number_equilibrated, -1)
                    energy_sub = analyze.multistate.utils.subsample_data_along_axis(energy_sub, g_t, -1)
                    state_sub = analyze.multistate.utils.subsample_data_along_axis(state_sub, g_t, -1)
                    samples_per_state = np.zeros([n_states], dtype=int)
                    unique_sampled_states, counts = np.unique(state_sub, return_counts=True)
                    # Assign those counts to the correct range of states
                    samples_per_state[unique_sampled_states] = counts
                    mbar = MBAR(energy_sub, samples_per_state)
                    fe_data = mbar.getFreeEnergyDifferences(compute_uncertainty=True)
                    # Trap theta_ij output
                    try:
                        fe, dfe, _ = fe_data
                    except ValueError:
                        fe, dfe = fe_data
                    ref_i, ref_j = analyzer.reference_states
                    storage[j, :] = fe[ref_i, ref_j] * kcal, dfe[ref_i, ref_j] * kcal
            format_trace_plot(p, free_energy_trace_f, free_energy_trace_r)
            p.set_title("{} Phase".format(phase_name.title()), fontsize=20)
            traces[phase_name]['f'] = free_energy_trace_f
            traces[phase_name]['r'] = free_energy_trace_r
            serial['forward'] = free_energy_trace_f.tolist()
            serial['reverse'] = free_energy_trace_r.tolist()
        # Finally handle last combined plot
        combined_trace_f = np.zeros([n_trace, 2], dtype=float)
        combined_trace_r = np.zeros([n_trace, 2], dtype=float)
        for phase_name in self.phase_names:
            phase_f = traces[phase_name]['f']
            phase_r = traces[phase_name]['r']
            combined_trace_f[:, 0] += phase_f[:, 0]
            combined_trace_f[:, 1] = np.sqrt(combined_trace_f[:, 1]**2 + phase_f[:, 1]**2)
            combined_trace_r[:, 0] += phase_r[:, 0]
            combined_trace_r[:, 1] = np.sqrt(combined_trace_r[:, 1] ** 2 + phase_r[:, 1] ** 2)
        p = free_energy_trace_figure.add_subplot(plot_grid[-1])
        format_trace_plot(p, combined_trace_f, combined_trace_r)
        p.set_title("Combined Phases", fontsize=20)

        return free_energy_trace_figure

    def report_version(self):
        current_version = version.version
        self._serialized_data['yank_version'] = current_version
        print("Rendered with YANK Version {}".format(current_version))

    def dump_serial_data(self, path):
        """Dump the serialized data to YAML file"""
        true_path, ext = os.path.splitext(path)
        if not ext:  # empty string check
            ext = '.yaml'
        true_path += ext
        with open(true_path, 'w') as f:
            f.write(yaml.dump(self._serialized_data))


    @staticmethod
    def report_version():
        print("Rendered with YANK Version {}".format(version.version))

