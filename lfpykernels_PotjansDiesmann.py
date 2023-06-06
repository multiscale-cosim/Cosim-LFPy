import os
import json
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import scipy.stats as st

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# (NEURON MUST BE AFTER MPI SOMETIMES)
import neuron
# from torbjone/LFPykernels:
from lfpykernels import KernelApprox, GaussCylinderPotential

mod_folder = os.path.join('science', 'mod')
mech_loaded = neuron.load_mechanisms(mod_folder)
if not mech_loaded:
    os.system(f'cd {mod_folder} && nrnivmodl && cd -')
    mech_loaded = neuron.load_mechanisms(mod_folder)
assert mech_loaded


binzegger_file = os.path.join('science', 'parameters',
                              'binzegger_connectivity_table.json')
morphology_folder = os.path.join('science', 'models',
                                 'morphologies', 'stretched')
template_folder = os.path.join('science', 'models', 'morphologies')

class PotjansDiesmannKernels:
    """
    Convention:
        X: Presynaptic population (e.g. L4E)
        x: Presynaptic subpopulation (e.g. p4)
        Y: Postsynaptic population
        y: Postsynaptic subpopulation

    """

    def __init__(self, sim_dict, net_dict, stim_dict,
                 network_model_folder, calculate_kernels=True):

        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.network_model_folder = network_model_folder
        self.neuron_params = net_dict['neuron_params']
        self.dt = sim_dict['sim_resolution']

        # Extract some useful neuron parameters:
        self.tau_syn = self.neuron_params['tau_syn']
        self.tau_m = self.neuron_params['tau_m']
        self.C_m = self.neuron_params['C_m']
        self.E_L = self.neuron_params['E_L']

        # Convert postsynaptic potential into postsynaptic current
        # function "postsynaptic_potential_to_current", Potjans2014/helpers.py
        sub = 1. / (self.tau_syn - self.tau_m)
        pre = self.tau_m * self.tau_syn / self.C_m * sub
        frac = (self.tau_m / self.tau_syn) ** sub
        self.PSC_over_PSP = 1. / (pre * (frac ** self.tau_m -
                                         frac ** self.tau_syn)) * 1e-3  # nA

        self.sim_saveforlder = 'sim_results'
        os.makedirs(self.sim_saveforlder, exist_ok=True)

        self.fig_folder = 'figures'
        if stim_dict['thalamic_input']:
            self.fig_folder += '_with_thalamic'
        os.makedirs(self.fig_folder, exist_ok=True)

        self.calculate_kernels = calculate_kernels
        self.plot_conn_data = True
        self.plot_kernels = True
        self.plot_firing_rate = False

        with open(binzegger_file) as f:
            conn_dict = json.load(f)
        self.conn_data = conn_dict['data']

        self._prepare_populations()
        self._find_layer_specific_pathways()
        self._set_extracellular_elec_params()
        self._set_kernel_params()
        if self.calculate_kernels:
            self._calculate_all_pathway_kernels()
            # The calculation of pathway kernels must be completed on all
            # ranks before we continue
            comm.Barrier()
        self._load_pathway_kernels()
        self._find_kernels()

        # self._load_firing_rates_from_file()
        # self.plot_lfps()

    def _set_kernel_params(self):
        # Ignore first 200 ms of simulation in kernel prediction:
        self.TRANSIENT = 200
        self.t_X = self.TRANSIENT
        self.tau = 50  # time lag relative to spike for kernel predictions
        self.kernel_length = int(self.tau / self.dt)
        self.g_eff = False

    def _set_extracellular_elec_params(self):
        self.num_elecs = 16
        # class RecExtElectrode parameters:
        self.elec_params = dict(
            x=np.zeros(self.num_elecs),  # x-coordinates of contacts
            y=np.zeros(self.num_elecs),  # y-coordinates of contacts
            z=np.arange(self.num_elecs) * (-100),  # z-coordinates of contacts
            sigma=0.3,  # extracellular conductivity (S/m)
            method="linesource"  # use line sources
        )

        self.dz = np.abs(self.elec_params['z'][1] -
                         self.elec_params['z'][0])

    def _prepare_populations(self):
        """
        Set up the modelled neural populations, and declare
        their respective morphologies etc.
        """
        pop_names = self.net_dict['populations']
        self.presyn_pops = pop_names + ['TC']
        self.postsyn_pops = pop_names

        self.pop_clrs = {pop_name: plt.cm.rainbow(i / (len(self.presyn_pops) - 1))
                         for i, pop_name in enumerate(self.presyn_pops)}

        if self.net_dict['N_scaling'] != 1 and rank == 0:
            print("Scaling population sizes by factor {:1.2f}".format(
                self.net_dict['N_scaling']))

        pop_sizes = np.array(self.net_dict['full_num_neurons']) * \
            self.net_dict['N_scaling']
        self.pop_sizes = np.r_[pop_sizes,
                               [self.stim_dict["num_th_neurons"]]]

        self.layers = ["1", "23", "4", "5", "6"]
        self.layer_boundaries = {
            "1": [0.0, -81.6],
            "23": [-81.6, -587.1],
            "4": [-587.1, -922.2],
            "5": [-922.2, -1170.0],
            "6": [-1170.0, -1491.7]
            }

        self.layer_mids = [np.mean(self.layer_boundaries[layer])
                           for layer in self.layers]
        self.layer_thicknesses = [self.layer_boundaries[layer][0] -
                                  self.layer_boundaries[layer][1]
                                  for layer in self.layers]

        self.subpop_dict = {
            'L23E': ['p23'],
            'L23I': ['b23', 'nb23'],
            'L4E': ['p4', 'ss4(L23)', 'ss4(L4)'],
            'L4I': ['b4', 'nb4'],
            'L5E': ['p5(L23)', 'p5(L56)'],
            'L5I': ['b5', 'nb5'],
            'L6E': ['p6(L4)', 'p6(L56)'],
            'L6I': ['b6', 'nb6'],
            'TC': ['TCs', 'TCn'],
        }

        self.subpop_mapping_dict = {
            'p23': 'L23E',
            'b23': 'L23I',
            'nb23': 'L23I',
            'p4': 'L4E',
            'ss4(L23)': 'L4E',
            'ss4(L4)': 'L4E',
            'b4': 'L4I',
            'nb4': 'L4I',
            'p5(L23)': 'L5E',
            'p5(L56)': 'L5E',
            'b5': 'L5I',
            'nb5': 'L5I',
            'p6(L4)': 'L6E',
            'p6(L56)': 'L6E',
            'b6': 'L6I',
            'nb6': 'L6I',
            'TCs': 'TC',
            'TCn': 'TC',
        }
        self.subpop_names = self.subpop_mapping_dict.keys()

        self.morph_map = {
            # 'p23': 'L23E_oi24rpy1.hoc',
            # 'b23': 'L23I_oi38lbc1.hoc',
            # 'nb23': 'L23I_oi38lbc1.hoc',
            'L23E': 'L23E_oi24rpy1.hoc',  #
            'L23I': 'L23I_oi38lbc1.hoc',

            # 'p4': 'L4E_53rpy1.hoc',
            # 'ss4(L23)': 'L4E_j7_L4stellate.hoc',
            # 'ss4(L4)': 'L4E_j7_L4stellate.hoc',
            # 'b4': 'L4I_oi26rbc1.hoc',
            # 'nb4': 'L4I_oi26rbc1.hoc',
            'L4E': 'L4E_53rpy1.hoc',
            'L4I': 'L4I_oi26rbc1.hoc',

            # 'p5(L23)': 'L5E_oi15rpy4.hoc',
            # 'p5(L56)': 'L5E_j4a.hoc',
            # 'b5': 'L5I_oi15rbc1.hoc',
            # 'nb5': 'L5I_oi15rbc1.hoc',
            'L5E': 'L5E_j4a.swc',
            'L5I': 'L5I_oi15rbc1.hoc',

            # 'p6(L4)': 'L6E_51-2a.CNG.hoc',
            # 'p6(L56)': 'L6E_oi15rpy4.hoc',
            # 'b6': 'L6I_oi15rbc1.hoc',
            # 'nb6': 'L6I_oi15rbc1.hoc',
            'L6E': 'L6E_oi15rpy4.hoc',
            'L6I': 'L6I_oi15rbc1.hoc'
        }

        self.conn_probs = np.zeros((len(self.postsyn_pops),
                                    len(self.presyn_pops)))
        self.conn_probs[:len(self.postsyn_pops),
                        :len(self.postsyn_pops)] = self.net_dict['conn_probs']
        self.conn_probs[:, -1] = self.stim_dict['conn_probs_th']

    def _find_layer_specific_pathways(self):
        """
        We need to find the normalized layer-specific input for each
        connection pathway (e.g. L4E -> L5E).
        Each population has sub-populations making up given
        fractions of the population.
        Many sup-populations given in the connection
        dictionary are not used.
        For each layer of each post-synaptic subpopulation,
        the data is normalized, but this includes
        input from many unused sub-populations.
        We need to take into account the relative abundance of
        each post-synaptic subpopulation.
        """

        # Prepare layer-specific input distribution array for each
        # post-synaptic subpopulation
        syn_pathways_subpops = {}
        for postsyn_pop in self.postsyn_pops:
            for postsyn_subpop in self.subpop_dict[postsyn_pop]:
                for presyn_pop in self.presyn_pops:
                    subpop_pathway_name = f'{postsyn_subpop}:{presyn_pop}'
                    syn_pathways_subpops[subpop_pathway_name] = np.zeros(5)

        # Find number of inputs to each layer for each
        # post-synaptic subpopulation:
        for postsyn_pop in self.postsyn_pops:
            for postsyn_subpop in self.subpop_dict[postsyn_pop]:
                for l_idx, layer in enumerate(self.layers):
                    if layer in self.conn_data[postsyn_subpop]['syn_dict']:
                        conn_data_yL = self.conn_data[postsyn_subpop]['syn_dict'][layer]
                        sum_ = 0
                        # Total number of synapses to this layer of this subpopulation:
                        k_yL = conn_data_yL['number of synapses per neuron']
                        for p in conn_data_yL:
                            if not p == 'number of synapses per neuron':
                                sum_ += conn_data_yL[p]
                            # Include all considered presynaptic populations:
                            if p in self.subpop_names:
                                p_yxL = conn_data_yL[p]
                                # Number of synapses from presynaptic subpopulation
                                # to this layer of the postsynaptic subpopulation:
                                k_xyL = (p_yxL / 100) * k_yL

                                # Number of inputs from each included presynaptic
                                # population to this layer is summed:
                                presyn_pop = self.subpop_mapping_dict[p]
                                subpop_pathway_name = f'{postsyn_subpop}:{presyn_pop}'
                                syn_pathways_subpops[subpop_pathway_name][l_idx] += k_xyL
                        # Sanity check that sum of all percentage-wise input to this layer of this
                        # subpopulation sums to 100 %:
                        assert np.round(sum_) == 100.0

        # Normalize layer-specific input for each postsynaptic population:
        for subpop_pathway_name in syn_pathways_subpops.keys():
            if np.sum(syn_pathways_subpops[subpop_pathway_name]) > 0.0:
                syn_pathways_subpops[subpop_pathway_name] /= np.sum(
                    syn_pathways_subpops[subpop_pathway_name])

        # Make a dictionary with the relative fraction of
        # each subpopulation within a population:
        subpop_rel_frac = {}
        for pop_name in self.presyn_pops:
            rel_frac = []
            for subpop in self.subpop_dict[pop_name]:
                rel_frac.append(self.conn_data[subpop]['occurrence'])
            subpop_rel_frac[pop_name] = np.array(rel_frac) / np.sum(rel_frac)

        # Add postsynaptic subpopulations weighted by relative occurrence.
        # The resulting dictionary 'syn_pathways' contains the
        # normalized layer-specific synaptic distribution
        # for each synaptic pathway:
        self.syn_pathways = {}
        for postsyn_pop in self.postsyn_pops:
            for presyn_pop in self.presyn_pops:
                pathway_name = f'{postsyn_pop}:{presyn_pop}'
                self.syn_pathways[pathway_name] = np.zeros(5)
                for p_idx_, postsyn_subpop in enumerate(
                        self.subpop_dict[postsyn_pop]):
                    subpop_pathway_name = f'{postsyn_subpop}:{presyn_pop}'
                    rel_frac_ = subpop_rel_frac[postsyn_pop][p_idx_]
                    self.syn_pathways[pathway_name] += syn_pathways_subpops[
                                                                subpop_pathway_name] * rel_frac_

        if self.plot_conn_data:
            # Plot layer-specific connectivity data, similar
            # to Fig. 5D in Hagen et al. (2016)
            fig = plt.figure(figsize=[10, 10])
            fig.subplots_adjust(wspace=0.3, hspace=0.2, bottom=0.05,
                                top=0.95, right=0.98, left=0.05)
            plot_idx = 1
            num_rows = 4
            num_cols = 4
            for postsyn_pop in self.postsyn_pops:
                for postsyn_subpop in self.subpop_dict[postsyn_pop]:
                    conn_matrix = np.zeros((len(self.layers),
                                            len(self.presyn_pops)))
                    presyn_pops_reordered = ["TC"] + self.postsyn_pops
                    for pre_idx, presyn_pop in enumerate(presyn_pops_reordered):
                        subpop_pathway_name = f'{postsyn_subpop}:{presyn_pop}'
                        conn_matrix[:, pre_idx] = syn_pathways_subpops[
                            subpop_pathway_name]

                    ax = fig.add_subplot(num_rows, num_cols, plot_idx,
                                         title=f'$y$={postsyn_subpop}')
                    ax.set_xticks(np.arange(len(presyn_pops_reordered)))
                    ax.set_xticklabels(presyn_pops_reordered, rotation=-90)
                    ax.set_yticks(np.arange(len(self.layers)))
                    ax.set_yticklabels(self.layers)
                    ax.set_xlabel("$X$")
                    ax.set_ylabel("$L$")
                    ax.imshow(conn_matrix, cmap="hot", vmax=1, vmin=0)

                    plot_idx += 1
            plt.savefig(os.path.join(self.fig_folder,
                                     "layer_specific_conn_data.png"))

    def _calculate_one_pathway_kernel(self, postsyn_pop, presyn_pop):

        postsyn_pop_idx = np.where([p_ == postsyn_pop
                                  for p_ in self.postsyn_pops])[0][0]
        presyn_pop_idx = np.where([p_ == presyn_pop
                                  for p_ in self.presyn_pops])[0][0]

        pathway_name = f'{postsyn_pop}:{presyn_pop}'
        postsyn_l_idx = np.where([l_ == postsyn_pop[1:-1]
                                  for l_ in self.layers])[0][0]

        if np.abs(self.conn_probs[postsyn_pop_idx, presyn_pop_idx]) < 1e-9:
            # No pathway from presyn_pop to postsyn_pop
            #self.H[pathway_name] = None
            return

        layered_input = self.syn_pathways[pathway_name]

        if np.sum(layered_input) < 1e-9:
            # If this has happened the connection probability is non-zero,
            # but no connection data was extracted from data_dict.
            # This happens in one case, with a low connection probability.
            # Unclear why this happens, but we just assume the connection is
            # in the layer of the postsynaptic cell.
            # print(f"{presyn_pop} to {postsyn_pop}: {layered_input} " +
            #      "while connection probability is non-zero: " +
            #      f"{self.conn_probs[postsyn_pop_idx, presyn_pop_idx]}")
            layered_input[postsyn_l_idx] = 1.0

        cell_params = dict(
            morphology=os.path.join(morphology_folder,
                                    self.morph_map[postsyn_pop]),
            templatename='LFPyCellTemplate',
            templatefile=os.path.join(template_folder, 'LFPyCellTemplate.hoc'),
            v_init=self.E_L,
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(g_pas=1. / (self.tau_m * 1E3),  # assume cm=1
                                    e_pas=self.E_L),
            nsegs_method='lambda_f',
            lambda_f=100,
            dt=self.dt,
            delete_sections=True,
            templateargs=None,
        )

        population_params = dict(
            radius=np.sqrt(1000 ** 2 / np.pi),  # population radius
            loc=self.layer_mids[postsyn_l_idx],  # population center along z-axis
            scale=self.layer_thicknesses[postsyn_l_idx] / 4)  # SD along z-axis

        rotation_args = {'x': 0.0, 'y': 0.0}
        sections = "allsec" if "I" in presyn_pop else ["dend", "apic"]
        syn_pos_params = [dict(section=sections,
                               fun=[st.norm] * len(self.layers),
                               funargs=[dict(loc=self.layer_mids[l_idx],
                                             scale=self.layer_thicknesses[l_idx] / 4)
                                        for l_idx in range(len(self.layers))],
                               funweights=layered_input
                               )]

        gauss_cyl_potential = GaussCylinderPotential(
            cell=None,
            z=self.elec_params['z'],
            sigma=self.elec_params['sigma'],
            R=population_params['radius'],
            sigma_z=population_params['scale'],
        )

        if presyn_pop == 'TC':
            PSP_mean = self.stim_dict['PSP_th']
            delay_mean = self.stim_dict['delay_th_mean']
            delay_rel_std = self.stim_dict['delay_th_rel_std']
        else:
            PSP_mean = self.net_dict['PSP_matrix_mean'][postsyn_pop_idx, presyn_pop_idx]
            delay_mean = self.net_dict['delay_matrix_mean'][postsyn_pop_idx, presyn_pop_idx]
            delay_rel_std = self.net_dict['delay_rel_std']

        C_YX = self.conn_probs[postsyn_pop_idx].copy()
        if self.net_dict['K_scaling'] != 1:
            # see Potjans2014/helpers.py,
            # function: adjust_weights_and_input_to_synapse_scaling
            # if rank == 0:
            #     print('Synapses are adjusted to compensate scaling of indegrees')
            PSP_mean /= np.sqrt(self.net_dict['K_scaling'])
            C_YX *= self.net_dict['K_scaling']

        weight = PSP_mean * self.PSC_over_PSP
        delay_params = [{'a': (self.dt - delay_mean) / delay_rel_std,
                         'b': np.inf,
                         'loc': delay_mean,
                         'scale': delay_rel_std}]

        syn_params = [dict(weight=weight, syntype='ExpSynI', tau=self.tau_syn)]

        # Create KernelApprox object
        kernel = KernelApprox(
            X=[presyn_pop],
            Y=postsyn_pop,
            N_X=np.array([self.pop_sizes[presyn_pop_idx]]),
            N_Y=self.pop_sizes[postsyn_pop_idx],
            C_YX=C_YX,
            cellParameters=cell_params,
            rotationParameters=rotation_args,
            populationParameters=population_params,
            multapseFunction=st.norm,
            multapseParameters=[dict(loc=1, scale=0.001)],  # Ignores multapses
            delayFunction=st.truncnorm,
            delayParameters=delay_params,
            synapseParameters=syn_params,
            synapsePositionArguments=syn_pos_params,
            extSynapseParameters=None,
            nu_ext=None,
            n_ext=None,
            nu_X=None,
            conductance_based=False,
        )

        # make kernel predictions and update container dictionary
        H_XY = kernel.get_kernel(probes=[gauss_cyl_potential],
                                 Vrest=self.E_L, dt=self.dt, X=presyn_pop,
                                 t_X=self.t_X, tau=self.tau,
                                 g_eff=self.g_eff, fir=True)

        k_ = H_XY['GaussCylinderPotential']
        np.save(os.path.join(self.sim_saveforlder,
                             f'kernel_{pathway_name}.npy'), k_)

        # self.H[pathway_name] = k_

        if self.plot_kernels:
            t_k = np.arange(k_.shape[1]) * self.dt

            cell = kernel.cell

            plt.close("all")
            fig = plt.figure(figsize=[16, 5])
            fig.subplots_adjust(left=0.05, right=0.98, top=0.82, wspace=0.4)
            fig.suptitle(f"LFP kernel for input to {postsyn_pop} from {presyn_pop}")
            ax_m = fig.add_subplot(151, aspect=1, xlim=[-500, 500],
                                   ylim=[-1600, 200],
                                   title="postsynaptic neuron")
            ax_s = fig.add_subplot(152, ylim=[-1600, 200],
                                   title="synaptic input density\nBinzegger data")
            ax_g = fig.add_subplot(153, ylim=[-1600, 200],
                                   title="inferred gaussian input profile")
            ax_w = fig.add_subplot(154, ylim=[-1600, 200],
                                   title="per. comp synaptic weight")
            ax_k = fig.add_subplot(155, ylim=[-1600, 200],
                                   title="LFP kernel")

            [ax_m.axhline(boundary[0], c='gray', ls='--')
             for boundary in self.layer_boundaries.values()]
            ax_m.axhline(self.layer_boundaries["6"][1], c='gray', ls='--')

            ax_m.plot(cell.x.T, cell.z.T, c='k')

            poss_idx = cell.get_idx(section="allsec", z_min=-1e9, z_max=1e9)
            p = np.zeros_like(cell.area)
            p[poss_idx] = cell.area[poss_idx]
            mod = np.zeros(poss_idx.shape)

            xs_ = [0]
            ys_ = [0]
            for l_idx, layer in enumerate(self.layers):
                df = st.norm(loc=self.layer_mids[l_idx],
                             scale=self.layer_thicknesses[l_idx] / 2)
                # Normalize to have same area, regardless of layer thickhness
                mod += df.pdf(x=cell.z[poss_idx].mean(axis=-1)
                              ) * layered_input[l_idx]

                xs_.extend([layered_input[l_idx] /
                            self.layer_thicknesses[l_idx]] * 2)
                ys_.extend(self.layer_boundaries[layer])

            xs_.append(0)
            ys_.append(self.layer_boundaries["6"][1])
            ax_s.plot(xs_, ys_, c=self.pop_clrs[presyn_pop], label=presyn_pop)

            ax_g.plot(mod, cell.z.mean(axis=1), '.',
                      c=self.pop_clrs[presyn_pop])

            ax_w.plot(kernel.comp_weight, cell.z.mean(axis=1), 'k.')

            k_norm = np.max(np.abs(k_))

            for elec_idx in range(self.num_elecs):
                ax_k.plot(t_k, k_[elec_idx] / k_norm * self.dz +
                          self.elec_params["z"][elec_idx],
                          c='k')

            ax_k.plot([30, 30], [-1000, -1000 + self.dz], c='gray', lw=1.5)
            ax_k.text(31, -1000 + self.dz / 2, f"{k_norm * 1000: 1.2f} µV",
                      color="gray")

            fig.legend(frameon=False, ncol=6, loc=(0.3, 0.75))
            plt.savefig(os.path.join(self.fig_folder,
                                     f"fig_pathways_syn_input_"
                                     f"{postsyn_pop}_{presyn_pop}.png"))

    def _calculate_all_pathway_kernels(self):

        task_idx = 0
        # Loop through all synaptic pathways in the model and calculate kernels:
        for postsyn_pop_idx, postsyn_pop in enumerate(self.postsyn_pops):
            for presyn_pop_idx, presyn_pop in enumerate(self.presyn_pops):
                if task_idx % size == rank:
                    print(f"{presyn_pop} to {postsyn_pop} on rank {rank}")
                    self._calculate_one_pathway_kernel(postsyn_pop, presyn_pop)
                task_idx += 1

    def _load_pathway_kernels(self):
        self.H = {}
        for postsyn_pop_idx, postsyn_pop in enumerate(self.postsyn_pops):
            for presyn_pop_idx, presyn_pop in enumerate(self.presyn_pops):
                pathway_name = f'{postsyn_pop}:{presyn_pop}'
                f_name = os.path.join(self.sim_saveforlder,
                                      f'kernel_{pathway_name}.npy')
                if os.path.isfile(f_name):
                    self.H[pathway_name] = np.load(f_name)

    def _find_kernels(self):

        # Summing the kernels of each presynaptic populations, so LFP can be found
        # by convolving pre-synaptic firing rate with this summed kernel
        self.pop_kernels = {}
        for pop_idx, pop_name in enumerate(self.presyn_pops):
            self.pop_kernels[pop_name] = np.zeros((self.num_elecs,
                                                   self.kernel_length))
            for pathway_name in self.H.keys():
                if pathway_name.endswith(pop_name):
                    if self.H[pathway_name] is not None:
                        self.pop_kernels[pop_name] += self.H[pathway_name]

    def _load_firing_rates_from_file(self):
        self.firing_rate_path = os.path.join(self.network_model_folder, 'data')
        self.pop_gids = {}

        sim_start = self.sim_dict['t_presim']
        sim_end = sim_start + self.sim_dict['t_sim'] + self.dt

        self.bins = np.arange(sim_start, sim_end, self.dt)

        with open(os.path.join(self.firing_rate_path,
                               'population_nodeids.dat')) as f:
            self.gid_data = np.array([l_.split() for l_ in f.readlines()],
                                     dtype=int)
            for pop_idx, pop_name in enumerate(self.presyn_pops):
                # TC population is not in gid_data if not modelled
                if pop_idx < self.gid_data.shape[0]:
                    self.pop_gids[pop_name] = self.gid_data[pop_idx]

        pop_spike_times, self.firing_rates = self._load_and_return_spikes()

        if self.plot_firing_rate:
            plt.close('all')
            fig = plt.figure(figsize=[10, 10])
            ax1 = fig.add_subplot(111, xlim=[675, 750])
            fr_norm = 40

            for pop_idx, pop_name in enumerate(self.presyn_pops):
                ax1.plot(self.bins[:-1],
                         self.firing_rates[pop_name] / fr_norm + pop_idx,
                         c=self.pop_clrs[pop_name], label=pop_name)
            fig.legend(ncol=8, frameon=False)
            plt.savefig(os.path.join(self.fig_folder, "pop_firing_rates.png"))

    def _return_pop_name_from_gid(self, gid):
        for pop_name in self.presyn_pops:
            if self.pop_gids[pop_name][0] <= gid <= self.pop_gids[pop_name][1]:
                return pop_name

    def _load_and_return_spikes(self):

        firing_rates = {}
        pop_spike_times = {pop_name: [] for pop_name in self.presyn_pops}
        fr_files = [f for f in os.listdir(self.firing_rate_path)
                    if f.startswith('spike_recorder-')]
        for f_ in fr_files:
            with open(os.path.join(self.firing_rate_path, f_)) as f:
                d_ = [d__.split('\t') for d__ in f.readlines()[3:]]

                gids, times = np.array(d_, dtype=float).T
                for pop_idx, pop_name in enumerate(self.presyn_pops):
                    if pop_idx < self.gid_data.shape[0]:
                        p_spikes_mask = (self.pop_gids[pop_name][0] <= gids) & \
                                        (gids <= self.pop_gids[pop_name][1])
                        pop_spike_times[pop_name].extend(times[p_spikes_mask])

        for pop_idx, pop_name in enumerate(self.presyn_pops):
            pop_spike_times[pop_name] = np.sort(pop_spike_times[pop_name])
            fr__, _ = np.histogram(pop_spike_times[pop_name], bins=self.bins)
            firing_rates[pop_name] = fr__

        return pop_spike_times, firing_rates

    def plot_lfps(self, time_array, lfp, firing_rates):

        plt.close('all')
        fig = plt.figure(figsize=[8, 8])

        ax_fr = fig.add_subplot(212, title="firing rates", xlabel="time (ms)",
                                 xlim=[690, 750])

        max_fr = np.max([np.max(np.abs(fr_)) for fr_ in firing_rates.values()])

        for p_idx, pop in enumerate(self.presyn_pops):
            ax_fr.plot(time_array, firing_rates[pop] / max_fr + pop, label=pop)
        ax_fr.legend(frameon=False)

        ax_lfp = fig.add_subplot(212, title="LFP", xlabel="time (ms)",
                                 ylim=[-1600, 200], xlim=[690, 750])

        lfp_norm = np.max(np.abs(lfp))
        for elec_idx in range(self.num_elecs):
            ax_lfp.plot(time_array, lfp[elec_idx] / lfp_norm * self.dz +
                        self.elec_params["z"][elec_idx], c='k')
        ax_lfp.plot([730, 730], [-100, -100 + self.dz], c='gray', lw=1.5)
        ax_lfp.text(732, -100 + self.dz / 2, f"{lfp_norm * 1000: 1.2f} µV",
                    color="gray")
        fig.savefig(os.path.join(self.fig_folder, f"summary_LFP.png"))

    def update_lfp(self, lfp, t_idx, firing_rate):
        """
        Calculate LFP resulting from our kernel, given a
        firing rate at a given time index
        """
        # For every new timestep the predicted LFP signal is made
        # one timestep longer

        # self.lfp.append([0] * self.num_elecs)
        if lfp is None:
            lfp = np.zeros((self.num_elecs, self.kernel_length - 1))

        lfp = np.append(lfp, np.zeros((self.num_elecs, 1)), axis=1)
        # Find the time indexes where the LFP is calculated:
        window_idx0 = t_idx - int(self.kernel_length / 2)
        window_idx1 = t_idx + int(self.kernel_length / 2)
        sig_idx0 = 0 if window_idx0 < 0 else window_idx0
        sig_idx1 = window_idx1
        k_idx0 = -window_idx0 if window_idx0 < 0 else 0

        # This is essentially a manual convolution, one timestep at the time,
        # between the firingrate and the kernel
        for p_idx, pop in enumerate(self.presyn_pops):
            for elec_idx in range(self.num_elecs):
                lfp_ = firing_rate[pop] * self.pop_kernels[pop][elec_idx][k_idx0:]
                lfp[elec_idx, sig_idx0:sig_idx1] += lfp_
        return lfp


if __name__ == '__main__':

    # Get parameters from Potjans & Diesmann simulation
    network_model_folder = "Potjans_2014"

    from science.parameters.Potjans.stimulus_params import stim_dict
    from science.parameters.Potjans.network_params import net_dict
    from science.parameters.Potjans.sim_params import sim_dict

    # This is a necessary prestep, and the first time this is being run
    # calculate_kernels must be 'True'.
    PD_kernels = PotjansDiesmannKernels(sim_dict,
                                        net_dict,
                                        stim_dict,
                                        network_model_folder,
                                        calculate_kernels=True)

    # Example of how this could be used, given a functioning
    # version of self._mediator.spikes_to_rate(count, size_at_index=-2)
    # which is here assumed to return the time point (in ms), and a dictionary
    # of the firing rate of each population at this time point.

    # if rank == 0:
        # firing_rates = {pop: [] for pop in PD_kernels.presyn_pops}
        # time_array = []
        # lfp = np.zeros((PD_kernels.num_elecs, PD_kernels.kernel_length - 1))
        #
        # for t_idx in range(1000):
        #     time, firing_rates_t_idx = self._mediator.spikes_to_rate(count,
        #                                                       size_at_index=-2)
        #     time_array.append(time)
        #     for p_idx, pop in enumerate(PD_kernels.presyn_pops):
        #         firing_rates[pop].append(firing_rates_t_idx[pop])
        #      lfp_array = PD_kernels.update_lfp(t_idx, firing_rates_t_idx)
        #PD_kernels.plot_lfp(time_array, lfp, firing_rates)