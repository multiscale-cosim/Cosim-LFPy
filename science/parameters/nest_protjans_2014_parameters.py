<?xml version="1.0" encoding="UTF-8" ?>
<parameters>
        <title>Protjans 2014 Parameters for NEST</title>
        <description>Scientific Parameters for the NEST Protjans 2014 Simulation Model</description>
        <!-- NOTE below is given as a sample -->
        <!-- TODO fill with parameters values-->
        <nb_neurons datatype="int">100</nb_neurons>
        <resolution datatype="float">0.1</resolution>
        <nodes_model model="iaf_psc_alpha">
            <!-- https://nest-simulator.readthedocs.io/en/v3.3/models/iaf_psc_alpha.html -->

            <total_inhibitory_nodes datatype="int">25</total_inhibitory_nodes>
            <params datatype="dictionary">
                <V_m datatype="float">0.0</V_m>
                <E_L datatype="float">0.0</E_L>
                <C_m datatype="float">250.0</C_m>
                <tau_m datatype="float">20.0</tau_m>
                <t_ref datatype="float">2.0</t_ref>
                <V_th datatype="float">20.0</V_th>
                <V_reset datatype="float">0.0</V_reset>
                <tau_syn_ex datatype="float">0.5</tau_syn_ex>
                <tau_syn_in datatype="float">0.5</tau_syn_in>
                <I_e datatype="float">0.0</I_e>
                <V_min datatype="float">0.0</V_min>
            </params>
        </nodes_model>
        <noise_model model="poisson_generator">
            <!-- https://nest-simulator.readthedocs.io/en/v3.3/models/poisson_generator.html -->
            <params datatype="dictionary">
                <rate datatype="float">8894.503857360944</rate>
            </params>
        </noise_model>
        <spike_recorder_device datatype="string">spike_recorder</spike_recorder_device>
        <excitatory_spikes_model model="brunel-py-ex">
            <record_to datatype="string">ascii</record_to>
        </excitatory_spikes_model>
        <inhibitory_spikes_model model="brunel-py-in">
            <record_to datatype="string">ascii</record_to>
        </inhibitory_spikes_model>
        <predefined_synapse datatype="string">static_synapse</predefined_synapse>
        <excitatory_model model="excitatory">
            <synapse datatype="string">excitatory</synapse>
            <params datatype="dictionary">
                <weight datatype="float">20.68015524367846</weight>
                <delay datatype="float">1.5</delay>
            </params>
        </excitatory_model>
        <inhibitory_model model="inhibitory">
            <synapse datatype="string">inhibitory</synapse>
            <params datatype="dictionary">
                <weight datatype="float">-103.4007762183923</weight>
                <delay datatype="float">1.5</delay>
            </params>
        </inhibitory_model>
        <excitatory_connection model="excitatory_connection">
            <params datatype="dictionary">
                <rule datatype="string">fixed_indegree</rule>
                <indegree datatype="int">10</indegree>
            </params>
            <syn_spec datatype="string">excitatory</syn_spec>
        </excitatory_connection>
        <inhibitory_connection model="inhibitory_connection">
            <params datatype="dictionary">
                <rule datatype="string">fixed_indegree</rule>
                <indegree datatype="int">2</indegree>
            </params>
            <syn_spec datatype="string">inhibitory</syn_spec>
        </inhibitory_connection>
        <!-- Co-Simulation's Devices -->
        <input_to_simulator model="spike_generator">
            <params datatype="dict">
                <stimulus_source datatype="str">mpi</stimulus_source>
                <label datatype="str">/../transformation/spike_generator</label>
            </params>
            <conn_spec datatype="dict">
                <rule datatype="str">one_to_one</rule>
            </conn_spec>
            <syn_spec datatype="dict">
                <weight datatype="float">20.68015524367846</weight>
                <delay datatype="float">0.1</delay>
            </syn_spec>
        </input_to_simulator>
        <output_from_simulator model="spike_recorder">
            <params datatype="dict">
                <record_to datatype="str">mpi</record_to>
                <label datatype="str">/../transformation/spike_detector</label>
            </params>
            <conn_spec datatype="dict">
                <rule datatype="str">all_to_all</rule>
            </conn_spec>
            <syn_spec datatype="dict">
                <weight datatype="float">1.0</weight>
                <delay datatype="float">0.1</delay>
            </syn_spec>
        </output_from_simulator>

</parameters>