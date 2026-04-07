"""In-app parameter sweep configuration, execution, and results visualization."""
import threading
import time
import json
import numpy as np

try:
    import dearpygui.dearpygui as dpg
    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False


class SweepPanel:
    """Manages the sweep UI, background execution, and results display."""

    def __init__(self):
        self.is_running = False
        self.should_cancel = False
        self.results = []
        self.current_run = 0
        self.total_runs = 0
        self._thread = None
        self._sweep_config = None

    def create_ui(self, parent):
        """Create the sweep panel UI elements."""
        if not DPG_AVAILABLE:
            return

        with dpg.collapsing_header(label="Parameter Sweep", parent=parent,
                                    default_open=False, tag="sweep_panel"):
            # Configuration section
            dpg.add_text("Sweep Configuration", color=[100, 200, 255])

            # Experiment selector
            dpg.add_text("Experiment:")
            dpg.add_combo(["associative", "stimulus-response", "frequency-response", "reinforcement"],
                         tag="sweep_experiment", default_value="associative", width=-1)

            # Parameter selector
            dpg.add_text("Parameter to sweep:")
            params = ["stdp_a_plus", "stdp_a_minus", "stdp_tau_plus_ms", "stdp_tau_minus_ms",
                      "propagation_strength", "inhibitory_propagation_strength",
                      "ou_sigma_pA", "ou_tau_ms", "connectivity_k",
                      "hebbian_learning_rate", "reward_learning_rate",
                      "stp_U", "stp_tau_d", "stp_tau_f"]
            dpg.add_combo(params, tag="sweep_param", default_value="stdp_a_plus", width=-1)

            # Value range
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="Start", tag="sweep_start", default_value=0.004, width=100)
                dpg.add_input_float(label="End", tag="sweep_end", default_value=0.024, width=100)
                dpg.add_input_int(label="Steps", tag="sweep_steps", default_value=5, width=80)

            # Network size and trials
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="Neurons", tag="sweep_neurons", default_value=5000, width=100)
                dpg.add_input_int(label="Trials", tag="sweep_trials", default_value=30, width=100)

            dpg.add_separator()

            # Controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Sweep", tag="sweep_run_btn", width=100,
                              callback=self._start_sweep)
                dpg.add_button(label="Cancel", tag="sweep_cancel_btn", width=80,
                              callback=self._cancel_sweep)

            # Progress
            dpg.add_progress_bar(tag="sweep_progress", default_value=0.0, width=-1)
            dpg.add_text("", tag="sweep_status_text", color=[150, 150, 150])

            dpg.add_separator()

            # Results section
            dpg.add_text("Results", color=[100, 200, 255])

            # Results table
            with dpg.table(tag="sweep_results_table", header_row=True,
                           borders_innerH=True, borders_outerH=True,
                           borders_innerV=True, borders_outerV=True,
                           resizable=True):
                dpg.add_table_column(label="Parameter")
                dpg.add_table_column(label="Delta (Hz)")
                dpg.add_table_column(label="t-stat")
                dpg.add_table_column(label="Cohen's d")
                dpg.add_table_column(label="Sig?")

            dpg.add_separator()

            # Results plot
            with dpg.plot(label="Sweep Results", height=200, width=-1, tag="sweep_results_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Parameter Value", tag="sweep_plot_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Effect (Hz)", tag="sweep_plot_y"):
                    dpg.add_line_series([], [], label="Delta", tag="sweep_plot_series")
                    dpg.add_scatter_series([], [], label="Points", tag="sweep_plot_points")

            dpg.add_separator()

            # Export
            with dpg.group(horizontal=True):
                dpg.add_button(label="Export CSV", tag="sweep_export_csv_btn", width=100,
                              callback=self._export_csv)
                dpg.add_button(label="Export JSON", tag="sweep_export_json_btn", width=100,
                              callback=self._export_json)
                dpg.add_button(label="Export Figure", tag="sweep_export_fig_btn", width=100,
                              callback=self._export_figure)

    def _start_sweep(self):
        """Start the sweep in a background thread."""
        if self.is_running:
            return

        # Read config from UI
        experiment = dpg.get_value("sweep_experiment")
        param = dpg.get_value("sweep_param")
        start = dpg.get_value("sweep_start")
        end = dpg.get_value("sweep_end")
        steps = dpg.get_value("sweep_steps")
        neurons = dpg.get_value("sweep_neurons")
        trials = dpg.get_value("sweep_trials")

        values = np.linspace(start, end, steps).tolist()

        self._sweep_config = {
            "experiment": experiment,
            "num_neurons": neurons,
            "num_trials": trials,
            "parameters": {param: values},
            "sweep_mode": "grid",
        }

        self.results = []
        self.is_running = True
        self.should_cancel = False
        self.total_runs = len(values)
        self.current_run = 0

        self._thread = threading.Thread(target=self._run_sweep_thread, daemon=True)
        self._thread.start()

    def _run_sweep_thread(self):
        """Background thread that runs the sweep."""
        try:
            from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
            from sim.enums import NeuronModel
            from experiment import ExperimentEngine, ExperimentPresets
            import cupy as cp

            config = self._sweep_config
            experiment = config["experiment"]
            param_name = list(config["parameters"].keys())[0]
            values = config["parameters"][param_name]

            for i, val in enumerate(values):
                if self.should_cancel:
                    break

                self.current_run = i + 1

                # Create bridge with override
                core_cfg = CoreSimConfig()
                core_cfg.num_neurons = config["num_neurons"]
                core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
                core_cfg.neural_profile_name = "CORTEX_L23_RS_FS"
                core_cfg.dt_ms = 1.0
                core_cfg.enable_hebbian_learning = True
                core_cfg.enable_stdp = True
                core_cfg.enable_short_term_plasticity = True
                core_cfg.enable_homeostasis = True
                core_cfg.enable_reward_modulation = True
                core_cfg.stdp_a_plus = 0.012
                core_cfg.stdp_a_minus = 0.01
                core_cfg.reward_learning_rate = 0.05

                # Apply sweep parameter
                if hasattr(core_cfg, param_name):
                    setattr(core_cfg, param_name, val)

                sb = SimulationBridge(
                    core_config=core_cfg,
                    viz_config=VisualizationConfig(),
                    runtime_state=RuntimeState(),
                    gpu_config=GPUConfig(),
                )
                sb.runtime_state.max_delay_steps = int(core_cfg.max_synaptic_delay_ms / core_cfg.dt_ms)
                sb._initialize_simulation_data(called_from_playback_init=False)

                if not sb.is_initialized:
                    self.results.append({"param_value": val, "error": "init failed"})
                    continue

                # Load experiment
                presets = {
                    "associative": lambda: ExperimentPresets.associative_conditioning(
                        num_trials=config["num_trials"]),
                    "stimulus-response": lambda: ExperimentPresets.basic_stimulus_response(),
                    "frequency-response": lambda: ExperimentPresets.frequency_response_characterization(
                        num_frequencies=12, amplitude_pA=300.0),
                    "reinforcement": lambda: ExperimentPresets.reinforcement_learning(
                        num_trials=config["num_trials"]),
                }

                exp_config = presets[experiment]()
                engine = ExperimentEngine(core_cfg.num_neurons, core_cfg.dt_ms)
                engine.load_experiment(exp_config)
                engine.initialize(cp_traits=sb.cp_traits, cp_module=cp)
                engine.ensure_inter_group_connectivity(sb, cp)
                sb.experiment_engine = engine

                total_exp_ms = sum(p.duration_ms * p.num_repetitions for p in exp_config.phases)
                total_steps = int(total_exp_ms / core_cfg.dt_ms) + 2000

                engine.start(current_time_ms=0.0, sim_bridge_ref=sb)

                for step in range(total_steps):
                    if self.should_cancel:
                        break
                    sb._run_one_simulation_step()
                    sb.runtime_state.current_time_step += 1
                    sb.runtime_state.current_time_ms = sb.runtime_state.current_time_step * core_cfg.dt_ms
                    if engine.is_experiment_complete:
                        break

                # Extract metrics
                result = {"param_value": val, "param_name": param_name}

                if experiment == "associative":
                    pre_on, post_on = [], []
                    for entry in engine.log:
                        if entry.get("event") == "readout":
                            cs = entry.get("rates", {}).get("cs_input", 0)
                            us = entry.get("rates", {}).get("us_output", 0)
                            phase = entry.get("phase", "")
                            if cs > 20:
                                if phase == "pre_test":
                                    pre_on.append(us)
                                elif phase == "post_test":
                                    post_on.append(us)

                    pre_a = np.array(pre_on) if pre_on else np.array([0.0])
                    post_a = np.array(post_on) if post_on else np.array([0.0])
                    delta = float(post_a.mean() - pre_a.mean())
                    se = np.sqrt(pre_a.var() / max(len(pre_a), 1) + post_a.var() / max(len(post_a), 1))
                    t_stat = delta / se if se > 0 else 0
                    pooled_var = (pre_a.var() + post_a.var()) / 2
                    cohens_d = delta / np.sqrt(pooled_var) if pooled_var > 0 else 0
                    result.update({
                        "delta_hz": round(delta, 3),
                        "t_statistic": round(float(t_stat), 3),
                        "cohens_d": round(float(cohens_d), 3),
                        "p_significant": abs(t_stat) > 2.0,
                    })
                else:
                    # Generic: just get overall rates
                    result["delta_hz"] = 0
                    result["t_statistic"] = 0
                    result["cohens_d"] = 0
                    result["p_significant"] = False

                self.results.append(result)

                sb.clear_simulation_state_and_gpu_memory()
                cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            self.results.append({"error": str(e)})
        finally:
            self.is_running = False

    def _cancel_sweep(self):
        self.should_cancel = True

    def update_ui(self):
        """Called each frame to update progress and results."""
        if not DPG_AVAILABLE:
            return

        try:
            if self.is_running:
                progress = self.current_run / max(self.total_runs, 1)
                dpg.set_value("sweep_progress", progress)
                param_name = list(self._sweep_config["parameters"].keys())[0] if self._sweep_config else ""
                vals = self._sweep_config["parameters"].get(param_name, []) if self._sweep_config else []
                current_val = vals[self.current_run - 1] if 0 < self.current_run <= len(vals) else "?"
                dpg.set_value("sweep_status_text",
                             f"Run {self.current_run}/{self.total_runs}: {param_name}={current_val}")
            elif self.results:
                dpg.set_value("sweep_progress", 1.0)
                dpg.set_value("sweep_status_text",
                             f"Complete: {len(self.results)} runs")
                self._update_results_display()
        except Exception:
            pass

    def _update_results_display(self):
        """Update results table and plot."""
        # Clear table rows
        for child in dpg.get_item_children("sweep_results_table", 1) or []:
            dpg.delete_item(child)

        param_values = []
        deltas = []

        for r in self.results:
            if "error" in r:
                continue

            with dpg.table_row(parent="sweep_results_table"):
                dpg.add_text(f"{r.get('param_value', '?')}")
                dpg.add_text(f"{r.get('delta_hz', 0):+.2f}")
                dpg.add_text(f"{r.get('t_statistic', 0):.2f}")
                dpg.add_text(f"{r.get('cohens_d', 0):.2f}")
                sig = "YES" if r.get("p_significant") else "no"
                dpg.add_text(sig)

            param_values.append(r.get("param_value", 0))
            deltas.append(r.get("delta_hz", 0))

        # Update plot
        if param_values:
            dpg.set_value("sweep_plot_series", [param_values, deltas])
            dpg.set_value("sweep_plot_points", [param_values, deltas])
            dpg.fit_axis_data("sweep_plot_x")
            dpg.fit_axis_data("sweep_plot_y")

    def _export_csv(self):
        """Export results to CSV."""
        if not self.results:
            return
        import csv
        filepath = f"sweep_results_{int(time.time())}.csv"
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

    def _export_json(self):
        """Export results to JSON."""
        if not self.results:
            return
        filepath = f"sweep_results_{int(time.time())}.json"
        with open(filepath, 'w') as f:
            json.dump({"config": self._sweep_config, "results": self.results}, f, indent=2, default=str)

    def _export_figure(self):
        """Export results as a publication-quality matplotlib figure."""
        if not self.results:
            return
        from ui.figure_export import export_sweep_figure
        param_name = list(self._sweep_config["parameters"].keys())[0] if self._sweep_config else "param"
        export_sweep_figure(self.results, param_name)
