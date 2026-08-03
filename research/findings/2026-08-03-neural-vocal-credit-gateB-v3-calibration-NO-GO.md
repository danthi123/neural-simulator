---
type: finding
status: negative
date: 2026-08-03
mechanism: neural-vocal-action-credit-v3
runner: research/runners/_vocal_action_credit_gate_v3.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v3/calibration.json
  - research/findings/raw/vocal_action_credit_gate_v3/calibration.json.prov.json
---

# Omission circuit remains silent under intact critic normalization

<!--derived-->
**Verdict: NO-GO at Gate B v3 calibration.** Both preregistered calibration
seeds were valid scientific failures. Each learned the contingently rewarded
action on every frozen evaluation trial, but reward-count-matched yoked reward
also drove that action on every trial. The added expected-omission circuit did
not produce LHb or RMTg firing with intact critic normalization. Development
and held-out seeds remain locked.

## Mechanism tested

V3 retained the v2 actor, action-value critic, and value-to-SNc GABA-B operating
point. It added local fast-spiking (FS) normalization around each critic and a
spiking expected-omission path through omission-gate-, LHb-, and RMTg-like
populations. A shared reward-driven inhibitory population was intended to veto
the negative path on rewarded outcomes.

The fixed protocol ran contingent, reward-count-matched yoked,
executed-collateral-lesion, reward-to-SNc-lesion, critic-output-lesion,
omission-path-lesion, and local-normalization-lesion arms. Every arm used 20
frozen baseline trials, 40 training trials, two frozen outcome probes, and 40
frozen evaluation trials.

## Result

All ten validity preconditions passed on both seeds. The required control arms,
shared-bridge anatomy, generic outcome afferents, exact lesions, clean neural
commits, reward-count matching, tonic SNc calibration, probe activity, and
plasticity telemetry were present. The critic-output lesion also retained
measurable critic-route learning. Neither seed has an undefined reason.

The complete verdict metrics are:

| seed | contingent action 0 | cue-led | yoked action 0 | distance from balanced | yoked omission DA dip | yoked reward DA burst | critic-lesion omission dip | omission-path-lesion dip | intact critic spikes/trial | normalization-lesion critic spikes/trial | changed outside declared routes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 401 | 1.0 | 1.0 | 1.0 | 0.5 | 0.0001022457098265872 | 0.07630674812851812 | 0.00021410974518104586 | 0.0 | 0.725 | 122.725 | 0 |
| 409 | 1.0 | 1.0 | 1.0 | 0.5 | 0.0 | 0.08664130077643162 | 0.000974353212654433 | 0.0 | 1.15 | 106.475 | 0 |

Both seeds failed the same five fixed checks:

1. `yoked_reward_does_not_create_either_dominant_action`: action-0 preference
   was `1.0` on both seeds, outside the required `[0.25, 0.75]` interval.
2. `expected_omission_creates_lhb_rmtg_dopamine_dip`: the yoked omission probe
   produced `0/0` LHb/RMTg spikes on both seeds. Dopamine dips were
   `0.0001022457098265872` and `0.0`, below the required `0.001`.
3. `reward_veto_suppresses_negative_path_and_preserves_burst`: rewarded probes
   did activate the veto population (`45` and `49` spikes) and produced large
   dopamine bursts (`0.07630674812851812` and `0.08664130077643162`). The check
   still failed because rewarded LHb/RMTg activity was `0/0`, which could not be
   lower than the already-silent `0/0` omission response.
4. `critic_lesion_removes_expected_omission_dip`: critic-lesion dips were
   `0.00021410974518104586` and `0.000974353212654433`, not at least `0.0005`
   below the intact yoked dips. Both lesion probes also had `0/0` LHb/RMTg
   spikes.
5. `omission_path_lesion_removes_rmtg_dopamine_dip`: each lesion did report
   zero RMTg spikes and a `0.0` dip, but the intact dips were too small to show
   the required reduction of at least `0.0005`.

<!--derived-->
The remaining seven checks passed on both seeds. Contingent acquisition and
cue-led selection were perfect; executed-action eligibility and critic activity
were local; collateral and reward-to-SNc lesions blocked acquisition; local FS
normalization was load-bearing; and no synapse outside the declared actor and
critic routes changed. The collateral-lesion action-0 preferences were `0.4`
and `0.45`, with zero route change, while reward-to-SNc-lesion preferences were
`0.5` and `0.55`.

## Localization

<!--derived-->
The negative pathway was not generally incapable of firing. With local FS
normalization lesioned, mean critic activity rose from `0.725` to `122.725`
spikes per trial on seed 401 and from `1.15` to `106.475` on seed 409. The
corresponding omission probes then silenced the omission gate, produced `44/48`
LHb spikes and `16/16` RMTg spikes, and generated dopamine dips of
`0.0026806759703417726` and `0.0026013131973537384`.

With the intact circuit, the yoked omission probes instead produced only two
critic spikes in the executed channel, while the omission gate remained active
at `67` and `62` spikes and LHb/RMTg stayed silent. This localizes the present
failure to the critic-to-omission operating range or persistence of learned
expectation at outcome time, upstream of the functional LHb-RMTg-SNc chain.
The normalization lesion is diagnostic, not a successful alternative: it
caused extreme critic firing and therefore does not satisfy the intended
bounded local circuit.

## Provenance

The aggregate artifact was produced on the CuPy/NVIDIA backend from clean
commit `49f84fcbb0ea967af7fc6a78f077644d9f0e741a` in an immutable Git archive.
The provenance sidecar reports `git_dirty=false`, source kind `git_archive`,
manifest
`ebe4fe3e7f5873496a76d74d6563dbb7666c2126e44fb9202b6574068bbeb59f`,
and run ID `1785781381-1310481`. CuPy was requested and importable. The corpus
check was fresh, with query `Gate B v3 omission error and local critic
normalization; v1 and v2 failed yoked neutrality` and recorded age `2504.0`
seconds at run start.

Artifacts: `research/findings/raw/vocal_action_credit_gate_v3/calibration.json`
and its `.prov.json` sidecar.

## Decision

Do not open development seeds `419`, `421`, `431`, and `433`, or held-out seeds
`439` and `443`. Do not relax yoked neutrality, tune the actor against these
seeds, or remove normalization and accept the resulting critic saturation.

The next mechanism must give learned action value a bounded, persistent effect
at outcome time that can silence the omission gate enough to recruit LHb and
RMTg. It should preserve the now-demonstrated reward burst, local eligibility,
critic locality, normalization requirement, and plasticity scope, then prove
the negative signal with the same critic-output and omission-path lesions on a
fresh seed partition. This remains a small action-credit circuit test; it does
not establish vocal learning, natural speech, or general agency.
