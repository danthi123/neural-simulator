# GAP B scoping: Parser front-end driving RF operand registers

Recommendation: Start with B-ii (masked rf_kick on gate-open signal).

Load-bearing constraint: Transmission gates multiply cp_connections (Izhikevich), NOT RF synaptic weights. Verified bridge.py:5528 shows RF complex matvec has NO transmission_gain factor. This kills B-i unless new Izhikevich-to-RF driver layer is wired.

Two options:

B-i (All-synaptic): Gate RF-to-RF complex synapse via new Izhikevich role_src_driver. Requires transformation layer (not free).

B-ii (Masked rf_kick): Parser fires (neural decision) -> gate coupling detects -> runner issues bridge.rf_kick(role_R_phasor, neuron_mask=role_bank_mask). No new layer; phasor is fixed wiring constant. Brain-based-compliant.

First de-risk (STEP B1): One persistent RF bridge. Parser slice (BridgeParser), RF operands (op_A, op_B, role_bank, bnd_A, bnd_B, acc), transmission gates. Per word: drive parser -> role ensemble fires -> coupling opens gate -> runner detects and calls masked rf_kick -> set role_bank to role phasor -> FHRR bind -> bundle -> store.

GO criterion: 3 seeds x 2 D (6/6 exact) stored facts query back matching oracle; multi-word stress test.

Anti-cheats: parser-route lesion, permuted role mapping, moat battery.

Reusable machinery:
- BridgeParser (brain_conversational_agent.py:28)
- couple_gate_to_indices (unified_brain_bridge.py:123)
- couple_gate_to_pool (bridge.py:3085)
- _apply_gate_couplings (bridge.py:3108)
- rf_kick with neuron_mask (bridge.py:5448)
- RF step loop (bridge.py:5512-5547)
- merge_population_into_shared_bridge (unified_brain_bridge.py:151)

Honest risks: EMA timing vs parser spike (MEDIUM), RF tracker re-init (SMALL, pre-approved), role phasor drift (SMALL), query not yet parser-driven (MEDIUM, deferred).

Next steps: B1 GO -> STEP B2 (query-side parser routing) -> STEP B3 (OneBrainComposer class) -> STEP B-i (later: all-synaptic).

Summary: B-ii for first de-risk. Smaller, faster, compliant. B-i is later refinement.

Key insight: Fixed code projections (role phasor as wiring constant) are legitimate. Decision (which role) is neural; target code is fixed. This is compliant.

Verified references: Production scoping 2026-06-18-production-one-brain-composer-scoping.md. Gate coupling bridge.py:3085-3119, unified_brain_bridge.py:123-148. Masked rf_kick bridge.py:5448-5547.
