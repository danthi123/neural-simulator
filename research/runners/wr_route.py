"""Amendment E (RULING wrc-prereg-go-1016b): the word-role parse-route readout for the off-target precondition.

The `for` field names the serving subsystem, not the parse route, so it is NOT accepted as the parse-route
readout. The word-role-flip gate lines' intact-at-ON off-target arm must persist each listed turn's word-role
parse route as one of three values:

  engaged     -- the word-role parse ran this turn AND produced (agent, action) the caller used
                 (the reply's `word_role_trace` is present with both agent and action non-null);
  declined    -- the parse ran this turn but produced no usable structure (trace present, agent or action
                 null: organ decline, the a==v guard, or a swallowed organ error recorded in trace.error);
  not-reached -- the parse did not run this turn (no `word_role_trace` key in the reply: the flag is off,
                 or the turn took another comprehension path).

The trace itself is attached per turn by webapp/server.py (`_word_role_trace_reset` clears it before the turn,
`_resp["word_role_trace"]` attaches it after), so a trace can only ever describe the turn it is attached to.
On every gate seed the precondition reads these rows: a listed turn whose route shows `engaged` makes the
OFF-TARGET bar UNDEFINED plus a NEEDS (ruling wrc-prereg-go-1016b), never a silent drop.
"""


def wr_route(trace):
    """The 3-valued parse-route readout from a reply's `word_role_trace` block (None when absent)."""
    if not trace or not isinstance(trace, dict):
        return "not-reached"
    if trace.get("agent") and trace.get("action"):
        return "engaged"
    return "declined"
