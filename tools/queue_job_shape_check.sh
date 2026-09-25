#!/usr/bin/env bash
# queue_job_shape_check.sh — ENQUEUE-TIME "would this even start?" check, shared by every queue producer
# (tools/pool_queue.sh add, tools/gpu_queue.sh add, tools/queue_add.sh). Sourced, like
# tools/pool_revision_marker.sh, so there is exactly one place this predicate is spelled out.
#
# WHY (2026-09-25). Six SETTLE A2 pool lines were queued as
#   'A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/... && env ... .venv/bin/python -u -m
#    research.runners._affect_marker_settle_congruence --run-wiring --seeds 42 ...  #checked:...'
# -- a prose label ("A2 wiring seed 42:") glued onto the front of an otherwise-correct job -- so on the node
# bash ran `A2` as a command, got "command not found" (rc=127), and nothing ran, while the board and the
# queue's own claim record said the jobs were dispatched (research/queue/pool.queue.claims:1923-1928 -- six
# lines, seeds 42/43/44/100/101/102; research/FAILURE_LOG.md's 2026-09-25 SETTLE A2 row).
#
# This is not hypothetical or a one-off: research/queue/gpu_queue.log already carries a REAL, historical
# instance of the identical failure class (2026-08-31/09-01, three separate cycles, lines ~672190/711341/
# 760341) -- a queued job that was the single bare word `status` (almost certainly `gpu_queue.sh add status`,
# meant to run `gpu_queue.sh status`), which ran as `bash: line 1: status: command not found`, rc=127, in
# under a second each time, and was never flagged as anything other than an ordinary DONE(rc=127) line.
#
# NEITHER of pool_queue.sh's existing checks (COMMAND VALIDITY GATE / REMOTE VALIDITY) catches this class: both
# key off finding `-m research.runners.X` ANYWHERE in the line and validate THAT module's --help -- a line
# whose first word is prose/garbage but that happens to mention a real module further down the very same line
# (exactly the A2 shape) sails through every one of them, because none of them ever asks "would bash even
# attempt to run this line, or does it die on word 1?".
#
# WHAT THIS CHECKS (and nothing more -- it is a SHAPE check, not a correctness check):
#   1. Does the line PARSE as shell (`bash -n`)? Catches a syntax-broken line (unbalanced quotes, a dangling
#      `&&`, a torn line starting with `&`, ...).
#   2. Is the first word of the job's first REAL simple command something bash could invoke: a builtin, a
#      command on PATH, or a PATH-shaped token (anything containing a "/", e.g. `.venv/bin/python`)? If not,
#      the line would die on argv[0] with "command not found" (rc=127) the instant a shell ran it -- REFUSE it.
#      "First real simple command" means: descend into `( ... )`, `{ ...; }`, `!`, `time [-p]` and the
#      condition of `if`/`while`/`until`, skip leading `VAR=val` assignments and redirections, and step over
#      up to 8 leading no-op commands (`:`, `true`, a pure assignment such as `mem_gb=8`) joined by `&&`, `;`,
#      `&`, `|` or a newline -- 90 historical pool lines start `: mem_gb=N &&` or `mem_gb=N &&`, and for those the
#      command that matters is the one after the no-op.
#   3. `coproc` is REFUSED outright: a queued job's exit status is what the dispatchers record, and a
#      coprocess runs detached from it.
#
#   A PATH-shaped first word is accepted WITHOUT checking it actually exists on disk: whether
#   `.venv/bin/python` exists depends entirely on WHERE the job eventually runs (the local checkout for a GPU
#   job, `~/derisk-pool/sim` or an isolated `~/derisk-pool/revisions/<sha>` for a pool job, none of which is
#   necessarily this process's own cwd -- a worktree, in particular, has no `.venv` of its own at all) --
#   that is exactly what pool_queue.sh's own REMOTE VALIDITY check (and gpu_queue.sh's local argparse check)
#   already verify, each in the one environment where the answer is actually meaningful. The cost: a torn line
#   whose first surviving word happens to contain a "/" is NOT caught here -- pool.queue.claims holds seven
#   (lines 1455, 1461, 1473, 1625, 1630, 1637, 1909, e.g. `tive-memory/lb.json ...`). Those die rc=127 on the
#   node, and tools/pool_autodispatch.sh's check_fast_fail is what reports them.
#
# HOW -- A STATIC PARSER; NOTHING FROM THE JOB IS EVER EXECUTED (2026-09-25 fix round, review HIGH). The first
# version captured the first command with a DEBUG trap in a throwaway `bash` that ran the job; DEBUG traps are
# not inherited by subshells, so a job starting with `( ... )`, `coproc`, `(cmd) | cat`, `if (cmd)` or
# `time ( ... )` actually RAN on this machine when it was queued. This version never runs the job in any form.
# The only processes it starts are:
#   * `bash -n -c "$job"` -- parse only; `-n` reads commands and never executes them;
#   * one lookup, `type -t -- "$2"`, in a fresh bash started with `exec -c` (an EMPTY environment: no exported
#     functions, no BASH_ENV) and the caller's PATH passed in as data -- the candidate word is a positional
#     argument, never spliced into code. The empty environment also means the CALLER's own functions (e.g.
#     gpu_queue.sh's `daemon`, `selftest`) never make a job named after one of them look runnable.
# Everything else is a small tokenizer written in bash string operations below: quotes ('...', "...", $'...'),
# backslash escapes, $(...), $((...)), ${...}, backticks, <(...)/>(...), array assignments, redirections with an
# IO number or {varname}, comments and here-document bodies. tests/test_queue_job_shape_check.py drives shapes
# the old version executed and asserts a marker file each would create stays absent.
#
# WHEN THE PARSER IS UNSURE IT ACCEPTS (never refuses what it cannot positively identify as broken): a command
# word built from an expansion (`$PY`, `"$HOME"/x`, a glob, a brace expansion), `((...))`, `[[`, `for`, `case`,
# `select`, a function definition, an `else`/`elif` branch, the body of an `until` loop, or anything after `||`.
#
# Deliberately does NOT `set -e`/`-u`/`-o pipefail` at this top level: this file is SOURCED by callers with
# different shell-option regimes (pool_queue.sh already runs under `-uo pipefail`; gpu_queue.sh runs under
# plain `-e`), and a source-time `set` change here would alter the CALLER's shell options for the rest of its
# own run, not just this file. Every variable below is a `local` of queue_job_runnable_check, read and written
# by the `_qjs_*` helpers through bash's dynamic scoping, so behaviour does not depend on the caller's options;
# the direct-invocation CLI path at the bottom sets its own options explicitly.

# ------------------------------------------------------------------------------------------------ tokenizer
# State (locals of queue_job_runnable_check): _qs text, _qp index, _qn length, _qhd pending here-doc delimiters.
# Per token: _qk kind (W word, O control operator, R redirection operator, N newline, E end), _qt raw text,
# _qv the word's value with quotes removed, _qd=1 if the value depends on an expansion, _qb=1 if it contains an
# unquoted brace, _q0 the token's start index.

_qjs_sq() {   # just past an opening '  -- $1=1 appends the body to _qv
  local rest=${_qs:_qp} body
  body=${rest%%\'*}
  if [ "$1" = 1 ]; then _qv+=$body; fi
  _qp=$((_qp + ${#body} + 1))
}

_qjs_ansi() {   # just past $'  -- skip to the closing quote, honouring backslash escapes
  local c
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      '\') _qp=$((_qp + 2)) ;;
      "'") _qp=$((_qp + 1)); return 0 ;;
      *) _qp=$((_qp + 1)) ;;
    esac
  done
}

_qjs_bq() {   # just past an opening backtick
  local c
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      '\') _qp=$((_qp + 2)) ;;
      '`') _qp=$((_qp + 1)); return 0 ;;
      *) _qp=$((_qp + 1)) ;;
    esac
  done
}

_qjs_dq() {   # just past an opening "  -- $1=1 appends the literal value to _qv
  local c nx
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      '"') _qp=$((_qp + 1)); return 0 ;;
      '\')
        nx=${_qs:_qp+1:1}
        case "$nx" in
          '$'|'`'|'"'|'\') if [ "$1" = 1 ]; then _qv+=$nx; fi; _qp=$((_qp + 2)) ;;
          $'\n') _qp=$((_qp + 2)) ;;
          *) if [ "$1" = 1 ]; then _qv+='\'; fi; _qp=$((_qp + 1)) ;;
        esac ;;
      '$') _qjs_dollar ;;
      '`') _qd=1; _qp=$((_qp + 1)); _qjs_bq ;;
      *) if [ "$1" = 1 ]; then _qv+=$c; fi; _qp=$((_qp + 1)) ;;
    esac
  done
}

_qjs_paren() {   # just past an opening ( of $( ), $(( )), <( ), >( ) or an array assignment -- skip to its match
  local depth=1 c
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      '(') depth=$((depth + 1)); _qp=$((_qp + 1)) ;;
      ')') depth=$((depth - 1)); _qp=$((_qp + 1)); if [ "$depth" -eq 0 ]; then return 0; fi ;;
      "'") _qp=$((_qp + 1)); _qjs_sq 0 ;;
      '"') _qp=$((_qp + 1)); _qjs_dq 0 ;;
      '\') _qp=$((_qp + 2)) ;;
      '`') _qp=$((_qp + 1)); _qjs_bq ;;
      *) _qp=$((_qp + 1)) ;;
    esac
  done
}

_qjs_brace() {   # just past ${ -- skip to its matching }
  local depth=1 c
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      '{') depth=$((depth + 1)); _qp=$((_qp + 1)) ;;
      '}') depth=$((depth - 1)); _qp=$((_qp + 1)); if [ "$depth" -eq 0 ]; then return 0; fi ;;
      "'") _qp=$((_qp + 1)); _qjs_sq 0 ;;
      '"') _qp=$((_qp + 1)); _qjs_dq 0 ;;
      '\') _qp=$((_qp + 2)) ;;
      '`') _qp=$((_qp + 1)); _qjs_bq ;;
      '$') _qjs_dollar ;;
      *) _qp=$((_qp + 1)) ;;
    esac
  done
}

_qjs_dollar() {   # at a $ -- every form marks the value as expansion-dependent
  _qd=1
  local nx=${_qs:_qp+1:1}
  case "$nx" in
    '(') _qp=$((_qp + 2)); _qjs_paren ;;
    '{') _qp=$((_qp + 2)); _qjs_brace ;;
    "'") _qp=$((_qp + 2)); _qjs_ansi ;;
    '"') _qp=$((_qp + 2)); _qjs_dq 0 ;;
    *) _qp=$((_qp + 1)) ;;   # $name / $1 / $@ ...: the name's characters are ordinary word characters
  esac
}

_qjs_word() {
  local start=$_qp c nx re_arr='^[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?\+?=$'
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      ' '|$'\t'|$'\n'|'|'|'&'|';'|')') break ;;
      '(')
        if [[ ${_qs:start:_qp-start} =~ $re_arr ]]; then _qd=1; _qp=$((_qp + 1)); _qjs_paren; else break; fi ;;
      '<'|'>')
        if [ "${_qs:_qp+1:1}" = '(' ]; then _qd=1; _qp=$((_qp + 2)); _qjs_paren; else break; fi ;;
      '\')
        nx=${_qs:_qp+1:1}
        if [ "$nx" != $'\n' ]; then _qv+=$nx; fi
        _qp=$((_qp + 2)) ;;
      "'") _qp=$((_qp + 1)); _qjs_sq 1 ;;
      '"') _qp=$((_qp + 1)); _qjs_dq 1 ;;
      '$') _qjs_dollar ;;
      '`') _qd=1; _qp=$((_qp + 1)); _qjs_bq ;;
      '*'|'?') _qd=1; _qv+=$c; _qp=$((_qp + 1)) ;;
      '{'|'}') _qb=1; _qv+=$c; _qp=$((_qp + 1)) ;;
      '~') if [ "$_qp" -eq "$start" ]; then _qd=1; fi; _qv+=$c; _qp=$((_qp + 1)) ;;
      *) _qv+=$c; _qp=$((_qp + 1)) ;;
    esac
  done
  _qt=${_qs:start:_qp-start}
}

_qjs_heredocs() {   # just past the newline that ends a line carrying << / <<- redirections: skip their bodies
  local ent strip delim rest line tab=$'\t'
  for ent in "${_qhd[@]}"; do
    strip=${ent%%:*}; delim=${ent#*:}
    while [ "$_qp" -lt "$_qn" ]; do
      rest=${_qs:_qp}
      line=${rest%%$'\n'*}
      _qp=$((_qp + ${#line} + 1))
      if [ "$strip" = 1 ]; then line=${line#"${line%%[!$tab]*}"}; fi
      if [ "$line" = "$delim" ]; then break; fi
    done
  done
  _qhd=()
}

_qjs_next() {
  _qk=E; _qt=''; _qv=''; _qd=0; _qb=0
  local c two three rest re_io='^([0-9]+|\{[A-Za-z_][A-Za-z0-9_]*\})$' io io0
  while [ "$_qp" -lt "$_qn" ]; do
    c=${_qs:_qp:1}
    case "$c" in
      ' '|$'\t') _qp=$((_qp + 1)) ;;
      '\') if [ "${_qs:_qp+1:1}" = $'\n' ]; then _qp=$((_qp + 2)); else break; fi ;;
      '#') rest=${_qs:_qp}; rest=${rest%%$'\n'*}; _qp=$((_qp + ${#rest})) ;;   # a comment runs to end of line
      *) break ;;
    esac
  done
  _q0=$_qp
  if [ "$_qp" -ge "$_qn" ]; then return 0; fi
  c=${_qs:_qp:1}; two=${_qs:_qp:2}; three=${_qs:_qp:3}
  case "$c" in
    $'\n')
      _qk=N; _qt=$c; _qp=$((_qp + 1))
      if [ "${#_qhd[@]}" -gt 0 ]; then _qjs_heredocs; fi
      return 0 ;;
    '<'|'>')
      if [ "${_qs:_qp+1:1}" != '(' ]; then   # `<(`/`>(` is a process substitution, i.e. a word
        _qk=R
        case "$three" in
          '<<<'|'<<-') _qt=$three ;;
          *) case "$two" in '<<'|'>>'|'<&'|'>&'|'<>'|'>|') _qt=$two ;; *) _qt=$c ;; esac ;;
        esac
        _qp=$((_qp + ${#_qt})); return 0
      fi ;;
    '&')
      case "$three" in '&>>') _qk=R; _qt=$three; _qp=$((_qp + 3)); return 0 ;; esac
      case "$two" in
        '&>') _qk=R; _qt=$two; _qp=$((_qp + 2)); return 0 ;;
        '&&') _qk=O; _qt=$two; _qp=$((_qp + 2)); return 0 ;;
      esac
      _qk=O; _qt=$c; _qp=$((_qp + 1)); return 0 ;;
    '|')
      case "$two" in '||'|'|&') _qt=$two ;; *) _qt=$c ;; esac
      _qk=O; _qp=$((_qp + ${#_qt})); return 0 ;;
    ';')
      case "$three" in ';;&') _qt=$three ;; *) case "$two" in ';;'|';&') _qt=$two ;; *) _qt=$c ;; esac ;; esac
      _qk=O; _qp=$((_qp + ${#_qt})); return 0 ;;
    '('|')')
      _qk=O; _qt=$c; _qp=$((_qp + 1)); return 0 ;;
  esac
  _qk=W; _qjs_word
  # An IO number or {varname} written directly before a redirection (`2>/dev/null`, `{fd}>f`) belongs to it.
  if [[ $_qt =~ $re_io ]]; then
    c=${_qs:_qp:1}
    if { [ "$c" = '<' ] || [ "$c" = '>' ]; } && [ "${_qs:_qp+1:1}" != '(' ]; then
      io=$_qt; io0=$_q0; _qjs_next; _qt="$io$_qt"; _q0=$io0
    fi
  fi
  return 0
}

# --------------------------------------------------------------------------------------------- classifiers

_qjs_reserved() {   # an unquoted, expansion-free word that bash treats as a reserved word in command position
  [ "$_qd" = 0 ] && [ "$_qt" = "$_qv" ] || return 1
  case "$_qt" in
    '!'|'{'|'}'|if|then|else|elif|fi|while|until|do|done|for|case|esac|select|function|time|coproc|'[['|']]'|in) return 0 ;;
  esac
  return 1
}

_qjs_assignment() {   # NAME=..., NAME+=..., NAME[i]=...
  local re='^[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?\+?='
  [[ $_qt =~ $re ]]
}

_qjs_redir() {   # consume the operand of the redirection operator in _qt; 1 if there is none
  local op=$_qt
  _qjs_next
  [ "$_qk" = W ] || return 1
  case "$op" in
    '<<') _qhd+=("0:$_qv") ;;
    '<<-') _qhd+=("1:$_qv") ;;
  esac
  return 0
}

_qjs_time_opts() {   # after `time`: consume a -p option, if present
  local save=$_qp
  _qjs_next
  if [ "$_qk" = W ] && [ "$_qt" = '-p' ]; then return 0; fi
  _qp=$save
}

_qjs_funcdef() {   # is the command word just read followed by `( )`, i.e. a function definition?
  local rest=${_qs:_qp} tab=$'\t'
  rest=${rest#"${rest%%[! $tab]*}"}
  [ "${rest:0:1}" = '(' ] || return 1
  rest=${rest:1}
  rest=${rest#"${rest%%[! $tab]*}"}
  [ "${rest:0:1}" = ')' ]
}

# _qjs_command: classify the command word in _qv. 0 = runnable (or not decidable statically), 1 = would die
# with "command not found", 2 = a no-op (`:` / `true`) whose successor is the command that matters.
_qjs_command() {
  if [ "$_qd" = 1 ] || [ "$_qb" = 1 ]; then return 0; fi
  case "$_qv" in
    */*) return 0 ;;
    ':'|true) return 2 ;;
  esac
  if [ -n "$_qv" ] && ( exec -c "${BASH:-bash}" --norc --noprofile -c \
       'PATH=$1; type -t -- "$2" >/dev/null 2>&1' _qjs_type "${PATH:-}" "$_qv" ) 2>/dev/null; then
    return 0
  fi
  return 1
}

# queue_job_runnable_check <job-line>
#   OK:      prints nothing, returns 0
#   REFUSE:  prints one "⛔ REFUSED: ..." message to stdout, returns 1
queue_job_runnable_check() {
  local job="${1:-}"
  if [ -z "$(printf '%s' "$job" | tr -d '[:space:]')" ]; then
    printf '⛔ REFUSED: empty job line.\n'
    return 1
  fi
  # 0. ONE LINE. Every queue stores one job per line (the GPU daemon runs each line as its own job), so a job
  #    containing a newline or carriage return would be split into several jobs, each unchecked (review of
  #    03c53f5f7: `gpu_queue.sh add $'echo ok\nstatus'` queued a second job `status` that died rc=127).
  case "$job" in
    *$'\n'*|*$'\r'*)
      printf '⛔ REFUSED: the job contains a line break; a queue record is one line (join with && or ;).\n'
      return 1 ;;
  esac

  # 1. SYNTAX (parse only: -n never executes anything).
  local syn
  if ! syn=$(bash -n -c "$job" 2>&1); then
    printf '⛔ REFUSED: not valid shell syntax (bash -n): %s\n' "${syn:-parse error}"
    return 1
  fi

  # 2. STATIC walk to the first real simple command. States: CS = a command may start here; SP = inside a
  #    simple command's prefix (assignments/redirections seen, no command word yet); SKIP = inside a no-op
  #    command's arguments; AFTER = just closed a group made only of no-ops.
  local _qs="$job" _qp=0 _qn=${#job} _qk='' _qt='' _qv='' _qd=0 _qb=0 _q0=0
  local -a _qhd=()
  local state=CS neg=0 noops=0 kw='' cmd_at=0 rc reuse=0 first
  while :; do
    if [ "$reuse" = 1 ]; then reuse=0; else _qjs_next; fi
    rc=-1
    case "$state" in
      CS)
        case "$_qk" in
          E) return 0 ;;
          N) ;;
          R) cmd_at=$_q0; _qjs_redir || return 0; state=SP ;;
          O)
            case "$_qt" in
              '(') if [ "${_qs:_qp:1}" = '(' ]; then return 0; fi ;;   # `((...))` arithmetic; else a subshell: descend
              ')') state=AFTER ;;
              *) return 0 ;;   # an operator where a command should start: a construct not modelled here
            esac ;;
          W)
            if _qjs_reserved; then
              case "$_qt" in
                '!') neg=1 ;;
                '{'|then) ;;
                if|while|until) kw=$_qt ;;
                do) if [ "$kw" = until ]; then return 0; fi ;;   # an `until` body runs only if its condition fails
                time) _qjs_time_opts ;;
                '}'|fi|done|esac) state=AFTER ;;
                coproc)
                  printf '⛔ REFUSED: the job is a coprocess (`coproc ...`) -- a queued job must run in the foreground of its own shell, whose exit status the dispatcher records; a coprocess runs detached from it. Job: %s\n' "${job:0:160}"
                  return 1 ;;
                *) return 0 ;;   # for/case/select/[[/function/else/elif/...: accepted, see the header
              esac
            elif _qjs_assignment; then
              cmd_at=$_q0; state=SP
            else
              cmd_at=$_q0
              if _qjs_funcdef; then return 0; fi
              _qjs_command; rc=$?
            fi ;;
        esac ;;
      SP)
        case "$_qk" in
          R) _qjs_redir || return 0 ;;
          W) if ! _qjs_assignment; then _qjs_command; rc=$?; fi ;;
          *) rc=2; reuse=1 ;;   # assignments/redirections only (`mem_gb=8 && ...`): a no-op; re-read this token below
        esac ;;
      SKIP|AFTER)
        case "$_qk" in
          E) return 0 ;;
          N) state=CS; neg=0 ;;
          R) _qjs_redir || return 0 ;;
          O)
            case "$_qt" in
              '&&'|';'|'&'|'|'|'|&') state=CS; neg=0 ;;
              ')') state=AFTER ;;
              *) return 0 ;;   # `||`: what follows runs only if the no-op failed, which it cannot
            esac ;;
          W)
            if [ "$state" = AFTER ]; then
              if ! _qjs_reserved; then return 0; fi
              case "$_qt" in
                '}'|fi|done|esac) ;;
                then) state=CS ;;
                do) if [ "$kw" = until ]; then return 0; fi; state=CS ;;
                *) return 0 ;;
              esac
            fi ;;   # in SKIP, a word is just an argument of the no-op
        esac ;;
    esac
    case "$rc" in
      0) return 0 ;;
      1)
        first=${_qs:cmd_at:200}; first=${first%%$'\n'*}
        printf '⛔ REFUSED: first word %s of the job'"'"'s first command is not a shell builtin/keyword and does not resolve on PATH -- it would die instantly with "command not found" (rc=127), exactly like the 2026-09-25 SETTLE A2 lines and the historical `status` job in gpu_queue.log. First command: %s\n' \
          "'$_qv'" "$first"
        return 1 ;;
      2)
        if [ "$neg" = 1 ]; then return 0; fi   # `! :` is false, so what follows may never run
        noops=$((noops + 1))
        if [ "$noops" -gt 8 ]; then return 0; fi
        state=SKIP ;;
    esac
  done
}

if [ "${BASH_SOURCE[0]:-}" = "${0}" ]; then
  set -uo pipefail
  [ "$#" -eq 1 ] || { echo "usage: $0 '<job line>'" >&2; exit 2; }
  if out=$(queue_job_runnable_check "$1"); then
    exit 0
  else
    printf '%s\n' "$out" >&2
    exit 1
  fi
fi
