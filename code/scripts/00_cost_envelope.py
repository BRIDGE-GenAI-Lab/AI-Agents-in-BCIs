# Run-size and cost model for study_bci_agent_oversight Experiments 1-2.
CORE_MODELS, PANEL_MODELS = 6, 15
# Exp2 conditions requiring LLM calls
FACTORIAL = 6          # 3 uncertainty sources x 2 control mechanisms
CAUTION   = 12         # caution-wording family (all inside none x advisory)
ORACLE    = 1
EXP1_SINGLESHOT = 1    # single-shot LLM controller (Exp 1)
CORE_CONDS  = FACTORIAL + CAUTION + ORACLE + EXP1_SINGLESHOT   # 20
SCAFFOLDS_CORE = 3     # paraphrase robustness on the 6 factorial cells only
EPISODES = 300         # stratified per cell across confidence deciles x participants
CALLS_PER_EPISODE = 4  # multi-turn agent loop (single-shot = 1, averaged in below)
IN_TOK, OUT_TOK = 2000, 300
CACHED_FRAC = 0.75     # frozen system+tool prefix is cacheable

core_combos  = FACTORIAL*SCAFFOLDS_CORE + (CAUTION+ORACLE+EXP1_SINGLESHOT)*1   # 18+14=32
core_eps     = CORE_MODELS  * core_combos * EPISODES
panel_eps    = PANEL_MODELS * FACTORIAL   * EPISODES          # broad panel, primary contrast only
var_eps      = 5 * 30 * 50                                    # variance sub-study: 5 cells x 30 reps x 50 eps
total_eps    = core_eps + panel_eps + var_eps
calls        = total_eps * CALLS_PER_EPISODE
tin, tout    = calls*IN_TOK, calls*OUT_TOK
tin_cached, tin_full = tin*CACHED_FRAC, tin*(1-CACHED_FRAC)

print(f"core condition-scaffold combos : {core_combos}")
print(f"episodes  core/panel/variance  : {core_eps:,} / {panel_eps:,} / {var_eps:,}")
print(f"TOTAL episodes                 : {total_eps:,}")
print(f"TOTAL LLM calls                : {calls:,}")
print(f"input tok {tin/1e6:,.0f}M (cached {tin_cached/1e6:,.0f}M) | output tok {tout/1e6:,.0f}M\n")
print(f"{'blend $/M in':>12} {'$/M out':>8} {'no cache':>12} {'w/ cache':>12}")
for pin, pout in ((0.5,1.5),(1.0,4.0),(2.0,10.0),(3.0,15.0)):
    full = tin/1e6*pin + tout/1e6*pout
    cach = (tin_full/1e6*pin + tin_cached/1e6*pin*0.1) + tout/1e6*pout
    print(f"{pin:>12.2f} {pout:>8.2f} {'$'+format(full,',.0f'):>12} {'$'+format(cach,',.0f'):>12}")
print("\nSMOKE (1% of core, 3 models):")
s_eps = 3*core_combos*3
s_calls = s_eps*CALLS_PER_EPISODE
print(f"  episodes {s_eps:,}  calls {s_calls:,}  ~${s_calls*(IN_TOK*2.0+OUT_TOK*10.0)/1e6:,.2f} at blended 2/10")
