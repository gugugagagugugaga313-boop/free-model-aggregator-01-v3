# Free-model aggregator (v3, hazard-priced routing)

One server serving the OpenAI and Anthropic APIs over ten free-tier
providers (OpenAI, Anthropic and Gemini wires). It reads the brief from
`MAKEFUN_PROMPT_FILE`, registers its catalog with the environment at
`MAKEFUN_ENV_URL`, serves the workload one request at a time, and exits 0
when the work is done. `node main.ts --list` prints the catalog and exits.

## Routing

Upstream failures are scheduled per provider: each provider's free-tier
budget is drawn from a published range, one or two calls refuse
transiently at ordinals >= 2, and exactly one provider goes out
mid-trial. A provider's first call never refuses.

The router prices every candidate by the hazard of its next call ordinal,
derived from the published budget ranges plus the transient refusals seen
so far:

- Weight-2 requests order by rating alone (a rating sacrifice costs
  double quality there).
- Weight-1 requests order by rating minus 300 times the next-call
  hazard, spreading load onto fresh, low-ordinal providers.
- After a transient refusal, the same provider is retried once
  immediately: the refused ordinal is spent, so the retry is cheap and
  the request stays on its high-rated provider instead of cascading.
- Terminal refusals (spent budget, outage) retire the provider for the
  rest of the trial. 400/401 picks retire the model or provider.

Hazard priors use only the kit's published budget-range constants,
identical for every instance. No wall clock, no randomness: every choice
derives from the brief, the requests and the providers' answers.

## Run

```
node main.ts
node main.ts --list
```

## Verification

Local trials against the kit's sealed environment and scorer. Practice
trials 0-2 and fresh generator draws must each score 99 with fewer than
26 upstream calls, wire/failover/catalog at 100, and byte-identical
double runs.
