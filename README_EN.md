# Structural Fair Value (SFV) — reading what a price assumes

For about 1,900 US non-financial stocks, this tool inverts a residual-income valuation to answer one question: **how fast must earnings grow over the next ten years for today's price to be right?** It then shows how often firms asked for that much actually delivered it. Every input is free public data (SEC XBRL, 13F, N-PORT, FTD, FRED, Damodaran), and every calculation uses only what was public at the time (point in time).

The full documentation is in Korean: [`README.md`](README.md), [`TUTORIAL.md`](TUTORIAL.md) (concepts from present value to the inversion), [`FRAMEWORK.md`](FRAMEWORK.md) (current specification), [`reports/INDEX.md`](reports/INDEX.md) (results). Code comments and module docstrings are in English.

## 1. The valuation framework

**Residual income, not DCF.** Value is book value plus the present value of excess earnings:

```
V_t = B_t + Σ_{k=1..10} (ROE_k − r) · B_{k−1} / (1+r)^k + TV
```

Under clean surplus the two models give the same number; the difference is where the assumptions sit. In a DCF the terminal value is typically 60–80% of the total and hangs on one terminal growth rate. Starting from observed book value narrows the assumption to a single question, how long today's excess profitability lasts, and that one is estimated rather than assumed.

**Intangibles are capitalised.** Accounting expenses R&D and SG&A, so intangible-heavy firms show small books and inflated ROE. All R&D and part of SG&A are capitalised and amortised with industry-level parameters identified from acquisition prices by Ewens, Peters and Wang (2025): R&D amortisation 33–50%, SG&A capitalised share 20–51%, organisation capital amortised at 20%. This leaves theoretical value unchanged but changes what ROE means, which matters because persistence is estimated from that ROE.

**Persistence is estimated, not assumed.** Excess ROE x = adjusted ROE − industry median follows x_k = x_∞ + ω^k (x_0 − x_∞). ω and x_∞ come from a panel regression, x_{t+1} = a + (ω_0 + ω'z) x_t + γ'z + e, with z = intangible intensity, R&D intensity, gross-margin stability, size, age, industry concentration, a negative-excess dummy and top-20%/top-5% size dummies. Expanding window, estimates used from 30 June of the following year. Estimated persistence is about 0.56 equal-weighted and 0.84 for the largest 5% of firms. The fade target is the industry median ROE; the classical alternative (fade to the cost of equity) is reported as a floor.

**Discount rate and bounds.** r = 10-year Treasury + β × ERP (Damodaran implied ERP, β from 60 months clipped to 0.5–1.5). Book grows by clean surplus with the trailing four-quarter payout. Bounds: ω ≤ 0.90, long-run excess ROE ±10pp, initial ±50pp, book growth ≤ 25% a year, terminal growth ≤ 4%. Without them values explode; with them superstar firms are systematically under-valued, which is why the next step exists.

**The inversion.** When model value differs from price, the price is not declared wrong. Instead: what would have to be true for the price to be right? Holding everything else at model values, solve one unknown so that value equals price.

| Solved value | Meaning |
|---|---|
| Required growth g* (10 years) | Annual earnings growth for ten years, then the estimated fade, that makes value = price. The headline |
| Required persistence ω* | How slowly today's excess profitability must fade; read as a half-life |
| Required horizon T* | How many years today's excess ROE must be held unchanged |
| Band on g* | g* re-solved under six capitalisation parameter sets from the literature |
| Return axis r*(g) | One price, two unknowns; fix growth at 5–30% and solve the annual return the price offers |
| Normalised g* | g* with starting earnings replaced by the five-year average margin times current revenue; flags peak and trough earnings |

Solved values are then set against **base rates**: for every firm-quarter since 2014, the growth the price required versus what was delivered over the next three years.

| Growth the price required | Delivered over the next 3 years |
|---|---|
| 5–10% | 34% |
| 10–15% | 20% |
| 15–20% | 13% |
| 20–30% | 9% |
| 30–50% | 4% |
| over 50% | 3% |

Conditional rates by size, industry, recent growth and profitability are reported alongside.

**The forward direction.** A revenue-growth and margin path gives an earnings path; the same fade and terminal follow, and out come value, a per-share target price, the return the price offers on that path, and how often firms grew that fast. For loss makers, fix a margin path to solve the required revenue growth, or the reverse. Both directions are one engine: feeding g* back in as a ten-year path returns today's price.

**Example, NVIDIA on 2026-09-30.** Required ten-year growth 20.7% a year (20.1–21.5% across capitalisation assumptions). Of firms asked for 20–30%, 9% delivered it over three years, 21% among firms that had just grown as fast. If you expect 15% growth the price offers 8.5% a year, 10.8% at 20%. On normalised earnings the required growth is 24.7%, flagged "peak". A worked walkthrough with a profitable mega cap, a loss maker and a buy-date scorecard is in [`docs/walkthrough.md`](docs/walkthrough.md) (Korean).

## 2. The prediction test

**This is not a return-prediction model.** Under a verdict rule fixed before implementation, none of the decomposition components below, and none of the benchmarks (B/M, E/P, Bartram–Grinblatt peer-implied value, ICC), predicted subsequent returns over 2014-06 to 2026-09 (13F-implied quarterly returns including delistings, Fama–MacBeth). The pre-registered rejection criterion was met and is recorded as such ([`reports/phase3_validation.md`](reports/phase3_validation.md); criterion in [`docs/design_spec_20260905.md`](docs/design_spec_20260905.md), section 2.5).

So the tool is not used to chase excess returns. Its use is to pull the growth assumption out of a price as a number, set it against how often such growth has actually happened, and re-score the assumption once results come in. Whether to believe it is the user's call.

## 3. The system around it

**Data.** SEC XBRL company facts (fundamentals, 2009–), 13F (institutional holdings, 2013–), N-PORT (fund holdings and flows, 2019–), FTD (prices of delisted names), yfinance (month-end prices, splits, consensus), FRED, Damodaran. Fundamentals use first-reported values from their filing date; 13F holdings from quarter end + 46 days; persistence estimates from 30 June of the following year. Validation returns are 13F-implied so delistings are included (correlation 0.996 with yfinance). Universe: US non-financial common stocks above $300m, roughly 1,400–1,900 names a month.

**Decomposition (diagnostic).** log(P/V) is split each month, cross-sectionally, into structural demand, transient flow, model-fit characteristics, momentum, fundamental expectations, industry, and a residual. This is what the test in section 2 was run on; it has no direction at all among mega caps, so judgement rests on the inversion in section 1.

**Tooling.** A PySide6 desktop calculator (six tabs) and the command line share one calculation library, `sfv/calc.py`. 37 data-quality invariants run on every build, and `run_sfv.py` recomputes only stale stages.

![Premise tab](docs/img/app_premise.png)

## 4. Run it

The calculator and the command line work right after cloning, because the latest month's compact tables (`app/data/`, 19 MB) ship with the repository. Recomputing the full pipeline needs about 17 GB of raw public files; see [`data/README.md`](data/README.md).

```bash
pip install -r requirements.txt            # Python 3.11
python -m app                              # desktop calculator
python scripts/what_if.py --ticker NVDA
python scripts/what_if.py --ticker IREN --solve-growth --margin 30,35,40 --years 5
python scripts/what_if.py --ticker AAPL --since 2024-12-31
python -m pytest tests -q
```

## 5. Layout

| Path | Contents |
|---|---|
| `sfv/` | 28 pipeline modules: raw parsing → point-in-time panel → valuation → decomposition and tests → inversion → shared calculation library → checks and report |
| `app/` | Desktop calculator and the compact data it reads |
| `scripts/` | Analysis scripts and the command-line tool `what_if.py` |
| `run_sfv.py` | Monthly runner that recomputes only stale stages |
| `tests/` | Unit tests on synthetic data |
| `reports/` | Thirty-odd stage reports, the per-stock table (`sfv_report.html`, `sfv_latest.csv`), data checks, code-review record |
| `docs/` | Code map, the original design spec with the pre-registered criterion, walkthrough, images |

## 6. Limits

Not a prediction model; a twelve-year sample spanning one bull market. Bounds that prevent value explosions also under-value superstars, so mega caps are read through the inversion rather than point values. Capitalisation parameters move point values by a median 46% while ranks (rank correlation 0.92–0.98) and the g* band stay stable, so judgements are made on bands, not points. Financials excluded; IFRS periods of foreign filers are not parsed; negative-book firms cannot be valued. Knowing the company (products, competition, management) is outside the tool.

## 7. How it was built

About two weeks in September 2026: design spec, verdict rule fixed before implementation, then data pipeline to tooling with a written report at every stage. Built with an AI coding assistant (Claude Code); the modelling and direction decisions were mine. The code review, the defects it found and one operational incident are recorded in [`reports/code_review.md`](reports/code_review.md).

MIT licence. Data remain subject to their sources' terms.
