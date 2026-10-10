# FairBench: industry standards, vendor capability, and product-validation report

**Research date:** 8 October 2026

**Status on 10 October 2026.** This report was written before the real-fund work and describes the repository as it was on 8 October. Three of its recommendations have since been acted on: the pipeline ran on nine real US funds from SEC filings (actions 7 and 8 in section 17; [`README.md`](README.md), [`REAL_FUND_RESULTS.md`](REAL_FUND_RESULTS.md)); weighted reference distributions and a holdings database with provenance were added (actions 3 to 5, in part); and the code now says "within-mandate return difference" where this report asks for it. Where the text below says no real fund has been analysed, read it as of 8 October. The comparison with FactSet's Cabot Reveal Plus (action 9) and the customer interviews (action 1) are still open.

**Scope:** performance measurement and attribution, GIPS, standard/custom/mandate benchmarks, peer groups, ESG and carbon analytics, SFDR, ESMA and SEC fund-name rules, institutional manager oversight, and the specific gap FairBench may fill.

## 1. Executive verdict

The institutional problem is real: asset owners and consultants repeatedly have to reconcile a manager’s realized return with a benchmark, a mandate, ESG/carbon policies, external-manager reporting, and incomplete or changing data. Official materials from NBIM, CalPERS, Varma, Ilmarinen, Allianz, CFA/GIPS, ESMA, and the European Commission document versions of this workflow.

The stronger conclusion is narrower:

> FairBench should be a mandate-to-attribution audit layer that constructs an explicitly documented, mandate-conditioned feasible-portfolio reference distribution, then compares a real portfolio with it alongside ordinary benchmark, peer, Brinson, factor, risk, and ESG analytics.

FairBench is not currently a replacement for Bloomberg PORT, MSCI, Morningstar Direct, FactSet, SimCorp/Axioma, Aladdin, GIPS reporting, or an index provider. Those products already cover the mainstream questions: what the return was, what drove it, what risks were taken, how the portfolio compares with a benchmark or peer group, and whether a rule or limit was breached.

FairBench’s potentially differentiated question is more specific:

> **Given the same frozen universe, mandate rules, information set, and portfolio size/weighting policy, how unusual was the realized portfolio’s outcome among portfolios that were actually feasible under those rules?**

That is a plausible product opportunity, not yet a proven business. The repository itself says all attribution results are synthetic, the current analysis is substantially equal-weight and static-window, and no real fund has yet been analysed ([README](README.md), [attribution implementation](fairbench/apps/attribution.py), [data loader](fairbench/data.py)).

### Recommendation

Narrow the product and run a real-data pilot. Sell first to a mid-sized institutional asset owner or fund-of-funds with recurring external-equity-manager reviews, an internal performance/sustainability team, and access to holdings plus mandate documents. Deliver a versioned report and data lineage, not a new portfolio-management system.

Keep ESG as a useful wedge because it creates concrete policy language, compliance pressure, and data inconsistency. Do not make ESG the only category: the same method can apply to liquidity, country, sector, duration, style, capacity, or risk constraints.

Postpone quantum as a commercial claim. The repository supports a research statement about noiseless oracle-query complexity; it does not support a current-hardware speedup claim.

## 2. FairBench in one paragraph

FairBench takes a fund or mandate’s stated rules, translates them into a structured constraint specification, samples portfolios that satisfy those constraints, and compares the real fund’s return or risk metric with the resulting distribution. A broad market benchmark answers “how did the fund do against this market reference?” Traditional attribution answers “which allocation, selection, factor, or risk decisions explain the difference?” FairBench adds “how did the fund do against portfolios that could have been built under the same stated rules?” Its output should be described as a **mandate-conditioned feasible-portfolio distribution**, **conditional null**, or **counterfactual reference set**. It is not automatically a benchmark in the GIPS/index sense.

Simple example: suppose an active fund can own 20 stocks from a 100-stock universe, excludes tobacco, caps any sector at 15%, requires average ESG of at least 5% above the universe, and targets carbon intensity at least 25% below the universe. The S&P 500 may show that the fund lagged by 2%. A Brinson report may say sector allocation cost 80 bps and security selection added 30 bps. FairBench asks a different question: among thousands of portfolios satisfying those same rules, where did the actual fund’s forward return fall? If the constrained distribution itself lagged the broad benchmark by 2.5%, the rules explain much of the broad-benchmark gap; if the fund also lagged the constrained distribution, that is a within-rule result. It is not proof of skill or lack of skill, and it is not a claim that the random portfolios are investable alternatives unless the universe, weights, turnover, liquidity, and timing rules make them so.

## 3. The critical conceptual distinction: benchmark versus feasible distribution

| Object | What it is | Typical purpose | What it does not establish |
|---|---|---|---|
| Market-index benchmark | A governed index with defined constituents, weights, and rebalancing rules | Relative return/risk measurement | That the manager could have selected any other index constituent under the mandate |
| Parent or ESG index | A market-cap or rules-based index, sometimes derived from a parent index | Passive implementation and benchmark-relative evaluation | That its constituents represent the full set of portfolios allowed by a bespoke mandate |
| Custom/blended benchmark | A deterministic blend or custom index methodology | Match asset-class/style exposures or policy weights | A probability distribution over feasible portfolios |
| Peer group | A selected set of other funds/managers | Relative ranking and manager research | That peers share the same constraints, information, capacity, or decision process |
| Bespoke factor benchmark | A conditional factor model or factor-mimicking reference | Adjust for style/sector/constraint exposures | A path-realistic portfolio set unless explicitly constructed that way |
| FairBench reference distribution | A modeled set/distribution of portfolios satisfying specified constraints | Mandate-conditioned comparison and constraint/within-rule decomposition | Causality, manager skill, or economic realism unless validated and modeled |

The academic literature supports the general premise that mandates and constraints affect measured performance. Beber, Brandt, Cen, and Kavajecz construct a conditional multi-factor bespoke benchmark and find that mandate-aware performance and peer rankings differ from generic comparisons ([Journal of Empirical Finance version](https://img1.wsimg.com/blobby/go/d70dd63-b877-459f-a3ef-43a1e0215aac/downloads/Mutual%20Fund%20Performance%20-%20Using%20Bespoke%20Benchmarks%20to%20Disentangle%20Mandates%2C%20Constraints%20and%20Skill.pdf?ver=1608176043074); [SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3211588)). That is evidence for mandate-aware evaluation, not evidence that a custom benchmark equals a feasible-portfolio distribution.

## 4. What the current industry standards answer

### GIPS

The GIPS standards are a voluntary ethical framework for fair representation and full disclosure of investment performance. They cover firms, asset owners, and verifiers; GIPS reports use composites or pooled funds and require or guide benchmark, return, asset, dispersion, and risk disclosures. GIPS benchmark guidance expects an appropriate benchmark for the mandate/objective/strategy, with disclosures where no appropriate benchmark exists ([GIPS standards](https://www.gipsstandards.org/standards/); [Guidance Statement on Benchmarks for Firms](https://www.gipsstandards.org/wp-content/uploads/2023/08/gs_benchmarks_firms.pdf); [Asset Owner Handbook](https://www.gipsstandards.org/standards/gips-standards-for-asset-owners/gips-standards-handbook-for-asset-owners/)).

**FairBench relationship:** complementary. A feasible-set model could support benchmark-appropriateness review, composite documentation, or supplemental analysis. GIPS does not require sampling a distribution of feasible portfolios.

### Brinson and holdings-based attribution

Brinson-style attribution explains benchmark-relative performance using allocation, selection, and interaction effects. It needs portfolio and benchmark weights and group/security returns. CFA Research Foundation materials distinguish holdings-based, transaction-based, returns-based, and risk attribution and note the difficulty of combining multi-period effects because of compounding ([CFA Research Foundation performance attribution review](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf)).

**FairBench relationship:** complementary. Brinson explains realized active return relative to a benchmark. FairBench models an alternative reference object conditioned on mandate rules. FairBench should not replace Brinson.

### Factor and risk attribution

Factor attribution asks which systematic and specific factors explain return; risk attribution asks which exposures explain volatility, tracking error, or another risk measure. It needs factor definitions, exposures, factor returns, covariance/risk-model inputs, portfolio weights, and dates. It is model-dependent and is not causal proof.

**FairBench relationship:** complementary. Factor exposures should be reported for both the actual fund and the feasible distribution. Otherwise a “manager effect” can simply be a factor tilt hidden inside the rule set.

### Peer groups

Peer groups answer “how did this manager perform relative to selected peers?” CFA’s “Fixing the Peer Group Problem” documents survivor bias, composition bias, timeliness, and mandate mismatch, and discusses simulated peer groups as useful when no suitable index or peer group exists ([CFA Institute](https://rpc.cfainstitute.org/research/cfa-magazine/2017/fixing-the-peer-group-problem)).

**FairBench relationship:** complementary or stronger for one narrow comparison. A transparent feasible distribution conditions on an explicit mandate; a peer group is an empirical comparison set. Neither should be presented as a universal substitute for the other.

### ESG, carbon, and climate analytics

Standard analytics calculate ESG scores, controversies, business involvement, carbon footprint, weighted-average carbon intensity, financed emissions, temperature or transition metrics, scenario exposures, and comparison against conventional or ESG indexes. These are useful for measurement and reporting but depend on provider definitions and historical availability.

The academic evidence on ESG data inconsistency is material. Berg, Kölbel, and Rigobon find substantial divergence across six rating agencies, with measurement divergence accounting for 56%, scope divergence 38%, and weight divergence 6% in their decomposition ([Review of Finance](https://doi.org/10.1093/rof/rfac033)). The OECD’s 2025 report similarly emphasizes differences in metric scope and characteristics across rating products ([OECD, Behind ESG Ratings](https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/02/behind-esg-ratings_4591b8bb/3f055f0c-en.pdf)).

**FairBench relationship:** ESG values are inputs to constraints, not ground truth. FairBench must preserve provider, methodology, score vintage, effective date, missingness, and imputation. It should permit sensitivity analysis across providers or rule interpretations.

### SFDR

SFDR Regulation (EU) 2019/2088 establishes sustainability-related disclosure duties for financial-market participants, advisers, and products. Article 8 and Article 9 are different product categories; classification is not a universal sustainability rating and does not itself establish outcome quality. Pre-contractual, website, and periodic disclosures require defined sustainability information and indicators ([EUR-Lex consolidated SFDR text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32019R2088); [ESMA/ESA SFDR Q&A](https://www.esma.europa.eu/sites/default/files/2023-05/JC_2023_18_-_Consolidated_JC_SFDR_QAs.pdf)).

The European Commission’s implementation consultation found major data and methodology pain: 98% of responding financial-market participants who answered the data question reported difficulty obtaining good-quality data, and 80% reported methodological challenges concerning principal adverse impacts in the DNSH context ([European Commission summary](https://finance.ec.europa.eu/document/download/0f2cfde1-12b0-4860-b548-0393ac5b592b_en?filename=2023-sfdr-implementation-summary-of-responses_en.pdf)).

**FairBench relationship:** possible evidence and controls layer. A versioned rule engine can test whether holdings are consistent with binding policy elements and create an audit trail for reported metrics. It cannot, by itself, certify SFDR classification or prove sustainable impact.

### ESMA fund-name guidelines

ESMA’s guidelines address ESG and sustainability-related terms in fund names. The final guidance generally requires at least 80% of investments to meet the environmental/social characteristics or sustainable investment objective suggested by the name, and applies exclusions linked to Paris-aligned benchmark and Climate Transition Benchmark rules by term category ([ESMA final guidelines](https://www.esma.europa.eu/document/guidelines-funds-names-using-esg-or-sustainability-related-terms); [ESMA announcement, 14 May 2024](https://www.esma.europa.eu/press-news/esma-news/esma-guidelines-establish-harmonised-criteria-use-esg-and-sustainability-terms)).

**FairBench relationship:** a feasible-set membership and evidence layer could support monitoring of the 80% basket and complex binding elements. The ESMA rule is a threshold/compliance rule, not a distribution of feasible portfolios.

### SEC fund-name and ESG rules

The SEC’s 2023 amendments to Rule 35d-1 broadened the Names Rule to cover investment characteristics, including ESG-related characteristics. The general structure is an 80% investment policy aligned with the name’s suggested focus, periodic review, remediation after breaches, definitions of name-related terms, and recordkeeping ([SEC final rule](https://www.sec.gov/files/rules/final/2023/33-11238.pdf); [SEC 2025–26 Names Rule FAQs](https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/2025-26-names-rule-faqs)).

The SEC’s 2022 ESG disclosure proposal should not be treated as current law: the SEC formally withdrew the proposed ESG-disclosure rules in June 2025 ([SEC withdrawal notice](https://www.sec.gov/rules-regulations/2025/06/s7-17-22)). The proposal remains a useful design reference for strategy-description, portfolio-information, and emissions-data lineage, but not a binding current regime.

## 5. Comparison matrix: method and standard

| Method/standard | Specific question it solves | Core data needs | What it does well | Limits | FairBench overlap | Integration opportunity |
|---|---|---|---|---|---|---|
| GIPS | How should realized performance be presented fairly and comparably? | Composite membership, portfolio returns, benchmark returns, assets, fees, dispersion, risk, policies and records | Governance, consistency, disclosure, verification | Retrospective; does not create a feasible set or prove skill | Adjacent | Export GIPS-compatible supplemental analysis with benchmark and method disclosure |
| Standard market benchmark | Did the fund beat the market/style/asset-class reference? | Index history, fund returns, currency, fees, dates | Simple, recognizable, investable reference | Can mismatch mandate, constraints, cash, style, or ESG policy | Baseline | Import and reconcile official benchmark; never replace it |
| Parent/ESG index | How did a screened/tilted index perform relative to parent? | Index methodology, constituent history, weights, ESG screens | Passive implementation, transparent rules, low-cost comparison | One deterministic path; not all feasible portfolios | Partial | Use as one reference distribution anchor or comparator |
| Custom/blended benchmark | What deterministic policy mix or exposure reference should be used? | Index components, weights, rebalance rules, data permissions | Policy benchmarks, asset allocation, style matching | Still a single path or governed index; not a probability distribution | Superficial but important distinction | Compare FairBench distribution with custom index; quantify difference |
| Brinson attribution | What allocation/selection/interaction effects created active return? | Portfolio/benchmark weights and group/security returns | Explain realized excess return in intuitive terms | Benchmark-dependent; not suitable for every asset/strategy; not causal | Complementary | Add Brinson outputs next to constraint and within-rule effects |
| Factor attribution | Which systematic factors explain return? | Factor exposures, factor returns, risk model, dates | Separates systematic exposures from residual selection | Model risk; factor choices drive result | Complementary | Report factor distributions and residual manager effect |
| Risk attribution | Which factors/assets/groups explain volatility or tracking error? | Covariance, exposures, benchmark, portfolio weights | Risk budgeting, monitoring, scenario analysis | Depends on risk model and estimates | Complementary | Make risk-model version a run input and compare risk distributions |
| Peer group | How did the manager rank against similar funds? | Returns, classifications, peer membership, historical composition | Market context, selection and monitoring | Survivor/composition/timeliness/mandate mismatch bias | Complementary | Use as external comparison; do not call it a feasible null |
| ESG/carbon analytics | What ESG/carbon exposures and outcomes does the portfolio have? | Provider data, emissions, revenue, ownership, coverage, methodology and dates | Reporting, exposure identification, policy monitoring | Provider disagreement, missing/stale/estimated data | Input layer | Preserve provenance and run provider/rule sensitivity |
| SFDR | What sustainability characteristics/objectives must be disclosed? | Product policy, binding elements, indicators, DNSH/PAI/taxonomy data | Structured disclosure and anti-greenwashing workflow | Disclosure regime, not performance attribution or portfolio optimization | Evidence layer | Generate rule/holdings evidence and exceptions for review |
| ESMA fund-name rules | Does a fund name’s sustainability language match the portfolio? | Name, 80% basket, screens, holdings, policy and periodic review | Harmonized anti-misrepresentation threshold | Does not assess full mandate feasibility or skill | Evidence layer | Test membership and document classifications |
| SEC Names Rule | Does a US fund’s portfolio align with its name’s investment focus? | Asset values, 80% basket, prospectus definitions, quarterly review, derivative treatment | Operational naming-policy compliance | Not a full investment-policy or performance model | Evidence layer | Use rule compiler and exception log; not a legal certification |
| FairBench feasible distribution | How unusual was the real portfolio among portfolios feasible under the same rules? | Frozen universe, point-in-time rules/data, holdings/weights, returns, rebalance/cost model | Makes constraints explicit; gives distributional context | Model realism, rule interpretation, and data lineage are difficult; current prototype is static/equal-weight-heavy | Core | Add-on API/report to existing PMS, warehouse, and attribution stack |

## 6. Vendor capability matrix

“Not publicly documented” means the reviewed public official material did not establish the capability; it is not proof that no private implementation exists. “Counterfactual distribution” is intentionally stricter than “what-if,” “optimization,” “scenario,” or “custom benchmark.”

| Vendor/platform | Customer type | Standard benchmark / custom benchmark | Performance, factor, risk attribution | ESG/carbon | Policy/mandate ingestion | Feasible counterfactual distribution | FairBench overlap | Integration opportunity |
|---|---|---|---|---|---|---|---|---|
| Bloomberg PORT / PORT ESG | Asset managers, owners, risk/performance teams | Strong index and bespoke-index ecosystem; ESG/climate index customization documented | Multi-asset performance, risk, customizable attribution, factor and scenario analytics ([PORT](https://professional.bloomberg.com/products/bloomberg-terminal/portfolio-analytics/)) | ESG scores, carbon/climate, regulatory/SFDR/PAI workflows documented ([scores](https://professional.bloomberg.com/products/bloomberg-terminal/sustainable-finance/scores/)) | Guideline monitoring/alerts documented; arbitrary natural-language IPS ingestion not clearly public | No public evidence found for a general mandate-conditioned portfolio distribution | High in data/analytics; low in exact distributional method | PORT/AIM data export, benchmark and holdings adapter, report plug-in |
| MSCI Performance Attribution / ESG Manager / Barra | Asset managers, pension funds, consultants | Strong custom index and benchmark services | Brinson, fixed-income, factor, risk, custom models, multi-asset ([attribution factsheet](https://www.msci.com/downloads/documents/products/msci-performance-attribution-factsheet.pdf)) | ESG Portfolio Analytics and Carbon Portfolio Analytics compare scores, carbon, exposures and selection/weighting effects ([ESG](https://www.msci.com/documents/10199/242721/MSCI_ESG_Portfolio_Analytics.pdf)) | Restricted lists/compliance; arbitrary IPS ingestion not public | What-if/backtest/simulated-index tools; no explicit probability-distribution engine found | High for data and attribution; distinct feasible-set layer | Consume MSCI holdings, index, ESG and carbon vintages; return distribution alongside Barra outputs |
| Morningstar Direct | Institutional research, consultants, asset/wealth managers | Market indices, blended/custom benchmarks, imported portfolios | Equity and total-portfolio attribution, manager/asset-class hierarchy ([product overview](https://admainnew.morningstar.com/directhelp/General/Product_Overview.htm)) | Portfolio analysis and data available; detailed carbon-attribution scope depends on product/data | Policy benchmarks/custom classifications; arbitrary mandate-text ingestion not public | No explicit general feasible-distribution engine found | Strong peer/benchmark overlap; FairBench’s conditional null remains different | Import Direct holdings/returns and attach FairBench report to Direct research workflow |
| FactSet Portfolio Analysis / Cabot Reveal Plus | Asset managers, owners, consultants, performance teams | Blended, custom, hurdle and market-neutral benchmarks | Equity, fixed income, balanced, macro, risk-based, passive and decision attribution ([portfolio analysis](https://insight.factset.com/hubfs/Resources%20Section/Brochures/portfolio-analysis-brochure.pdf)) | ESG/SDG data and client-sourced scores supported | Custom data/governance; arbitrary natural-language mandate ingestion not public | Closest public match: counterfactual portfolios, Monte Carlo simulations, and return distributions for skill analysis ([Reveal Plus](https://go.factset.com/hubfs/Website/Resources%20Section/Brochures/Cabot_Reveal_Plus.pdf)) | Potential overlap is material; FairBench must differentiate by mandate-text lineage, rule semantics, and institutional constraint fidelity | Coexist with FactSet as a custom analysis engine or data service; test whether Reveal Plus already satisfies the target use case |
| SimCorp / Axioma | Asset managers, asset owners, front-to-back institutions | Benchmark-relative attribution; custom risk models; strong investment operations | Brinson and factor attribution, point-in-time/time-series risk, automated reports ([Axioma Analytics](https://www.simcorp.com/solutions/axioma-solutions/axioma-portfolio-optimizer/axioma-portfolio-analytics)) | ESG metrics/constraints and sustainable portfolio construction documented | Compliance rules exist; public material says mandate-rule setup can be manual | What-if and stress testing; no public general feasible-distribution engine found | Strong operational overlap; FairBench can be a research/reporting extension | Integrate via SimCorp/IBOR exports and return versioned artifacts |
| BlackRock Aladdin | Large asset managers, pension funds, insurers, whole-portfolio institutions | Benchmark-relative construction and projected-benchmark optimization | Multi-asset risk, scenarios, factor/security/sector exposures, performance attribution ([portfolio managers](https://www.blackrock.com/aladdin/benefits/portfolio-managers)) | Aladdin Sustainability and Climate integrate third-party ESG and climate analytics ([sustainability](https://www.blackrock.com/aladdin/platforms/products/aladdin-sustainability)) | Risk Radar/compliance and APIs support rules, exceptions and audit workflows | What-if/scenario/optimization documented; no public general feasible-distribution engine found | High platform overlap; FairBench should be a narrow analytical layer, not a platform | Aladdin APIs/warehouse export; target institutions that cannot justify full Aladdin first |
| Nasdaq eVestment | Asset owners, consultants, manager-research teams | Peer benchmarking and custom peer groups | Performance contributors/detractors, holdings/sector/region/style/factor views | ESG analytics includes carbon, controversies, weapons, human rights, diversity, SDGs ([ESG Analytics](https://www.nasdaq.com/products/evestment/asset-managers/esg-analytics)) | ESG questionnaires and stated approaches; arbitrary IPS ingestion not public | No general mandate-conditioned distribution documented | Strong peer/manager research overlap, weaker feasible-set overlap | Use eVestment peer data as an external comparator and FairBench for mandate conditioning |
| MSCI/S&P/FTSE Russell/Solactive index providers | Asset managers, ETF issuers, asset owners, banks | Custom indices, ESG/climate rules, governed methodologies, backtests | Index-level analytics, not usually manager-level causal attribution | ESG screens, carbon metrics, CTB/PAB and climate methodologies | Methodology and benchmark governance, not arbitrary fund-policy ingestion | Index simulations are deterministic methodology paths; not random feasible distributions | FairBench can consume index constituents and compare index versus feasible set | License point-in-time constituents, weights, methodology and ESG data |
| Internal institutional team | Asset owners, managers, insurers, consultants | Official benchmarks plus policy/custom composites | SQL/Python/Excel, warehouse, custodian and vendor feeds; often bespoke | Provider data, questionnaires, manual normalization, internal estimates | Policy documents and spreadsheets; review/approval often manual | May prototype it, but reproducibility and governance vary | Most likely early buyer/champion if workflow pain is real | Deliver Python package/CLI, signed manifest, API, warehouse tables, and reviewer workflow |

The market is therefore not empty. FactSet’s public Cabot material is the closest documented adjacency because it explicitly discusses counterfactual portfolios and Monte Carlo return distributions. FairBench must test this competitor directly before making a “no one does this” claim. The narrower, still-plausible differentiation is the chain:

`policy text → reviewed rule specification → point-in-time feasible set → distribution → benchmark/attribution report → audit trail`

## 7. Institutional customer and buyer map

| Segment | Example organizations | Economic buyer / budget owner | Daily user / champion | Current workflow and pain | FairBench value | Buying likelihood | Priority |
|---|---|---|---|---|---|---|---|
| Mid-sized pension or insurance asset owner with external equity mandates | Varma/Ilmarinen-like Nordic institution; regional pension funds | CIO, Head of Investment Operations, Head of Investment Performance | Performance analyst, responsible-investment analyst, manager-research analyst | Quarterly manager review across benchmarks, mandate clauses, ESG questionnaires, holdings and carbon data; handoffs across teams | Versioned mandate-to-evidence report; constraint-cost and within-rule context; exceptions | Medium-high after a data pilot; procurement is real | **1** |
| Fund-of-funds / OCIO / investment consultant | Mercer-like consultant, multi-manager allocator | Head of Manager Research or delegated-CIO budget owner | Manager analyst, quantitative researcher | Peer groups and benchmark reports struggle with heterogeneous mandates; repeated client reporting | Reusable mandate-conditioned reports across many managers | High if report production is a billable workflow; competition is high | **1–2** |
| Active ESG asset manager | Parnassus-, Nordea-, Calvert-like manager | COO/CIO/Head of Product or Client Reporting | Performance analyst, ESG analyst, product specialist | Needs to explain underperformance, show consistency with stated policy, and answer due diligence | Defensive evidence for clients and boards; policy drift alerts; not a “skill judge” | Medium; may resist an external score that can be marketed against them | 3 |
| Large sovereign wealth fund / very large pension | NBIM-like institution | CIO/COO/Chief Risk Officer | Specialist internal teams | Sophisticated internal platform, daily transparency, extensive vendor stack | Narrow research module or independent challenge function | Low initial due to procurement and build/buy capabilities | Later lighthouse, not first customer |
| Insurer | Allianz-/Nordic-insurer-like general account | CIO, Chief Risk Officer, Head of Investment Risk | Investment risk, regulatory reporting, ESG data team | Statutory reporting, investment guideline compliance, manager oversight, data timeliness | Evidence and exception layer that connects constraints to portfolio/risk/ESG outputs | Medium; strong pain but high integration/regulatory burden | 2–3 |
| Passive ETF/index provider | iShares/Vanguard/Calvert index-like | Product/Index CIO or methodology team | Index research, product governance | Methodology, tracking error, ESG-screen and index governance | Feasible-set stress testing around methodology; usually not urgent | Low as first customer | Exclude initially |
| ESG data vendor | MSCI/S&P/ISS-like | Product head | Data/product analyst | Data coverage, methodology and provider comparisons | Consumer of data, not primary buyer; use as partner | Medium as integration partner | Partner, not ICP |
| Regulator/compliance team | ESMA/SEC-supervised manager | Chief Compliance Officer / regulator budget owner | Compliance analyst | Fund-name, disclosure, policy evidence and exception reviews | Evidence pack and repeatable controls | Low for a startup sale; high evidentiary value | Design partner only |
| Retail investor | Individual | Individual | Individual | Wants simple ESG and performance comparison | Too complex and data-expensive | Low willingness to pay | Exclude |

### Initial ideal customer profile

An institutional asset owner or fund-of-funds with approximately 5–30 externally managed public-equity mandates, quarterly or monthly manager reviews, an ESG/responsible-investment policy, holdings transparency, and a performance/data team that already uses a warehouse or vendor feed but still reconciles mandate evidence in spreadsheets. The buyer is likely the Head of Investment Performance, CIO office, or Head of Manager Research. The daily user is a performance/ESG/manager analyst. The CIO, board, beneficiaries, and compliance teams benefit but are not necessarily buyers.

Do not start with retail investors, generic sustainable funds, or the largest sovereign wealth funds. Do not assume every asset manager wants an external product that can attribute underperformance to its own constraints.

## 8. Evidence that the problem is real

| Problem | Evidence classification | What is validated | Source |
|---|---|---|---|
| Benchmark and mandate are not always interchangeable | Strongly validated | General finance problem and FairBench motivation | NBIM says returns are measured against a Ministry benchmark while portfolios may differ for special characteristics; GIPS says an appropriate benchmark should reflect mandate/objective/strategy ([NBIM strategy](https://www.nbim.no/en/investments/investment-strategy/), [GIPS owner handbook](https://www.gipsstandards.org/standards/gips-standards-for-asset-owners/gips-standards-handbook-for-asset-owners/)) |
| Peer groups can mismatch mandates | Strongly validated | General manager-evaluation problem; supports simulated/conditional comparison | [CFA Fixing the Peer Group Problem](https://rpc.cfainstitute.org/research/cfa-magazine/2017/fixing-the-peer-group-problem) |
| Mandates and constraints affect measured performance | Strongly validated | Academic problem; not FairBench-specific implementation | [Beber et al., Journal of Empirical Finance](https://img1.wsimg.com/blobby/go/d70dd63d-b877-459f-a3ef-43a1e0215aac/downloads/Mutual%20Fund%20Performance%20-%20Using%20Bespoke%20Benchmarks%20to%20Disentangle%20Mandates%2C%20Constraints%20and%20Skill.pdf?ver=1608176043074) |
| External-manager oversight is recurring and multi-dimensional | Strongly validated | Institutional workflow and buyer pain | [NBIM external mandates](https://www.nbim.no/en/investments/external-mandates/), [NBIM external-management policy](https://www.nbim.no/en/about-us/about-the-fund/governance-structure/policies/external-management--/), [Varma manager survey](https://www.varma.fi/en/this-is-varma/current-issues/news-and-articles/news/2023-q3/varma-looked-into-the-responsibility-of-the-fund-management-companies--almost-all-are-aligned-with-varmas-principles/), [Ilmarinen ownership policy](https://www.ilmarinen.fi/en/about-ilmarinen/investments/responsibility-in-investments/ownership-policy/) |
| ESG/carbon data is incomplete or inconsistent | Strongly validated | ESG workflow and data problem, not automatically FairBench demand | [European Commission SFDR implementation summary](https://finance.ec.europa.eu/document/download/0f2cfde1-12b0-4860-b548-0393ac5b592b_en?filename=2023-sfdr-implementation-summary-of-responses_en.pdf), [Berg et al.](https://doi.org/10.1093/rof/rfac033), [OECD Behind ESG Ratings](https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/02/behind-esg-ratings_4591b8bb/3f055f0c-en.pdf) |
| ESG policies differ materially even under similar labels | Strongly validated | Need for rule-level comparison; not proof of willingness to pay | [SEC ESG proposal statement](https://www.sec.gov/newsroom/speeches-statements/crenshaw-statement-esg-investment-practices-052522), [ESMA fund-name guidelines](https://www.esma.europa.eu/document/guidelines-funds-names-using-esg-or-sustainability-related-terms) |
| Greenwashing/name-alignment pressure exists | Strongly validated | Compliance/reporting problem; FairBench could support evidence but not certify compliance | [ESMA announcement](https://www.esma.europa.eu/press-news/esma-news/esma-guidelines-establish-harmonised-criteria-use-esg-and-sustainability-terms), [SEC Names Rule](https://www.sec.gov/files/rules/final/2023/33-11238.pdf) |
| Institutions need FairBench specifically | Not yet validated | Hypothesis only; requires workflow interviews and real-fund data | No public source establishes demand for this exact product |

## 9. Large real-world examples and exact-versus-proxy feasibility

| Organization/product | Type and public evidence | Benchmark / constraints | Public data | Exact FairBench analysis? | Likely value |
|---|---|---|---|---|---|
| iShares ESG Aware MSCI USA ETF (ESGU) | Passive US ETF; official page reported net assets about $17.9bn on 10 Sep 2026 and 269 holdings | MSCI USA Extended ESG Focus Index; screens include civilian firearms, controversial weapons, tobacco, thermal coal and oil sands ([official page](https://www.ishares.com/us/products/286007/ishares)) | Holdings, index, returns, prospectus and screens are public | **Near-exact for a public-data prototype** if point-in-time index/ESG data is licensed or reconstructed. Exact daily index methodology and historical data may still require permission | Clean test of passive rule compliance and index-vs-feasible comparison; manager effect should be small |
| iShares MSCI World SRI UCITS ETF | Passive global ETF; official factsheet reported about $9.0bn fund net assets and 368 holdings as of Mar/Apr 2026 | MSCI World SRI Select Reduced Fossil Fuel Index; Article 8; SRI screens and sector representation ([official page/factsheet](https://www.ishares.com/uk/professional/en/literature/fact-sheet/susw-ishares-msci-world-sri-ucits-etf-fund-fact-sheet-en-gb.pdf?siteEntryPassthrough=true&switchLocale=y)) | Holdings, benchmark, prospectus, SFDR web disclosure, returns | **Near-exact/proxy mix.** The stated index rules are public, but the provider’s historical ESG/controversy data and effective dates may be proprietary | Strong European/ESMA/SFDR demonstration |
| Vanguard ESG U.S. Stock ETF (ESGV) | Passive US ETF; official page describes FTSE US All Cap Choice Index and showed 1,228 holdings on 31 Jul 2026 ([official page](https://advisors.vanguard.com/investments/products/esgv/vanguard-esg-us-stock-etf)) | FTSE US All Cap Choice Index with provider-defined ESG screens | Public holdings, benchmark and returns; complete historical screen data may not be public | Proxy unless historical FTSE screen vintages are obtained | Shows that “ESG ETF” does not mean one common rule set |
| Parnassus Core Equity Fund (PRBLX/PRILX) | Active US equity fund; official holdings page lists holdings and reports five-year tracking error/active share relative to S&P 500 as of Jun 2026 ([holdings](https://www.parnassus.com/parnassus-mutual-funds/core-equity/full-holdings); [prospectus](https://content.parnassus.com/094ee837-48bb-001c-6959-053670061ebf/f9f934c0-17af-4e4e-8e65-a67059f7379f/Parnassus%20Core%20Equity%20Fund%20Summary%20Prospectus%20May%201%2C%202023.pdf)) | S&P 500 comparison; ESG factors and screens; some fossil-fuel-free/ESG language | Public prospectus, current holdings, returns and benchmark; full internal scoring, research, trades and historical rule vintages are not public | **Proxy active case.** Exact internal mandate and point-in-time ESG data require manager cooperation | Best public active demonstration of constraint-vs-selection question, but unsafe to claim exact attribution |
| Nordea Global Climate and Environment Fund | Active thematic fund; official Nordea materials say it invests in companies offering climate solutions ([Nordea](https://www.nordea.com/en/sustainability/our-products-with-sustainable-focus)) | Thematic climate mandate; detailed eligibility and scoring need prospectus/manager data | Public product description, some holdings/returns depending share class; full rule history not guaranteed | Proxy without manager data | Nordic and thematic example; less clean for broad constraint attribution |
| Calvert US Large-Cap Core Responsible Index Fund | Passive responsible index fund; official page reported fund assets about $6.5bn on 29 Sep 2026 | Calvert US Large-Cap Core Responsible Index; Russell 1000 benchmark; annual reconstitution and quarterly rebalance ([Calvert](https://www.calvert.com/investment-solutions/mutual-funds/us-equity/calvert-us-large-cap-core-responsible-index-fund.shareclass.I.html)) | Public benchmark, index description, holdings/returns and responsible-investment principles | Near-exact for disclosed methodology; historical data vintage remains a limitation | Useful for index-methodology and benchmark distinction |
| Calvert Responsible Allocation Funds | Multi-asset responsible allocation products | Official materials disclose internally constructed blended benchmarks with fixed index weights ([Calvert material](https://www.calvert.com/content/dam/im/assets/publication/sales-material/sales-idea/27581.pdf)) | Public blend definitions and returns | Exact for benchmark blend; not enough for all portfolio constraints | Demonstrates a custom benchmark is a deterministic blend, not a feasible distribution |
| NBIM Government Pension Fund Global external mandates | Sovereign wealth fund with external managers; NOK 1.062tn/5.0% externally managed at end-2025, 111 mandates and 103 organizations ([NBIM](https://www.nbim.no/en/investments/external-mandates/)) | Mandate-specific benchmarks, limits, ESG expectations, daily transparency, external manager contracts | Public policies and aggregate reports; individual holdings/transactions and full mandate data may be restricted | **Client-data ideal; not public exact** | Strongest institutional buyer archetype, but very difficult first sale |
| CalPERS | Large public pension; official sustainable-investment reviews track asset-class benchmarks, climate risk, ESG diligence and external-manager governance ([CalPERS](https://www.calpers.ca.gov/investments/sustainable-investments-program/esg-integration)) | Asset-class benchmarks plus sustainability KPIs and policy constraints | Public policies/reports, not all manager-level holdings/rule data | Proxy publicly; exact with internal data | Validates recurring multi-team workflow |
| Varma | Finnish pension insurer; principles apply across asset classes; 2025 review reported 43.7% climate allocation and a goal of 50% by 2027 ([policy](https://www.varma.fi/en/this-is-varma/about-us/how-we-do-things/principles-for-responsible-investment/); [2025 review](https://www.varma.fi/en/this-is-varma/current-issues/news-and-articles/news/2026-q1/responsible-investment-review-2025-varmas-climate-targets-progressing-ahead-of-schedule/)) | Negative screens, due diligence, engagement, climate targets, fund-manager oversight | Public policy, review, manager survey; complete holdings/manager mandate details may be restricted | Proxy publicly; exact with internal data | Strong Nordic design-partner archetype |
| Ilmarinen | Finnish pension insurer; responsible investment policy, climate plan, ownership policy and external-manager oversight ([responsible investment](https://www.ilmarinen.fi/en/about-ilmarinen/investments/responsibility-in-investments/)) | ESG integration, climate and active ownership across asset classes | Public policies/reports; detailed manager data not all public | Proxy publicly; exact with client data | Strong Nordic design-partner archetype |

### Recommended cases

1. **Passive public-data case:** ESGU, because its benchmark, screens, holdings, returns, and prospectus are easy to explain. Use a frozen MSCI USA universe and explicitly reconstruct only documented index rules. Report any missing historical ESG vintage as a proxy.
2. **Active public-data case:** Parnassus Core Equity, because it has a recognizable benchmark, public holdings, active share/tracking error, and explicit ESG-screen language. The report must say “proxy mandate,” not “exact fund mandate.”
3. **Ideal client-provided pilot:** a Varma/Ilmarinen/NBIM-like external equity mandate with daily or monthly holdings, the signed investment management agreement, policy amendments, manager benchmark, ESG/carbon data, and transaction/cost records.

## 10. Killer empirical comparison

### Study design

Pre-register a 20–50 fund, 8–12 quarterly-date study, ideally spanning both passive and active funds. For each date:

1. Freeze the investable universe, security master, benchmark constituents/weights, mandate version, ESG/carbon vintage, risk model, and all information available at the decision date.
2. Obtain actual holdings and weights as-of the date, not only a later restatement. Mark whether holdings are snapshot or transaction-level.
3. Have two human analysts independently encode the policy into a gold-standard rule set. Compare FairBench extraction with that set; require human approval before analysis.
4. Run the industry analysis: official benchmark excess return, Brinson allocation/selection, factor and risk attribution, tracking error, active share where available, peer rank, and ESG/carbon comparison.
5. Run FairBench on the same date: sample feasible portfolios under the gold-standard rules, with explicit weighting/rebalance/cost assumptions. Calculate constraint effect, within-rule effect, percentile, interval, and rule compliance.
6. Evaluate next-period forward returns, never the same information window used to set rules or select assets.
7. Repeat monthly or quarterly with versioned rules, turnover, fees, spread/impact assumptions, and currency treatment where available.

### Statistical calibration

Create pseudo-funds by sampling from the same feasible set. Under the null, their percentiles should be approximately uniform. Run planted within-rule tilts to test whether the system recovers a known signal. Measure coverage of nominal 95% intervals. Repeat under alternative rule interpretations, ESG providers, and weighting schemes.

### Required report

- Executive summary with exact/proxy label.
- Traditional benchmark and Brinson/factor results.
- Mandate-adjusted feasible distribution.
- Difference between benchmark interpretation and mandate-conditioned interpretation.
- Quarter-by-quarter within-rule effect, with uncertainty.
- Cumulative constraint-cost chart.
- Percentile stability and sensitivity to rules/providers/weights.
- Rule changes and unmapped clauses over time.
- Holdings versus trade-data limitations.
- Reproducibility manifest: data snapshots, vendor/methodology versions, mandate hash, compiler/model version, random seeds, run time, and reviewer sign-off.

### Success criteria

- At least 95% sentence-level rule fidelity after review in the pilot sample.
- Calibrated null percentiles and interval coverage.
- Recovery of planted effects within pre-specified tolerance.
- Stable attribution signs across adjacent dates and reasonable rule definitions.
- Clear incremental decision value over official benchmark, peer, and standard attribution outputs.
- A target institution agrees that the report changes a manager-review, mandate-renewal, or escalation decision.

### Falsification criteria

- Existing vendor/internal tools answer the same question with no meaningful loss of transparency.
- Analysts do not use the decomposition in manager review.
- Results change sign under minor reasonable rule interpretations.
- Exact mandate/point-in-time ESG data cannot be obtained at acceptable cost.
- Feasible portfolios are mathematically valid but economically unrealistic.
- The distribution adds no information beyond a factor model plus peer group.
- Active managers or asset owners reject the method as unfair or unreviewable.
- Report production costs more time than the current process it is meant to improve.

## 11. Production-readiness audit of the repository

### Current prototype: supported or partially supported

- Natural-language mandate to structured rule specification with source quotes, mapped/trivial/unmapped status, deterministic compilation, and rule checking ([mandate compiler](fairbench/mandate.py), [rules](fairbench/rules.py)).
- Binary feasibility constraints including holdings count, exclusions, sector/group limits, ESG/carbon averages, minimum groups, and some risk limits.
- Classical rejection sampling and small simulator-based quantum backends.
- Percentile, median, confidence-interval and constraint/manager decomposition for synthetic/static windows.
- Equal-weight baseline and limited post-sampling weighting schemes.

### Major gaps

| Area | Current state | Classification |
|---|---|---|
| Unequal weights | Binary support is central; position/group/sector logic often converts weights to name counts; explicit weights are not fully checked as weighted feasibility | **Critical blocker for general production; acceptable for equal-weight prototype** |
| Turnover/rebalancing | Static membership over a window; rebalance/cost path is not institutional-grade | **Critical blocker for longitudinal use** |
| Transaction costs, spread, impact, taxes, liquidity, capacity | Not modeled as a complete path | **Required for production; acceptable to omit in narrow prototype if labeled** |
| Cash, shorts, leverage, derivatives, corporate actions | Not fully modeled | **Required for broad institutional deployment** |
| Changing universes | No robust point-in-time membership/security-master layer | **Critical blocker** |
| Changing ESG data | No provider/vintage/effective-date lineage | **Critical blocker for ESG claims** |
| Missing data | Some fields optional; no comprehensive coverage/confidence/imputation governance | **Required for production** |
| Mandate versioning | Compiler output exists, but no immutable document/spec/version/approval registry | **Required for production** |
| Exact risk model | Limited; equal-weight universe assumptions appear in example/rules | **Required for risk attribution** |
| Statistical uncertainty | Monte Carlo sampling uncertainty only; no model/data/mandate uncertainty, multi-period correction, or multiple-testing control | **Critical for manager-skill claims** |
| Survivorship/look-ahead bias | Not yet demonstrated with point-in-time data | **Critical blocker for historical validation** |
| Feasibility realism | Uniform feasible binary set is a valid mathematical object, but may not be the manager’s feasible economic set | **Required research question** |
| Sampler scaling | Rejection cost grows as feasible fraction shrinks; no robust weighted/path sampler or MCMC diagnostics | **Production performance risk** |
| Integrations | CSV/JSON/Python; no PMS/OMS/custodian/vendor adapters, RBAC, report registry, or API | **MVP integration work** |
| Auditability | Source quotes and deterministic rules are strong; full data lineage and approvals absent | **MVP/production requirement** |

The code should explicitly avoid saying that a fund’s “manager effect” is manager skill. Use “within-rule return difference” until a multi-period, out-of-sample, data-aware study supports a stronger interpretation.

## 12. Product form and business model

### Recommended form

1. **Pilot:** analyst-operated Python package/CLI plus signed input package and PDF/HTML report.
2. **MVP:** batch API and warehouse tables: mandate version, rule set, universe snapshot, data provenance, feasible samples, results, exceptions, and review state.
3. **Production:** connectors for custodian/PMS/warehouse and major data vendors, human review queue for mandate extraction, schedule, RBAC, immutable run manifest, and report templates.

FairBench should be an add-on/reference-model service, not a portfolio optimizer, order-management system, ESG data vendor, or regulatory certification product.

### Who pays

- Primary economic buyer: Head of Investment Performance, CIO office, Head of Manager Research, or investment-risk leader.
- Daily user: performance analyst, quantitative manager researcher, or responsible-investment analyst.
- Champion: analyst who currently reconciles mandate documents, benchmark definitions, ESG data, and quarterly holdings in spreadsheets.
- Beneficiary: CIO, investment committee, trustees/board, compliance, and beneficiaries.

Pricing below is a hypothesis, not market data: a paid pilot in the low five to low six figures depending on data/integration scope; annual enterprise subscription in the low-to-mid six figures for recurring mandates and reports; consulting/data-onboarding fees initially. Validate willingness to pay against hours saved, avoided rework, manager-review decisions, or governance risk reduced.

## 13. Market sizing without abusing AUM

Do not use sustainable-fund AUM as software TAM. AUM is the capital managed by products, not software revenue or customer count. Use a bottom-up account model:

- Identify the number of institutional allocators in the chosen geography that have recurring external-manager reviews and material ESG/mandate reporting.
- Exclude organizations whose existing platforms already provide the needed workflow unless FairBench can be a plug-in.
- Estimate the number of mandates per account, reports per year, and data onboarding burden.
- Multiply qualified accounts by validated annual contract value.

Illustrative scenario, explicitly an assumption: 20–50 qualified initial accounts at €100k–€300k annual contract value implies a €2m–€15m initial serviceable software/research opportunity. A consultant or data-vendor channel could expand the account count but would change pricing and margin. This is not a forecast and should be replaced by interview evidence.

## 14. Revised positioning

**One sentence:** FairBench turns a fund’s documented mandate into an auditable feasible-portfolio reference distribution, so allocators can separate benchmark/constraint effects from within-mandate portfolio decisions.

**Tagline:** *Measure manager decisions against the rules they were actually hired to follow.*

**30-second pitch:** Existing benchmark and attribution systems tell you whether a fund beat an index and which sectors, securities, or factors drove the gap. FairBench adds a mandate-conditioned reference set: portfolios drawn from the same frozen universe and satisfying the same reviewed constraints. It reports how unusual the fund’s outcome was within that set, with rule-level evidence and uncertainty. It complements existing systems; it does not call a percentile proof of skill.

**90-second pitch:** Institutional investors evaluate managers against indexes, peers, factor models, ESG metrics, and policy limits, but those references often answer different questions. FairBench links the mandate text, benchmark, holdings, ESG data, and returns into one versioned analysis. It distinguishes a broad-benchmark gap from the return distribution created by the fund’s own constraints, then reports the fund’s within-rule position. This is useful when a manager is constrained by exclusions, carbon targets, country/sector limits, liquidity or tracking-error rules and the investment committee needs to decide whether a shortfall reflects the mandate, implementation, or a decision inside the allowed set. The analysis is only as credible as its point-in-time data, rule interpretation, weights, turnover, and costs, so those are first-class report outputs.

**Pension fund pitch:** “Give the investment committee a repeatable, evidence-backed answer to whether an external manager’s underperformance was driven by the policy we imposed or by decisions inside that policy.”

**Consultant pitch:** “Turn bespoke mandate and manager-monitoring work into a versioned, reusable report instead of rebuilding the same spreadsheet for every client.”

**Active-manager pitch:** “Show clients and boards how the strategy performed relative to the opportunity set created by its disclosed constraints, with the assumptions visible and reviewable.”

**Compliance/ESG pitch:** “Create a rule-level evidence trail from stated policy and fund name to holdings, data vintages, exceptions, and remediation—not a black-box ESG score.”

## 15. Quantum positioning

Separate three products:

1. **Finance product:** mandate-conditioned performance and policy evidence. It must work classically.
2. **Classical algorithm:** better feasible-set sampling, weighting, sequential constraints, calibration, and data lineage. This creates practical product value.
3. **Quantum research:** a possible future accelerator for very sparse or fragmented feasible sets under an exact oracle.

Safe wording from the current repository is:

> In noiseless simulation, for a restricted equal-weight, buy-and-hold percentile statistic and an exact feasibility/return oracle, amplitude estimation uses fewer oracle queries for a target precision than classical sampling. This is not a demonstrated wall-clock advantage on current hardware and is not a production speedup.

Do not lead with “quantum speedup,” “quantum benchmark,” or “quantum manager skill.” Put the quantum result in an appendix or research track. A buyer should be able to approve FairBench because the report improves governance and manager evaluation even if quantum hardware never becomes useful.

## 16. Skeptical judge questions and honest answers

1. **Is this just a custom benchmark?** No. A custom benchmark is usually a deterministic index or blend. FairBench estimates a distribution over portfolios satisfying explicit constraints. The two may be compared but are not equivalent.
2. **Is the feasible distribution investable?** Not automatically. It is only economically meaningful after modeling weights, turnover, liquidity, costs, capacity, and the manager’s actual decision process.
3. **Does a high percentile prove manager skill?** No. It is a within-rule historical result. Skill requires out-of-sample, multi-period evidence and controls for factors, costs, selection, and multiple testing.
4. **Why not use Brinson?** Use Brinson. It answers a different, complementary question: how did realized allocation/selection decisions explain benchmark-relative return?
5. **Why not use a factor model?** Use it too. A factor model helps prevent a hidden factor tilt from being mislabeled as manager effect.
6. **Does GIPS require FairBench?** No. GIPS governs fair performance presentation and benchmark disclosure; it does not require feasible-set sampling.
7. **Does SFDR require it?** No. FairBench could support evidence and controls for SFDR-related policies but cannot certify Article 8/9 status or impact.
8. **Does ESMA’s 80% rule make this necessary?** It creates a concrete monitoring use case, not a mandate for FairBench. Existing compliance systems may already cover the threshold.
9. **What does SEC ESG regulation require today?** The Names Rule is operative; the 2022 enhanced ESG-disclosure proposal was withdrawn in 2025. Do not market the proposal as current law.
10. **What does Bloomberg already do?** Portfolio/risk/performance attribution, ESG, carbon, scenarios, custom index workflows, and data integration. FairBench should plug into it or challenge a narrower question.
11. **What does FactSet already do?** It documents counterfactual portfolios and Monte Carlo return distributions for skill analysis, making it the closest public adjacency. FairBench must prove rule/mandate lineage and decision value beyond it.
12. **What is unique?** The proposed combination of reviewed natural-language policy mapping, point-in-time feasible-set generation, and auditable constraint/within-rule reporting. It is plausible, not yet verified as unique in the whole market.
13. **Where does ESG data come from?** A licensed provider, manager-supplied data, public disclosures, or index methodology. Every result must show provider, vintage, effective date, coverage, and imputation.
14. **Can public data validate an active fund exactly?** Usually no. Parnassus is a useful proxy, but internal scores, decisions, trades, and historical rule versions are not fully public.
15. **What if the mandate says “best in class” but not a numeric threshold?** Treat it as ambiguous, route to human review, and show the rule interpretation sensitivity. Do not silently invent a cutoff.
16. **What if the manager violates its stated rule?** Report that the fund is outside the tested null and stop calling the result a within-mandate comparison until the rule or data is reconciled.
17. **How do you avoid look-ahead bias?** Freeze universe, holdings, ESG, mandate, benchmark, and model data at each decision date; evaluate only later returns.
18. **What about the same asset being eligible in hindsight but not then?** Use point-in-time security master and data vintages; otherwise the analysis is a retrospective proxy.
19. **Why would a manager buy a tool that may say it underperformed?** The first buyer should be the allocator/consultant; for managers, position it as transparent evidence and client-reporting support, not a punitive ranking.
20. **Could internal teams build this?** Yes. That is a competitor. FairBench must reduce repeat work, preserve auditability, and be cheaper/faster/more neutral than internal maintenance.
21. **What is the economic buyer?** Probably the Head of Investment Performance, Manager Research, CIO office, or investment-risk leader—not a generic ESG team.
22. **Can a random portfolio represent the manager’s opportunity set?** Only under an explicit null. A uniform feasible set is a mathematical reference, not a behavioral model.
23. **What if the feasible set is disconnected or tiny?** Diagnose feasible fraction, sampler mixing/effective sample size, coverage, and sensitivity. Escalate to exact/optimization methods or label the result unverified.
24. **Does the quantum circuit solve the production problem?** No. Current results are synthetic/noiseless/query-complexity research; the classical product comes first.
25. **What would falsify FairBench?** A real pilot shows no decision value over existing vendor/internal tools, unstable outputs under reasonable rules, or data costs too high for recurring use.

## 17. Kill criteria and next ten actions

### Kill or narrow if

- Three target institutions say the question is interesting but not decision-relevant.
- Existing FactSet/Bloomberg/MSCI/Aladdin/internal workflows already answer it at adequate quality and cost.
- Active-manager results are too sensitive to reasonable rule interpretations.
- Point-in-time ESG and holdings data cannot be sourced at a viable price.
- Feasible portfolios fail basic liquidity/turnover/capacity realism checks.
- No buyer will provide mandate documents and historical holdings.
- Quantum remains the only persuasive feature.

### Next ten actions

1. Interview 10 institutional performance/manager-research/ESG analysts about the last quarterly manager-review workflow.
2. Obtain one real mandate, one benchmark file, 8–12 holdings dates, ESG/carbon vintages, and returns under NDA.
3. Add immutable run manifests, mandate version/effective dates, source links, provider/methodology fields, and reviewer approval.
4. Build weighted-feasibility support; stop translating all weight constraints into counts.
5. Add point-in-time universe, security-master, delisting, corporate-action, and survivorship controls.
6. Add turnover, transaction-cost, cash, liquidity, and rebalance-path modeling.
7. Reproduce ESGU exactly or explicitly proxy it and publish the data audit.
8. Run Parnassus as an explicitly proxy active case and compare against standard Brinson/factor/peer outputs.
9. Benchmark FairBench against FactSet Cabot Reveal Plus and one internal spreadsheet workflow; quantify incremental decision value.
10. Relegate quantum to an appendix until a fair, reproducible hardware or fault-tolerant resource comparison changes the economics.

## 18. Research ledger

### High-quality primary/professional sources used

- [GIPS Standards](https://www.gipsstandards.org/standards/), [GIPS benchmark guidance](https://www.gipsstandards.org/wp-content/uploads/2023/08/gs_benchmarks_firms.pdf), [GIPS asset-owner handbook](https://www.gipsstandards.org/standards/gips-standards-for-asset-owners/gips-standards-handbook-for-asset-owners/)
- [CFA peer-group research](https://rpc.cfainstitute.org/research/cfa-magazine/2017/fixing-the-peer-group-problem)
- [CFA Research Foundation performance attribution review](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf)
- [ESMA fund-name guidelines](https://www.esma.europa.eu/document/guidelines-funds-names-using-esg-or-sustainability-related-terms)
- [SEC Names Rule](https://www.sec.gov/files/rules/final/2023/33-11238.pdf), [SEC Names Rule FAQs](https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/2025-26-names-rule-faqs), [SEC ESG proposal withdrawal](https://www.sec.gov/rules-regulations/2025/06/s7-17-22)
- [EUR-Lex SFDR](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R2088), [SFDR Q&A](https://www.esma.europa.eu/sites/default/files/2023-05/JC_2023_18_-_Consolidated_JC_SFDR_QAs.pdf), [European Commission implementation summary](https://finance.ec.europa.eu/document/download/0f2cfde1-12b0-4860-b548-0393ac5b592b_en?filename=2023-sfdr-implementation-summary-of-responses_en.pdf)
- [Bloomberg PORT](https://professional.bloomberg.com/products/bloomberg-terminal/portfolio-analytics/), [Bloomberg sustainability scores](https://professional.bloomberg.com/products/bloomberg-terminal/sustainable-finance/scores/)
- [MSCI Performance Attribution](https://www.msci.com/downloads/documents/products/msci-performance-attribution-factsheet.pdf), [MSCI ESG Portfolio Analytics](https://www.msci.com/documents/10199/242721/MSCI_ESG_Portfolio_Analytics.pdf)
- [Morningstar Direct product overview](https://admainnew.morningstar.com/directhelp/General/Product_Overview.htm), [Total Portfolio Attribution](https://advisor.morningstar.com/documentation_component_totalportfolioattribution.htm)
- [FactSet Portfolio Analysis](https://insight.factset.com/hubfs/Resources%20Section/Brochures/portfolio-analysis-brochure.pdf), [Cabot Reveal Plus](https://go.factset.com/hubfs/Website/Resources%20Section/Brochures/Cabot_Reveal_Plus.pdf)
- [SimCorp/Axioma Portfolio Analytics](https://www.simcorp.com/solutions/axioma-solutions/axioma-portfolio-optimizer/axioma-portfolio-analytics)
- [BlackRock Aladdin portfolio managers](https://www.blackrock.com/aladdin/benefits/portfolio-managers), [Aladdin Sustainability](https://www.blackrock.com/aladdin/platforms/products/aladdin-sustainability)
- [NBIM external mandates](https://www.nbim.no/en/investments/external-mandates/), [NBIM external-management policy](https://www.nbim.no/en/about-us/about-the-fund/governance-structure/policies/external-management--/)
- [CalPERS ESG integration](https://www.calpers.ca.gov/investments/sustainable-investments-program/esg-integration), [CalPERS sustainable-investment review](https://calpers.ca.gov/docs/board-agendas/202411/invest/item05d-01_a.pdf)
- [Varma responsible-investment principles](https://www.varma.fi/en/this-is-varma/about-us/how-we-do-things/principles-for-responsible-investment/), [Varma manager survey](https://www.varma.fi/en/this-is-varma/current-issues/news-and-articles/news/2023-q3/varma-looked-into-the-responsibility-of-the-fund-management-companies--almost-all-are-aligned-with-varmas-principles/)
- [Ilmarinen responsible investment](https://www.ilmarinen.fi/en/about-ilmarinen/investments/responsibility-in-investments/)
- [Berg, Kölbel, Rigobon](https://doi.org/10.1093/rof/rfac033), [OECD Behind ESG Ratings](https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/02/behind-esg-ratings_4591b8bb/3f055f0c-en.pdf)
- [Beber, Brandt, Cen, Kavajecz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3211588)
- [ESGU official page](https://www.ishares.com/us/products/286007/ishares), [iShares World SRI factsheet](https://www.ishares.com/uk/professional/en/literature/fact-sheet/susw-ishares-msci-world-sri-ucits-etf-fund-fact-sheet-en-gb.pdf?siteEntryPassthrough=true&switchLocale=y), [Vanguard ESGV](https://advisors.vanguard.com/investments/products/esgv/vanguard-esg-us-stock-etf), [Parnassus Core Equity](https://www.parnassus.com/parnassus-mutual-funds/core-equity/full-holdings), [Calvert responsible index](https://www.calvert.com/investment-solutions/mutual-funds/us-equity/calvert-us-large-cap-core-responsible-index-fund.shareclass.I.html), [Nordea sustainable products](https://www.nordea.com/en/sustainability/our-products-with-sustainable-focus)

### Contradictions and unresolved questions

- The market has several forms of “counterfactual” or “simulation.” Public documentation is not uniform. FactSet documents counterfactual/Monte Carlo return distributions, while other vendors document scenarios, optimizers, backtests, and what-if analysis. These must not be collapsed into one capability.
- Official vendor pages are marketing/product descriptions, not independent proof of implementation quality, customer adoption, or data lineage. A procurement pilot is required.
- Asset-owner policies prove recurring oversight and data work, not willingness to buy FairBench.
- Public fund documents reveal stated policy and some screens, not all internal investment decisions or point-in-time data vintages.
- SFDR and ESMA/SEC rules create compliance workflows but do not imply that a feasible-distribution product is legally required.

### Claims requiring customer interviews

- Who owns the budget: performance, manager research, CIO, sustainability, risk, or compliance.
- Whether quarterly reports actually include a constraint-versus-manager decomposition.
- How much analyst time is spent reconciling mandates, benchmarks, ESG data, and holdings.
- Whether FairBench would be accepted as an independent manager-evaluation reference.
- Willingness to share signed mandates and historical holdings.

### Claims requiring real fund data

- Rule-extraction accuracy on live mandates.
- Feasible-set calibration and realism.
- Stability across dates, providers, and weighting schemes.
- Incremental decision value relative to existing vendor/internal analytics.
- Any inference about manager skill.

## Decisive answer

If I were the FairBench team, I would now focus on **a versioned mandate-to-attribution audit layer**, sell to **institutional asset owners, fund-of-funds, and consultants with recurring external-manager reviews**, validate it using **a point-in-time multi-quarter pilot against official benchmark/Brinson/factor/peer outputs**, and stop saying **that custom benchmarks are feasible-portfolio distributions or that current quantum results show a practical speedup**.
