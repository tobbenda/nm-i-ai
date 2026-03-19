
### Principles
- **Feynman**: Understand the problem before solving it — but don't over-understand. Executing teaches faster than reading.
- **Pareto (80/20)**: The simplest solution gets you most of the way. Add complexity only where error analysis tells you to.
- **Speed to first baseline is everything.**
- **Try simple solutions first**

### Steps

**1. Understand the problem (~10 min)**
- State the task in one sentence: "Given ___, predict/classify/generate ___"
- What is the scoring metric? What does it reward/punish?
- What is the input format? Output format?
- What are the constraints?


**2. Understand the repo/architecture (~5 min)**
- API spec? Which files to edit? How does submission work?
- *Tip: Draw an ASCII schematic of the pipeline*

**3. Visualize data samples**
- Look at actual inputs. Does the data match your assumptions from step 1?
- Is it evenly distributed? (class imbalance)

**4. Baseline submit**
- Simplest possible end-to-end solution. Get a score on the board. Clock the time.

**5. Light local observability rig**
- Local evaluator + error visualization. Apply to baseline: what's actually going wrong?
- *Error analysis!*

**Notes**
- Time the time per image. Test has 254, but we dont know how many will be during eval

**6. Research the space (~25 min)**
- What problem category is this? What solution types dominate?
- Don't over-analyze! Note down 2–7 different models/algorithms/approaches to try rapidly.

**7. Implement simple versions**
- For each approach from step 6:
  - Quickly implement the default-parameter version
  - Look at the evaluation
  - Do 1–2 tunings max
  - Always log experiment + result to the experiment log

**8. Tune best models or ensemble**
- Continue tuning the winner, or combine top 2–3 via ensemble
- always consider whether your top 2 approaches complement each other's weaknesses.
- *Tip: Union voting when scoring metric punishes missing a class. Simple average otherwise.*




Validate the input - And return default valid response if its not valid
Have a latency kill switch - Return valid default response if solution too slow