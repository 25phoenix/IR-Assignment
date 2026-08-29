You are an AI coding/research agent. Build a COMPLETE, RUNNABLE, REPRODUCIBLE implementation for the following Computer Science assignment:

# Assignment 2: Zipf's Law and Tokenization

Investigate whether tokens follow Zipf's law.

Using Wikipedia/corpus data, analyze the Zipf distribution of tokens across different languages such as English, Hindi, and Arabic.

Compare the token distributions produced by different tokenizers/models (e.g., LLaMA, Qwen, Kimi) and study how their vocabulary sizes and tokenization strategies differ.

Analyze how language and tokenization affect the observed Zipf-law behavior.

Investigate whether there is a "sweet spot" for vocabulary size when training a tokenizer for a given language.

Explore whether the point at which Zipf-law patterns emerge or stabilize can provide an indication of when to stop tokenizer training / how to choose vocabulary size.

Investigate whether an algorithm or analytical criterion can be developed to determine an appropriate vocabulary size for a given language and corpus when training a language model.

The uploaded assignment document is the authoritative specification. Read it carefully before implementing anything.

==================================================
0. PRIMARY OBJECTIVE
==================================================

Do NOT create a superficial demo.

Build a complete experimental research pipeline that allows us to answer:

1. Does language affect Zipf-law behavior?
2. Does tokenizer choice affect Zipf-law behavior?
3. Does tokenizer vocabulary size affect Zipf-law behavior?
4. Can we identify a vocabulary-size "sweet spot"?
5. Can we formulate and experimentally evaluate an analytical criterion for selecting vocabulary size?

The final project must be executable end-to-end and must produce data, plots, tables, and a research report summarizing the findings.

Do not fabricate experimental results.

If an experiment cannot be run because of unavailable data, compute, model access, or dependencies, clearly report that limitation and provide a working fallback.

==================================================
1. FIRST INSPECT THE PROJECT
==================================================

Before writing code:

1. Inspect the entire project directory.
2. Locate the existing source files, README, requirements, configuration files, notebooks, and data.
3. Read the assignment PDF.
4. Determine the current environment:
   - Python version
   - OS
   - available CPU
   - available GPU if any
   - available RAM
5. Check which Python packages are already installed.
6. Do NOT unnecessarily install large dependencies.
7. Prefer lightweight and reproducible solutions.
8. Never download huge language-model weights merely to perform tokenizer experiments if the tokenizer files/configuration can be used independently.

After inspection, create a short implementation plan in `IMPLEMENTATION_PLAN.md`.

Then implement the project.

==================================================
2. PROJECT ARCHITECTURE
==================================================

Create a clean structure similar to:

project/
│
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── english/
│   │   ├── hindi/
│   │   └── arabic/
│   └── processed/
│
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── tokenizer_comparison.py
│   ├── tokenizer_training.py
│   ├── frequency_analysis.py
│   ├── zipf_analysis.py
│   ├── stability_analysis.py
│   ├── sweet_spot.py
│   └── utils.py
│
├── experiments/
│   ├── language_comparison.py
│   ├── existing_tokenizers.py
│   └── vocabulary_sizes.py
│
├── results/
│   ├── raw/
│   ├── tables/
│   ├── plots/
│   └── final/
│
└── report/
    └── report.md

You may modify this structure if a better architecture is appropriate, but keep the project modular.

==================================================
3. CORPUS COLLECTION
==================================================

Use Wikipedia or another publicly available corpus.

The main languages must be:

- English
- Hindi
- Arabic

Make the data acquisition reproducible.

Do NOT assume the user has already downloaded the corpora.

Implement a corpus acquisition/preparation pipeline.

If direct automatic download from Wikipedia is feasible, implement it.

If full Wikipedia dumps are too large, use a reproducible subset or a standard downloadable corpus with clear documentation.

IMPORTANT:

The corpora must be made as comparable as reasonably possible.

Record:

- language
- source
- number of documents
- characters
- words
- raw size
- processed size

Do not silently compare a 100-million-token English corpus against a 500-thousand-token Hindi corpus and attribute every difference to language.

Provide a configurable corpus-size limit.

For development/testing, support a small sample mode.

For final experiments, allow a larger mode.

==================================================
4. PREPROCESSING
==================================================

Implement consistent preprocessing.

Document every preprocessing operation.

At minimum consider:

- Unicode normalization
- whitespace normalization
- handling of punctuation
- handling of URLs
- handling of markup
- handling of numbers
- casing where appropriate

DO NOT blindly lowercase languages where casing itself may be meaningful.

The preprocessing pipeline must be identical in spirit across languages while respecting language-specific Unicode characteristics.

Save preprocessing metadata.

==================================================
5. TOKEN DEFINITIONS
==================================================

Clearly distinguish:

1. word-level tokens
2. character-level units
3. subword tokenizer tokens

Do not mix them.

The primary experiment must analyze tokenizer-produced tokens.

The word-level analysis may be used as a baseline.

Document exactly what constitutes a token in each experiment.

==================================================
6. ZIPF ANALYSIS
==================================================

Implement a robust Zipf analysis module.

For a tokenized corpus:

1. Count token frequencies.
2. Sort tokens by descending frequency.
3. Assign rank:
   rank = 1, 2, 3, ...
4. Create rank-frequency data.
5. Plot:
   log(rank) vs log(frequency)

The approximate Zipf relationship is:

f(r) ∝ 1 / r^s

Equivalently:

log(f(r)) = C - s log(r)

Estimate the Zipf exponent `s`.

Report:

- Zipf exponent
- intercept
- R²
- sample size
- fitting range
- regression method

IMPORTANT:

Do not blindly fit the entire distribution and claim the entire corpus follows a perfect power law.

Real token distributions can deviate from an ideal Zipf law, especially in the head and tail.

Therefore support:

- full-range fitting
- configurable rank ranges
- tail/head analysis
- goodness-of-fit comparison

Clearly document the fitting procedure.

==================================================
7. LANGUAGE COMPARISON
==================================================

Implement the first major experiment:

English vs Hindi vs Arabic.

For each language calculate:

- total tokens
- unique tokens
- vocabulary/type count
- top-k token frequencies
- frequency distribution
- Zipf exponent
- R²
- fitting range
- token frequency statistics

Generate:

1. Individual Zipf plots.
2. Combined comparison plot.
3. Summary table.

Save results as CSV/JSON.

Do not fabricate interpretations.

Generate an automatically produced summary based only on actual results.

==================================================
8. EXISTING TOKENIZER COMPARISON
==================================================

Investigate tokenizers associated with:

- LLaMA
- Qwen
- Kimi

IMPORTANT:

Use the actual tokenizer implementations/configurations available through reliable libraries or repositories.

Do NOT assume all models use the same tokenization algorithm.

For each tokenizer determine, where available:

- tokenizer type/strategy
- vocabulary size
- special tokens
- tokenizer library
- relevant configuration
- normalization/pre-tokenization behavior

Record all of this in a machine-readable table.

Do not require the full language model weights if the tokenizer can be loaded independently.

==================================================
9. SAME TEXT, DIFFERENT TOKENIZERS
==================================================

For equivalent text in:

- English
- Hindi
- Arabic

run:

LLaMA tokenizer
Qwen tokenizer
Kimi tokenizer

Measure:

- token count
- tokens per word
- tokens per character
- unique tokens
- average token length where meaningful
- vocabulary coverage where meaningful

Produce examples showing how identical text is segmented differently.

Do not cherry-pick only favorable examples.

==================================================
10. TOKEN FREQUENCY ANALYSIS FOR EXISTING TOKENIZERS
==================================================

For every:

language × tokenizer

combination:

1. tokenize the same corpus
2. count token frequencies
3. rank tokens
4. calculate Zipf statistics
5. produce Zipf plots

This creates approximately:

English × LLaMA
English × Qwen
English × Kimi
Hindi × LLaMA
Hindi × Qwen
Hindi × Kimi
Arabic × LLaMA
Arabic × Qwen
Arabic × Kimi

Generate a summary table containing at least:

language
tokenizer
vocabulary_size
corpus_tokens
unique_tokens_used
tokens_per_word
zipf_exponent
R2

==================================================
11. CONTROLLED VOCABULARY-SIZE EXPERIMENT
==================================================

This is the MOST IMPORTANT experiment.

We need to investigate whether there is a vocabulary-size sweet spot.

Train our own tokenizer(s) while varying vocabulary size.

CRITICAL EXPERIMENTAL RULE:

When studying vocabulary size, keep everything else fixed.

For example:

Tokenizer algorithm = BPE
Corpus = fixed
Preprocessing = fixed
Training parameters = fixed
Only vocabulary size changes

Possible vocabulary sizes:

1024
2048
4096
8192
16384
32768
65536

Do NOT blindly use all values if the corpus or compute constraints make some invalid.

Automatically determine reasonable bounds.

The configuration must allow the user to change the vocabulary sizes.

==================================================
12. TOKENIZER TRAINING
==================================================

Use a well-established tokenizer-training library, preferably Hugging Face `tokenizers` if appropriate.

Choose ONE tokenizer algorithm for the controlled vocabulary-size experiment.

BPE is a reasonable default, but document the choice.

Do not compare tokenizer algorithms in the same experiment unless explicitly requested.

For every language and vocabulary size:

Train tokenizer
→ tokenize evaluation corpus
→ compute metrics
→ save results

Store trained tokenizer artifacts if reasonable.

Avoid unnecessary storage duplication.

==================================================
13. VOCABULARY-SIZE METRICS
==================================================

For every:

language × vocabulary size

calculate:

1. vocabulary size
2. actual unique tokens used
3. token count
4. tokens per word
5. tokens per character
6. average token length where meaningful
7. Zipf exponent
8. Zipf R²
9. Zipf fitting range
10. token-frequency distribution

Also calculate the incremental improvement compared with the previous vocabulary size.

For example:

Δ tokenization efficiency
Δ Zipf exponent
Δ R²

==================================================
14. DISTRIBUTION STABILITY
==================================================

This is required for the final research question.

Do not simply eyeball plots.

Implement quantitative measures of stability between successive vocabulary sizes.

Investigate appropriate metrics such as:

- Jensen-Shannon divergence
- total variation distance
- rank correlation
- change in Zipf exponent
- change in R²

Use metrics appropriate to the representation.

Document why the chosen stability metric is appropriate.

Do not claim that one metric is universally correct.

Allow the stability metric to be configured.

==================================================
15. SWEET-SPOT ANALYSIS
==================================================

Develop a systematic method to identify a candidate vocabulary-size sweet spot.

Do NOT hard-code:

"16K is the answer."

Instead, use experimental evidence.

Investigate the point where:

1. tokenization efficiency improvements become small/diminishing
2. Zipf exponent stabilizes
3. Zipf fit becomes stable
4. token-frequency distribution becomes stable

Plot these quantities against vocabulary size.

Possible concept:

Vocabulary size V is a candidate stopping point if:

relative improvement in tokenization efficiency < threshold

AND

distributional change < threshold

AND

Zipf parameter change < threshold

But DO NOT assume these thresholds are scientifically valid.

Make thresholds configurable and clearly label them as experimental criteria.

==================================================
16. ANALYTICAL CRITERION
==================================================

Implement a module that attempts to answer:

"Given a language and corpus, when should tokenizer vocabulary growth stop?"

The criterion should produce:

- candidate vocabulary size
- metrics at that size
- reason for selecting it
- confidence/strength of evidence if possible

The algorithm must be deterministic given the same data/configuration.

Example conceptual output:

Language: Hindi
Candidate vocabulary size: XXXXX

Reason:
- tokenization-efficiency gain after this point is below threshold
- Zipf exponent change is small
- distributional divergence is small
- further vocabulary growth provides diminishing returns

IMPORTANT:

This is an experimental research criterion, NOT a proven universal law.

The report must explicitly discuss limitations.

==================================================
17. CROSS-LANGUAGE SWEET SPOT
==================================================

Run the vocabulary-size experiment for:

English
Hindi
Arabic

Compare:

- candidate sweet spot
- Zipf behavior
- tokenization efficiency
- vocabulary utilization
- stability

Determine whether the candidate vocabulary size appears to depend on language.

==================================================
18. VALIDATION
==================================================

Do not evaluate the sweet-spot criterion only on the same data used to create it if avoidable.

Consider:

- training corpus
- validation corpus
- held-out corpus

If computationally feasible, train tokenizer on one portion and evaluate tokenization/statistics on another.

Explain this methodology in the report.

==================================================
19. RESULTS
==================================================

All experiments must save machine-readable results.

Use:

CSV for tabular results
JSON for metadata/configuration
PNG/SVG for plots

Every result should contain enough metadata to reproduce it.

For example:

language
corpus
tokenizer
vocabulary_size
preprocessing_version
experiment_version

==================================================
20. PLOTS
==================================================

Generate publication-quality but simple plots.

At minimum:

1. Zipf plot per language.
2. Language comparison Zipf plot.
3. Tokenizer comparison Zipf plots.
4. Tokens-per-word vs tokenizer.
5. Vocabulary size vs tokens/word.
6. Vocabulary size vs Zipf exponent.
7. Vocabulary size vs Zipf R².
8. Vocabulary size vs stability metric.
9. Vocabulary size vs incremental efficiency improvement.
10. Sweet-spot visualization.

Use log scales where appropriate.

Do not distort axes to manufacture visual conclusions.

==================================================
21. TABLES
==================================================

Generate summary tables for:

A. Language comparison

B. Existing tokenizer comparison

C. Vocabulary-size experiment

D. Sweet-spot results

E. Final recommended/observed candidate vocabulary sizes

==================================================
22. REPRODUCIBILITY
==================================================

Create:

`config.yaml`

containing:

- corpus size
- languages
- random seed
- tokenizer algorithm
- vocabulary sizes
- tokenizer models
- train/validation split
- preprocessing configuration
- Zipf fitting configuration
- stability metric
- thresholds

The entire pipeline should be runnable from documented commands.

Example:

python run_experiment.py --config config.yaml

If a better CLI structure is appropriate, use it.

==================================================
23. SMALL-SAMPLE MODE
==================================================

Because the project must be testable without huge downloads, implement:

`--mode demo`

and

`--mode full`

Demo mode should use a small corpus and reduced vocabulary sizes.

Full mode should use the configured real datasets.

The code must actually run in demo mode without requiring enormous downloads or model weights.

==================================================
24. ERROR HANDLING
==================================================

Handle:

- missing datasets
- unavailable tokenizer
- invalid vocabulary size
- insufficient corpus size
- missing dependencies
- Unicode issues
- empty corpus
- tokenizer training failure

Provide useful error messages.

Do not silently continue with invalid data.

==================================================
25. README
==================================================

Write a complete README explaining:

1. Research question
2. Experimental design
3. Installation
4. Data acquisition
5. Demo execution
6. Full execution
7. Existing tokenizer experiment
8. Vocabulary-size experiment
9. Sweet-spot criterion
10. Output directories
11. How to reproduce results
12. Limitations

==================================================
26. REPORT
==================================================

Generate:

`report/report.md`

Structure:

# Assignment 2: Zipf's Law and Tokenization

## 1. Objective

## 2. Research Questions

## 3. Dataset

## 4. Preprocessing

## 5. Zipf's Law Methodology

## 6. Language Comparison

## 7. Existing Tokenizer Comparison

## 8. Vocabulary-Size Experiment

## 9. Stability Analysis

## 10. Sweet-Spot Criterion

## 11. Results

## 12. Cross-Language Comparison

## 13. Limitations

## 14. Conclusions

## 15. Future Work

IMPORTANT:

Do not write fake numerical results.

If experiments have not been run, leave clearly marked placeholders such as:

"[RESULT GENERATED AFTER RUNNING EXPERIMENT]"

Once experiments are actually executed, populate the report automatically from the results.

==================================================
27. FINAL AUTOMATED ANALYSIS
==================================================

Create a script that reads the generated result files and automatically produces:

- summary tables
- plots
- report sections
- candidate sweet spots

The analysis must be based entirely on generated experimental data.

Do not insert manually invented values.

==================================================
28. RESEARCH INTEGRITY
==================================================

This is critical.

Do NOT:

- fabricate results
- invent tokenizer vocabulary sizes
- invent corpus statistics
- claim a sweet spot before experimentation
- claim Zipf's law is perfectly followed
- assume the same vocabulary size is optimal for all languages
- assume LLaMA/Qwen/Kimi use identical tokenization strategies
- use different preprocessing when comparing vocabulary sizes
- change multiple experimental variables simultaneously
- cherry-pick results

Every conclusion must be traceable to an actual result.

==================================================
29. COMPUTATIONAL EFFICIENCY
==================================================

Optimize the implementation.

Use:

- streaming where possible
- cached tokenized corpora
- cached frequency counts
- multiprocessing only where useful
- incremental experiments
- configurable corpus limits

Do not repeatedly tokenize the same corpus unnecessarily.

Cache expensive operations.

==================================================
30. FINAL CHECK
==================================================

After implementation:

1. Run static checks.
2. Install missing lightweight dependencies if necessary.
3. Run demo mode.
4. Verify that demo mode completes successfully.
5. Run the language-analysis pipeline.
6. Run the existing-tokenizer pipeline if model/tokenizer access is available.
7. Run a reduced vocabulary-size experiment.
8. Verify that plots are generated.
9. Verify CSV/JSON outputs.
10. Verify report generation.
11. Fix all runtime errors.
12. Update README with exact commands that worked.

At the end, give me:

A. Project structure

B. Exact commands to run the project

C. Dependencies installed

D. Which experiments successfully ran

E. Which experiments could not run and why

F. Location of every important output

G. Summary of actual findings

H. Important methodological limitations

Do not claim an experiment ran unless you actually ran it.

==================================================
31. MOST IMPORTANT DESIGN PRINCIPLE
==================================================

The project must tell a coherent scientific story:

CORPUS
  ↓
LANGUAGE
  ↓
TOKENIZATION
  ↓
TOKEN FREQUENCY
  ↓
ZIPF DISTRIBUTION
  ↓
VOCABULARY SIZE
  ↓
DISTRIBUTION STABILITY
  ↓
DIMINISHING RETURNS
  ↓
CANDIDATE SWEET SPOT
  ↓
ANALYTICAL VOCABULARY-SELECTION CRITERION

The final implementation should make it possible to investigate whether Zipf-law behavior can provide evidence for deciding when tokenizer vocabulary growth should stop.

Do not merely create plots.

Build the complete experimental framework needed to answer the research question.