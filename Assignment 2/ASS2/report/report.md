# Assignment 2: Zipf's Law and Tokenization

## 1. Objective
Measure rank-frequency behavior across English, Hindi, and Arabic and test whether controlled BPE vocabulary growth has diminishing returns.

## 2. Research Questions
Language, tokenizer, and vocabulary size are varied in separate experiments. The stopping rule is an experimental heuristic, not a universal law.

## 3. Dataset
Mode: `demo`. Text source used: deterministic built-in demo corpus. Corpus metadata:

| language | source | documents | characters | words | raw_size_bytes | processed_size_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| english | built-in deterministic corpus | 1 | 12000 | 1698 | 12000 | 12000 |
| hindi | built-in deterministic corpus | 1 | 12000 | 4433 | 31092 | 31092 |
| arabic | built-in deterministic corpus | 1 | 12000 | 1944 | 21849 | 21849 |

## 4. Preprocessing
NFC Unicode normalization, URL replacement, whitespace normalization, and Unicode-aware word extraction. Casing and punctuation are not blindly removed.

## 5. Zipf's Law Methodology
Token frequencies are sorted by descending count. Ordinary least squares fits `log(frequency)` against `log(rank)` from rank 2 through the configured range; head and tail deviations remain possible.

## 6. Language Comparison
| language | source | documents | characters | words | raw_size_bytes | processed_size_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| english | built-in deterministic corpus | 1 | 12000 | 1698 | 12000 | 12000 |
| hindi | built-in deterministic corpus | 1 | 12000 | 4433 | 31092 | 31092 |
| arabic | built-in deterministic corpus | 1 | 12000 | 1944 | 21849 | 21849 |

## 7. Existing Tokenizer Comparison
No rows generated.

## 8. Vocabulary-Size Experiment
Fixed corpus, preprocessing, and character BPE algorithm; only requested vocabulary size changes.

| language | tokenizer | requested_vocab_size | actual_vocab_size | corpus_tokens | unique_tokens_used | tokens_per_word | tokens_per_character | average_token_length | zipf_exponent | r2 | fit_min_rank | fit_max_rank | js_divergence | relative_efficiency_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| english | controlled_char_bpe | 32 | 29 | 10116 | 29 | 5.957597173144876 | 0.843 | 0.9813167259786477 | 0.9537875726647579 | 0.7783364141383455 | 2 | None | None | None |
| english | controlled_char_bpe | 64 | 45 | 8100 | 40 | 4.770318021201414 | 0.675 | 1.2255555555555555 | 0.7674157499982114 | 0.9022191270836247 | 2 | None | 0.1284462926952966 | 0.19928825622775795 |
| english | controlled_char_bpe | 128 | 77 | 5898 | 45 | 3.4734982332155475 | 0.4915 | 1.6831129196337742 | 0.7594649781837568 | 0.9221714283865479 | 2 | None | 0.14825191562768775 | 0.2718518518518519 |
| hindi | controlled_char_bpe | 32 | 31 | 9139 | 30 | 2.06158357771261 | 0.7615833333333333 | 0.6118831381989277 | 0.8662525843265644 | 0.8682712419456579 | 2 | None | None | None |
| hindi | controlled_char_bpe | 64 | 47 | 5999 | 35 | 1.3532596435822242 | 0.4999166666666667 | 0.9321553592265378 | 0.8551539296692157 | 0.9147902191628361 | 2 | None | 0.4766696991586272 | 0.34358244884560674 |
| hindi | controlled_char_bpe | 128 | 70 | 4433 | 37 | 1.0 | 0.36941666666666667 | 1.2614482291901647 | 0.6581610085036637 | 0.8899179942284203 | 2 | None | 0.3369377022105289 | 0.2610435072512085 |
| arabic | controlled_char_bpe | 32 | 31 | 10339 | 30 | 5.318415637860082 | 0.8615833333333334 | 0.95260663507109 | 1.1133814343842847 | 0.8973451212545506 | 2 | None | None | None |
| arabic | controlled_char_bpe | 64 | 47 | 7767 | 38 | 3.9953703703703702 | 0.64725 | 1.268057164928544 | 0.8069891665535985 | 0.9083730673279131 | 2 | None | 0.17647632731860785 | 0.24876680530031917 |
| arabic | controlled_char_bpe | 128 | 79 | 5326 | 44 | 2.7397119341563787 | 0.44383333333333336 | 1.849230191513331 | 0.8044215762349497 | 0.4879679416000382 | 2 | None | 0.1965363345060794 | 0.3142783571520535 |

## 9. Stability Analysis
Jensen-Shannon divergence compares normalized token-frequency distributions between successive vocabulary sizes. It is symmetric and bounded, but no single stability metric is universally correct.

## 10. Sweet-Spot Criterion
A candidate is the first size after which relative tokens-per-word improvement, JS divergence, Zipf-exponent change, and R² change are all below configurable thresholds. The result is evidence from this corpus, not a proven optimum.

## 11. Results
| language | candidate_vocab_size | evidence | confidence |
| --- | --- | --- | --- |
| english | None | no size passed all checks | experimental |
| hindi | None | no size passed all checks | experimental |
| arabic | None | no size passed all checks | experimental |

## 12. Cross-Language Comparison
Compare candidates only after rerunning full mode on balanced held-out corpora; demo values are smoke-test outputs.

## 13. Limitations
The demo corpus is small and repetitive. Optional LLaMA/Qwen/Kimi models may be unavailable offline. Pretrained tokenizer rows are reported only when actually loaded.

## 14. Conclusions
No conclusion is asserted beyond the measured rows above. Reproducibility metadata and raw tables are stored under `results/`.

## 15. Future Work
Use balanced Wikipedia extracts, multiple held-out samples, bootstrap uncertainty intervals, and accessible exact Kimi tokenizer configuration.

## Execution Notes
- llama: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
- qwen: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
- kimi: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
