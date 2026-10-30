// Response to Reviewers for NLP-D-26-00114 as a Word document.
// Text: Prof. Elnagar's draft of 25 Oct 2026, updated on 31 Oct 2026 for the final manuscript
// (open items closed, page/line locations from revision/LETTER_LOCATIONS.md).
//   node revision/make_response_docx.js   ->  TEX_paper/revised/Response_to_Reviewers_NLP-D-26-00114.docx
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType, BorderStyle } = require("docx");

const FONT = "Calibri";
const run = (text, opts = {}) => new TextRun({ text, font: FONT, size: 22, ...opts });
const P = (children, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts, children: Array.isArray(children) ? children : [run(children)] });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 }, children: [run(t, { bold: true, size: 30 })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 }, children: [run(t, { bold: true, size: 26 })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 80 }, children: [run(t, { bold: true, size: 23 })] });
const comment = (t) => P([run("Reviewer comment. ", { bold: true }), run(t, { italics: true })]);
const response = (parts) => (Array.isArray(parts) ? parts : [parts]).map((t, i) => P(i === 0 ? [run("Response. ", { bold: true }), run(t)] : [run(t)]));
const location = (t) => P([run("Change location. ", { bold: true }), run(t)], { spacing: { after: 220, line: 276 } });

const meta = [
  ["Manuscript", "NLP-D-26-00114"],
  ["Journal", "Natural Language Processing Journal"],
  ["Title", "Arabic News Classification: Lightweight Supervised Models vs. Zero-Shot Large Language Models"],
  ["Revision deadline", "22 November 2026"],
  ["Response date", "31 October 2026"],
];
const metaTable = new Table({
  width: { size: 9000, type: WidthType.DXA }, columnWidths: [2400, 6600],
  rows: meta.map(([k, v]) => new TableRow({ children: [
    new TableCell({ width: { size: 2400, type: WidthType.DXA }, borders: noBorders(), children: [P([run(k, { bold: true })], { spacing: { after: 40 } })] }),
    new TableCell({ width: { size: 6600, type: WidthType.DXA }, borders: noBorders(), children: [P([run(v)], { spacing: { after: 40 } })] }),
  ] })),
});
function noBorders() { const n = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }; return { top: n, bottom: n, left: n, right: n }; }

const body = [
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 200 }, children: [run("Response to Reviewers", { bold: true, size: 36 })] }),
  metaTable,
  P(""),
  P("Dear Professor Xie,"),
  P("Thank you for the opportunity to revise our manuscript. We appreciate the reviewers' careful assessment. The revision distinguishes frozen CAMeLBERT feature extraction from full encoder fine-tuning, narrows the research question to the effect of in-domain supervision, and adds a QLoRA-adapted open-weight LLM, few-shot conditions, supervised learning curves, bilingual prompt-robustness tests, output-mapping audits and a near-duplicate-aware shared test split."),
  P("The re-evaluation also identified problems in the original pipeline: truncated and incorrectly mapped Claude outputs, different test samples across model evaluations, a model-identity error in one GPT-labelled run, default stochastic decoding, and a discrepancy between the printed prompt and the prompt used in code. We describe these corrections explicitly in Section 3.7 and replace the affected results. Three unavailable API models were replaced by named successors; the revised benchmark identifies those models separately. We have withdrawn the original causal explanations of Claude's performance and the unsupported claims about internal model mechanisms."),
  P("Our responses and the locations of the corresponding changes follow. Page and line numbers refer to the clean revised manuscript; a copy with the additions and changes made for the reviewers printed in blue is also provided. The revised conclusions concern this publisher, dataset and supervision setting rather than a general superiority of lightweight models over LLMs."),
  P("Sincerely,"),
  P("Omar Elgendy, Ali Bou Nassif and Ashraf Elnagar (corresponding author), University of Sharjah"),

  H1("Reviewer 1"),
  P("We thank the reviewer for recognising the practical value of the benchmark and its category-level and efficiency analyses. The comments helped us correct the methodology and strengthen the evaluation."),

  H2("Minor comment 1. Classification terminology"),
  comment("The task is single-label, seven-way multi-class classification, but the keywords and text describe it as multi-label classification. Align the terminology throughout."),
  ...response("We agree. The revised abstract, keywords and dataset description identify the task as seven-way, single-label multi-class classification. Each article has exactly one category. References to multi-label studies remain only where they describe the related literature; we do not claim to have conducted multi-label experiments. The existing title does not itself describe the task as multi-label."),
  location("Abstract and keywords, pp. 1–2 (keywords lines 14–16); Dataset and Split, pp. 6–7, lines 181–196. Related-work references to multi-label research are contextual rather than descriptions of our task."),

  H2("Major comment 1. Frozen encoder and fine-tuning"),
  comment("State exactly what is trained for each supervised system, correct the discussion, and report any full fine-tuning protocol, trainable layers and parameter counts, including how the SVM embeddings are produced."),
  ...response([
    "The original lightweight systems use frozen CAMeLBERT-MSA representations. Only the dense head is updated by gradient descent; the RBF-SVM is fitted to precomputed embeddings and never backpropagates into CAMeLBERT. Table 2 now separates these systems from the newly added fully fine-tuned CAMeLBERT baseline and the QLoRA-adapted Qwen model. For both frozen systems the representation is the 768-dimensional final-layer [CLS] vector; SVM embeddings are precomputed in batches of 32. The dense head has 297,991 trainable parameters. The full fine-tuning baseline updates all encoder layers and its classification layer, approximately 109.1 million parameters.",
    "The revision reports AdamW, learning rates, schedules, batch sizes, input lengths, regularisation, maximum epochs and early stopping. Full fine-tuning uses learning rate 2 × 10⁻⁵, a linear schedule with 6 % warm-up, batch size 32, up to three epochs and validation-loss stopping. The discussion now attributes the frozen systems' performance to a pretrained Arabic representation combined with supervised decision-boundary learning, rather than to task-induced changes to the encoder.",
  ]),
  location("Systems, pp. 8–9, lines 231–249, and Table 2, p. 10; Training Configuration, p. 13, lines 313–335, and Table 4, p. 13; Discussion, pp. 25–27, lines 564–616. The original frozen-versus-fine-tuned interpretation is removed (visible in the tracked-changes copy)."),

  H2("Major comment 2. Supervision and LLM evaluation controls"),
  comment("The comparison confounds architecture with supervision. Add an adapted LLM, deterministic decoding, invalid-output rates, a precise label mapper and a robustness check."),
  H3("Task-adapted open-weight LLM"),
  ...response([
    "We added Qwen2.5-7B-Instruct both without task adaptation and after QLoRA on the same canonical training partition used by the supervised systems. QLoRA uses a 4-bit NF4 base with double quantisation and rank-16, alpha-32 adapters on the attention and MLP projections, with 40.4 million updated parameters (0.53 % of the 7.62 billion base parameters). We report adaptation at 100 articles per class, 1,000 per class and the full training split. The full-data configuration reaches 98.5 % accuracy, compared with 85.2 % for its zero-shot primary-prompt condition on the full test split. This result supports the importance of supervision and prevents a blanket claim that lightweight architectures outperform task-adapted LLMs.",
    "Table 11 reports 1.3, 8.9 and 20.2 million training tokens and 0.29, 1.48 and 3.27 GPU-hours respectively on one A100 40 GB GPU; peak reported VRAM is 9.3 GB. The full-data comparison uses the same 31,833 training articles and 6,822 test articles, while the systems keep their separately stated input-length limits.",
  ]),
  location("Introduction, pp. 2–4, lines 17–94; Tables 2 and 4, pp. 10 and 13; Main Results, pp. 15–16, and Table 5, p. 16; Training and Inference Cost, p. 22, lines 518–532, and Table 11, p. 23."),
  H3("Deterministic decoding"),
  ...response("Table 3 reports temperature 0, top-p 1, seed 42 where supported, a 60-second timeout and the retry handling. The output allowance is 16 tokens for the label-generation conditions and 256 for Gemini 3.8 Flash, whose route does not allow reasoning to be switched off; it was requested at the lowest reasoning level the route accepts and spent 0–246 thinking tokens per request (mean 45), which are logged. We report this Gemini exception explicitly. API evaluations are dated 16 October 2026, and the served identifiers and response metadata are logged for every request. Repeat-pass results are reported rather than assuming that temperature 0 guarantees identical predictions."),
  location("LLM Evaluation Protocol, pp. 9–13, lines 250–312; Table 3, p. 11; Table 8, p. 20; Infrastructure and Reproducibility, pp. 14–15, lines 364–375; Limitations, pp. 27–28, lines 617–638."),
  H3("Non-compliant and unmappable outputs"),
  ...response([
    "The revision counts outputs requiring normalisation separately from outputs that cannot be assigned one label. Empty, ambiguous and unmapped responses, including unresolved API errors, count as classification errors and remain in the evaluation denominator. Table 9 audits all recorded runs per model, including exact-label, resolved and invalid shares and worst-run rates. The accompanying text gives the primary full-test invalid rates: 0.00 % for GPT-4.1, 0.04 % for Claude, 0.06 % for Gemini and 0.01 % for DeepSeek.",
    "Appendix B, Table B.2, breaks the invalid outputs down by cause over all 132,430 outputs: 259 contained no recognised label, 9 were empty responses after an API error, none mentioned two labels, and 398 were cut off by the output budget, of which 390 still contained a mappable label.",
  ]),
  location("Label mapping, p. 12, lines 279–291; output audit, pp. 18–22 (Section 4.4, lines 464–502), and Table 9, p. 21; Appendix B, p. 30, and Table B.2, p. 32."),
  H3("Label mapping"),
  ...response("We replaced the previous partial-string and random-fallback behaviour with a fixed ordered mapper. It accepts a canonical label, a label after formatting normalisation, an accepted Arabic label form, or an output containing exactly one recognised label. For example, “sports.” maps to Sports and “The class is Finance” maps to Finance. Multiple recognised labels are ambiguous; no recognised label is unmapped. A secondary synonym-tolerant analysis, including Economy to Finance and Health to Medical, changes accuracy by no more than 0.2 percentage points. The complete accepted vocabulary (the seven labels, their Arabic forms, the normalisation rules and the lenient-tier synonyms) is given in Appendix B, Table B.1. The mapping implementation and the raw outputs are released."),
  location("Label mapping, p. 12, lines 279–291; Table 9, p. 21; Appendix B, pp. 30–31, and Table B.1, p. 31; Data and code availability, p. 29, lines 666–672."),
  H3("Prompt robustness and variability"),
  ...response("We added five prompt formulations, three English and two Arabic, with their full text in Appendix A. Table 8 reports mean accuracy, standard deviation and range across prompts on a fixed 700-article subset. Prompt standard deviations are 0.5, 0.7, 0.3, 1.5 and 2.5 percentage points for GPT-4.1, Claude, Gemini, DeepSeek and Qwen respectively. The table also reports a second primary-prompt pass. The primary benchmark uses the full 6,822-article test split, while prompt sweeps and few-shot comparisons use the shared subset. We keep comparisons within the same evaluation set and state that the subset is easier for the LLMs by 1.4–2.6 points."),
  location("Prompts and Evaluation sets, pp. 9–13, lines 262–312; Evaluation and Statistical Analysis, pp. 13–14, lines 336–363; Prompt Robustness, pp. 18–22, lines 464–502, Table 8, p. 20, and Figure 4, p. 21; Appendix A, pp. 29–30, lines 673–701; Limitations, pp. 27–28, lines 617–638."),

  H1("Reviewer 2"),
  P("We thank the reviewer for recognising the relevance of the question, the substantial test set, the class-level analysis and the examples showing errors in both directions. We have retained these elements while rebuilding the comparison around a common split and explicit supervision conditions."),

  H2("Major concern 1. Interpretation of frozen CAMeLBERT"),
  comment("The frozen encoder is incorrectly described as fine-tuned. The results support frozen pretrained representations with supervised decision-boundary learning, not Arabic-news-specific encoder adaptation."),
  ...response("We agree with this distinction. The two original lightweight classifiers are now explicitly described as frozen CAMeLBERT plus a trained dense head or a fitted RBF-SVM. The representation is not updated on SANAD in either system. Full encoder fine-tuning is a separate new baseline. We removed the statements attributing the original heads' gains to end-to-end fine-tuning and revised the scientific interpretation to supervised learning over fixed Arabic representations. Tables 2 and 4 identify what is trained for each system."),
  location("Systems, pp. 8–9, lines 231–249, and Table 2, p. 10; Training Configuration, p. 13, and Table 4; Discussion, pp. 25–27, lines 564–616; Conclusion, pp. 28–29, lines 639–665. See also our response to Reviewer 1, major comment 1."),

  H2("Major concern 2. Architecture and supervision confound"),
  comment("Narrow the claim, include few-shot and preferably adapted LLMs, and add learning curves at 10, 50, 100, 500, 1,000 and full training examples per class."),
  ...response([
    "The revised question is how much in-domain supervision, and which form of adaptation, is needed to exceed zero-shot or few-shot inference on this dataset. We added one and two in-context examples per class, QLoRA adaptation of Qwen2.5-7B-Instruct, full CAMeLBERT fine-tuning, and the requested supervised training sizes. Table 7 reports nested training subsets and variability over three seeds. For the 150-wordpiece dense head, mean full-test accuracy rises from 82.5 % at 10 training articles per class to 92.9 % at 100 and 96.4 % with the full split. The 512-wordpiece head reaches 95.1 % at 100 per class.",
    "Few-shot API conditions use the 700-article subset, so the revision compares them with supervised predictions restricted to those same articles. The 512-wordpiece head trained on 500 articles per class reaches 96.7 % on that subset, compared with 95.3 % for the best measured few-shot condition. These are empirical crossover observations under the reported protocol.",
    "The learning curve varies the training labels only. The dense head and full fine-tuning select their stopping epoch on the full validation partition at every training size, the SVM uses no validation data, and QLoRA trains for a fixed number of epochs and uses a validation sample only for monitoring. Sections 3.5 and 4.3 and the Limitations state this explicitly, so the per-class counts are to be read as training labels rather than as an end-to-end annotation budget.",
  ]),
  location("Introduction, pp. 2–4, lines 17–94; Main Results, pp. 15–16, and Table 5, p. 16; learning curve, pp. 17–18, lines 444–463, with Figure 3 and Table 7, p. 19; Evaluation and Statistical Analysis, pp. 13–14, lines 336–363; Table 8, p. 20; Discussion and Limitations, pp. 25–28, lines 564–638."),

  H2("Major concern 3. Prompt fragility and instruction language"),
  comment("Evaluate several equivalent prompts, including Arabic and English instructions, and report mean performance and variation so that capability, prompt compatibility and parsing are not conflated."),
  ...response([
    "We added three English and two Arabic prompt formulations, publish their full wording, and run the same variants through the same deterministic evaluation and mapping pipeline. Table 8 and Figure 4 report the mean, standard deviation and range on the same stratified subset. The study now measures prompt dependence rather than inferring it from differences between models. The primary prompt is evaluated on the full test set; subset results are presented as within-subset comparisons. The large original Claude deficit is withdrawn after the pipeline audit rather than interpreted as evidence of limited Arabic understanding.",
    "Section 4.4 reports that only GPT-4.1 is the most accurate model under every prompt; the order of Claude Haiku 4.5, Gemini 3.8 Flash and DeepSeek V4 Flash changes with the prompt, and the Discussion no longer claims a stable ranking.",
  ]),
  location("Prompts, pp. 9–12, lines 262–278; Evaluation sets, pp. 12–13, lines 292–312; Prompt Robustness, pp. 18–22, lines 464–502, Table 8, p. 20, and Figure 4, p. 21; Appendix A, pp. 29–30."),

  H2("Major concern 4. Reproducibility of LLM inference"),
  comment("Report exact snapshots, evaluation dates, temperature, top-p, output budgets, deterministic settings, retry policy and invalid-output rates; model family names alone are insufficient."),
  ...response("Table 3 lists the requested and served identifiers, provider routes, output allowances, evaluation date, decoding settings, timeout and retries. The revised models are GPT-4.1, Claude Haiku 4.5, Gemini 3.8 Flash, DeepSeek V4 Flash and local Qwen2.5-7B-Instruct. The manuscript explains the replacement of the unavailable original API models and logs the served identifiers, raw outputs, finish reasons, token counts and latency for every request. Table 9 supplies invalid-output rates. The manuscript distinguishes dated snapshots (claude-haiku-4-5-20251001, DeepSeek-V4-Flash-0731) from provider aliases without a dated version (gpt-4.1, gemini-3.8-flash); for the latter the logged identifier and the evaluation date establish what was evaluated but not an immutable model version, and the Limitations acknowledge that hosted aliases can change."),
  location("Models and snapshots, p. 9, lines 251–261; Table 3, p. 11; Table 9, p. 21; Infrastructure and Reproducibility, pp. 14–15, lines 364–375; Limitations, pp. 27–28, lines 617–638."),

  H2("Major concern 5. Post-processing as part of the classifier"),
  comment("Report non-conforming responses, ambiguous labels, additional explanations and unmappable outputs, including scoring and the effect on the poor-performing model."),
  ...response("We agree that the mapper is part of the evaluated classifier. The revision specifies deterministic mapping, retains invalid responses as errors, and audits exact versus resolved versus invalid outputs over all runs. It explains that some Arabic or English prompt variants generate formatted labels or explanations that the fixed mapper can resolve. A lenient-mapping sensitivity analysis changes accuracy by at most 0.2 points. The original Claude anomaly is explicitly attributed to output truncation and an incomplete mapper rather than to model capability. Appendix B, Table B.2, provides the finer split of invalid outputs into no label, two labels, empty responses and outputs cut by the output budget, per model."),
  location("Label mapping, p. 12, lines 279–291; Corrections to the Originally Reported Experiments, p. 15, lines 376–399; output audit, pp. 18–22, and Table 9, p. 21; Appendix B, p. 30, and Table B.2, p. 32."),

  H2("Major concern 6. Claude Religion result and causal interpretation"),
  comment("Verify the near-zero Religion recall before attributing it to culture, alignment, safety filters or training data. Remove causal speculation without targeted evidence."),
  ...response([
    "We agree and withdraw the original causal interpretation. The pipeline audit identified a one-token output allowance, Arabic outputs truncated to uninformative prefixes, and a mapper without a Religion branch that randomly assigned another label to unrecognised responses. These defects invalidate the original 75 % accuracy and zero Religion recall as evidence about Claude's linguistic or cultural ability. Section 3.7 reports the historical check separately: a re-run of the same model (claude-3-5-haiku-20241022) in December 2025, before its retirement, with a 10-token output budget on two 1,000-article samples gave 92.5 % and 93.1 % accuracy with Religion recall 0.76 and 0.77. The present benchmark uses the successor model Claude Haiku 4.5, which reaches 92.0 % accuracy with Religion recall 0.741 on the full test set; the two results are kept distinct.",
    "The discussion removes the safety-filter and alignment explanations. It describes the observed confusion with Culture and the publisher's category conventions, and notes that the weakness on Religion is shared by all the LLMs tested, while avoiding a claim that the experiments establish a specific internal cause.",
  ]),
  location("Corrections to the Originally Reported Experiments, p. 15, lines 376–399; Table 5, p. 16; Class-Level Analysis, pp. 16–17, lines 424–443, and Table 6, p. 18; Discussion, pp. 25–27, lines 564–616."),

  H2("Major concern 7. Unsupported mechanisms"),
  comment("Remove unsupported claims about surface heuristics, deeper semantics and a domain-specific knowledge graph, unless backed by probing, attribution or controlled interventions."),
  ...response("The original claims that training the frozen heads adapts the encoder or constructs a knowledge graph have been deleted. We added input-length and dateline-removal interventions and replaced the mechanistic narrative with measured differences between supervision regimes. The input-length comparison shows a benefit from longer article context (1.1 points for the frozen encoder; the LLMs lose 0.6–1.0 points when restricted to the first 100 words); dateline removal changes any system's accuracy by at most 0.4 points. The manuscript now states these as measured effects of the two cues examined rather than as proof that no model exploits lexical shortcuts, and the absolute formulations in the abstract, Sections 4.2 and 4.5, the Discussion and the Conclusion were replaced accordingly. The qualitative examples remain illustrations of possible error patterns, not tests of an internal reasoning mechanism."),
  location("Evaluation and Statistical Analysis, pp. 13–14, lines 336–363; Controlled Analyses, p. 22, lines 503–517; Qualitative Error Analysis, pp. 22–25, lines 533–563, and Tables 12–15, pp. 23–26; Discussion, pp. 25–27, lines 564–616."),

  H2("Major concern 8. Near duplicates and event leakage"),
  comment("Investigate duplicate stories and related-event leakage under random splitting; use deduplication or a temporal/event-based split to assess generalisation more credibly."),
  ...response([
    "The revision removes 23 exact duplicates and groups near-duplicates using TF-IDF cosine similarity over word unigrams and bigrams at threshold 0.8, with connected components formed by union-find. It reports 183 multi-article clusters containing 1,403 articles. Each cluster is assigned wholly to one partition. The resulting shared split contains 31,833 training, 6,822 validation and 6,822 test articles. No test article has a cross-partition neighbour at or above the threshold. All systems are re-evaluated on this canonical split.",
    "The analysis also reports residual overlaps at lower thresholds and scores the 6,305 test articles remaining after excluding the 517 articles with a training or validation neighbour at cosine 0.5 or above. Accuracy decreases only modestly for every system, including zero-shot GPT-4.1, which has seen no training article, and the supervised advantage remains. The split file, checksums and per-article similarity information are released.",
    "We have implemented the deduplication alternative proposed by the reviewer; we have not conducted a temporal or event-disjoint split, because the corpus carries no publication dates. Section 3.1 and the Limitations now state that lexically dissimilar reports about the same event can remain across partitions, so the procedure supports a near-duplicate-aware in-publisher comparison, not a general claim of event-free generalisation.",
  ]),
  location("Dataset and Split, pp. 6–8, lines 181–230, and Table 1, p. 9; Limitations, pp. 27–28, lines 617–638; Data and code availability, p. 29, lines 666–672."),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: body }],
});
const out = path.join(__dirname, "..", "TEX_paper", "revised", "Response_to_Reviewers_NLP-D-26-00114.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
