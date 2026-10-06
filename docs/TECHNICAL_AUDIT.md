# Mutable technical audit

- Date: 2026-10-05
- Audited Mutable revision: `c80014c`
- OPLM reference checkout: `b45c7b5fc6324755a99539d051095e028392fc6c`

## Assessment

**Mutable has working model components, but it is not ready for a trustworthy
end-to-end training run.** The largest problems are mismatches between training
and generation, broken checkpoint handoffs, and silent data corruption. These
should take priority over architectural changes or scaling.

The review covered model and flow mathematics, tokenization, masking, datasets,
evaluation, configuration, CLI entrypoints, training, persistence, and cleanup.
OPLM was used as a reference for configuration and operational conventions.
Findings describe the audited revision; this document does not implement fixes.

The complete test suite passed: **377 passed, one skipped, one warning**.
Additional behavioral probes exposed failures that those tests miss. Testing
used CPU with PyTorch `2.14.0+rocm7.2`, Transformers `5.16.1`, Datasets `5.0.1`,
Accelerate `1.14.0`, Hydra `1.3.2`, and OmegaConf `2.3.0`. The optional adaptive
ODE test was skipped because `torchdiffeq` was unavailable. Missing Hydra and
OmegaConf packages in the original shell environment were installed into a
temporary location; both are already declared project dependencies.

GPU execution, distributed training, throughput, and real training datasets were
not validated. Runtime observations, static integration gaps, and research
hypotheses are distinguished below. Neither repository's implementation was
changed during the audit.

## 1. Correctness: issues to fix before substantive training

### C1. Flow training and generation use different starting distributions

Training interpolates from encoded germline latents to encoded mutated latents.
Generation instead starts from independent Gaussian noise. The learned velocity
field is applied to a different problem at inference.

References: [training source](../src/mutable/models/flow_matching.py#L199) and
[generation source](../src/mutable/models/flow_matching.py#L281).

**Runtime evidence:** with dropout disabled and `sigma_min=0`, replacing the
velocity network with an oracle returning the exact target-minus-source latent
velocity produces zero training loss. One-step Euler generation still reaches
the wrong latent endpoint, with RMSE **1.3153**.

This requires an explicit modeling decision:

- Germline-to-mutated transport: use germline latents consistently, accepting
  deterministic latent generation for fixed inputs.
- Stochastic conditional generation: train and sample with the same stochastic
  source, conditioning on germline throughout.

Adding noise only during inference cannot provide correctly learned diversity.

### C2. Generation is not autoregressive

[`generate()`](../src/mutable/models/flow_matching.py#L305) supplies
`[BOS, PAD, PAD, ...]` to the decoder once and takes an argmax at every position.
Later positions never receive earlier predictions. This differs fundamentally
from the teacher-forced decoding used during training.

**Runtime evidence:** a decoder hook recorded exactly one call, with input
`[[0, 1, 1, 1, 1, 1]]`. Generation also inherits the input batch's padded width
and never stops at EOS.

The minimum correction is a prefix-growing decoding loop with EOS stopping and
an explicit length limit. Output handling also needs attention: the tokenizer
inserts spaces between amino acids, and the
[CLI preserves those internal spaces](../src/mutable/cli.py#L299). Missing chain
separators currently become empty light-chain outputs. Generated chain pairs
need validation before export.

### C3. The default Phase 1 checkpoint cannot be consumed by Phase 2

Phase 1 saves `model.safetensors`, but
[Phase 2 loading](../src/mutable/train.py#L519) hardcodes `pytorch_model.bin`.

**Runtime evidence:** an actual CLI handoff failed with `FileNotFoundError` after
a tiny Phase 1 run saved successfully.

Use the existing `MutableForDenoising.from_pretrained()` interface and take the
backbone configuration from that checkpoint. Requiring users to reconstruct the
pretrained architecture through CLI overrides is unnecessary and error-prone.
Check missing backbone/head weights explicitly instead of relying on permissive
`strict=False` loading.

### C4. Flow exports overwrite the configuration needed to reload the model

[Final saving](../src/mutable/train.py#L612) first saves the model's backbone
configuration, then saves `FlowMatchingConfig` into the same directory. Both
write `config.json`. The same collision occurs in the output root during setup.

The backbone's encoder depth, decoder depth, and latent count disappear.

**Runtime evidence:** a tiny backbone configured with encoder/decoder/latent
counts `1/1/2` reloads its configuration as `6/6/32` after the flow configuration
is saved. A final flow export failed to load because of weight-shape mismatches.

Periodic checkpoints have the complementary problem: ordinary model saving
preserves the backbone config but omits `self.flow_config`.

Both configurations must round-trip through the model's save/load implementation.
Use either one complete configuration with a nested flow section or a separately
named flow configuration file with consistent save/load handling. Test periodic
and final checkpoints using a deliberately nondefault architecture.

### C5. Frozen Phase 1 representations remain stochastic during flow training

[Freezing](../src/mutable/models/base.py#L60) disables gradients but leaves
dropout active when Trainer calls `model.train()`. The flow forward pass uses
`torch.no_grad()`, which does not disable dropout.

**Runtime evidence:** identical inputs produced different latents in a frozen
backbone, with maximum absolute difference **0.008104** in a tiny model.
Encoding identical germline and target sequences separately can therefore
create artificial mutation velocities. Inference uses different, deterministic
representations.

Keep the frozen backbone in evaluation mode while training the flow network.
Also freeze the LM head, which is unused by the flow training objective.

Separately, [Phase 2 accepts no pretrained checkpoint](../src/mutable/train.py#L510)
and freezes a randomly initialized backbone by default, despite CLI help saying
a checkpoint is required. This path was observed to report a normal training
loss. Missing or incomplete Phase 1 weights should be a startup error.

### C6. Passing a decoder attention mask disables causality

[Self-attention](../src/mutable/modules/attention.py#L77) disables SDPA's causal
flag whenever an explicit mask exists, without combining that mask with a
causal triangle.

**Runtime evidence:** changing only future tokens changed the first-token output
by **8.4165** with an all-ones attention mask; without the mask, the difference
was zero. The separately reconstructed attention weights still showed zero
attention to future positions, concealing the defect.

Combine padding and causal constraints in the shared attention implementation.
Test output invariance to future-token changes with an explicit mask, not just
the separately returned attention weights.

This is an exposed model API bug. The current default training collator
generally omits `decoder_attention_mask`, so it does **not** establish that every
default training run leaks targets.

### C7. CSV ingestion silently corrupts positional annotations

CSV loading infers numeric types for digit-only annotation columns. For example:

```text
Original:       00000101
CSV value:      101
Rebuilt mask:   10100000
```

[Mask construction](../src/mutable/datasets/denoising_dataset.py#L160) converts
the inferred number back to a string, then pads on the right. This was reproduced
through the actual CSV loader and relocates annotated residues, corrupting
weighted masking and region evaluation. The evaluation dataset duplicates the
same parsing behavior.

Preserve annotation columns as strings or typed arrays at ingestion. Reject
length mismatches instead of silently repairing them.

### C8. Truncation and mutation intensity disagree with the model inputs

[Annotation construction](../src/mutable/datasets/denoising_dataset.py#L214)
uses original sequence lengths after tokenization has truncated the sequence.
A reproduced example had 12 tokens and 19 annotations, causing the collator to
fail. Evaluation duplicates this defect.

[Flow intensity calculation](../src/mutable/datasets/flow_dataset.py#L129) has
two further problems:

- It concatenates heavy and light chains before positional comparison.
  `ACDE|FGHI -> ACD|FGHI` produces `mu = 0.625`, although there is only one deletion
  among eight original residues. The unchanged light chain is incorrectly
  counted as changed. Internal indels similarly shift subsequent positions.
- It computes intensity without accounting for truncation. A mutation entirely
  outside the retained tokens produced identical encoded endpoints with
  nonzero `mu = 0.125`.

For an initial training pipeline, rejecting unsupported lengths and indels is
simpler and safer than silently approximating their annotations. Support for
indels should use validated per-chain alignments and an explicit intensity
definition.

## 2. Training and recovery integration

These failures occur in orchestration around otherwise functioning components.

| Area | Finding | Minimum correction |
| --- | --- | --- |
| Logging | `--no-wandb` assigns `report_to=["none"]` after argument normalization, causing an unsupported-integration error in the tested environment. W&B is enabled by default but is not a declared dependency. | Set reporting during argument construction; use an empty list to disable it. Declare/document optional logging dependencies. |
| Default validation | Training presets enable evaluation and best-checkpoint loading, while data presets provide no validation dataset. | Validate this combination before constructing the model, or deliberately disable evaluation when no validation data are supplied. |
| Named validation | Custom loaders are built, but Trainer receives `eval_dataset=None` and rejects the enabled evaluation strategy. | Establish one HF-compatible evaluation contract. |
| Best-model selection | Named evaluation emits `eval/<dataset>/<metric>`, while the recipe expects `eval_loss`. | Explicitly select and consistently emit the monitored metric. |
| Flow evaluation | `FlowMatchingTrainer.evaluate()` returns timing metrics but no `eval_loss`; checkpoint selection raises `KeyError`. | Implement loss-returning evaluation through Trainer's prediction contract. |
| Evaluation callbacks | The custom denoising evaluation override skips HF's `on_evaluate` callback. | Preserve the normal callback lifecycle when overriding evaluation. |
| Config replay | CLI-injected Hydra group overrides turn `data` and `train` into strings when replaying a saved standalone YAML. | Separate preset selection from YAML overlays. |
| Seeds | CLI defaults overwrite YAML/dotlist seeds, and the resolved seed is not passed into `TrainingArguments`, which retains 42. | Resolve and propagate one seed. |
| Resume | Neither training entrypoint passes `resume_from_checkpoint`; there is no CLI recovery path. | Expose HF Trainer's existing resume support. |
| Unknown options | Introspection-based filtering silently discards unsupported keys, concealing attempted options or misspellings. | Explicitly handle application-only fields and reject unexpected keys. |

References: [training orchestration](../src/mutable/train.py#L363),
[evaluation override](../src/mutable/trainer/denoising_trainer.py#L118),
[flow trainer](../src/mutable/trainer/flow_trainer.py#L22),
[config loading](../src/mutable/train.py#L90), and
[argument filtering](../src/mutable/config/from_hydra.py#L131).

### Runtime boundary checks

To inspect downstream failures, reporting callbacks were bypassed **only in the
diagnostic harness**. Production code was not patched. With that bypass:

| CLI exercise | Observed result |
| --- | --- |
| Tiny Phase 1 run with a single validation CSV | Trained, evaluated, and saved. |
| Phase 1 without validation | Failed because evaluation was enabled without an eval dataset. |
| Phase 1 with named validation | Failed because Trainer still received no eval dataset. |
| Phase 1 export passed into Phase 2 | Failed on the hardcoded checkpoint filename. |
| Flow training with validation | Failed on missing `eval_loss` during best-model selection. |
| Flow training without pretrained weights, with evaluation disabled | Completed despite freezing a random backbone. |
| Final flow export passed into generation | Failed because the backbone configuration had been overwritten. |

Configuration probes also confirmed that a requested global seed of 71 left
`TrainingArguments.seed` at 42, and replaying saved YAML through CLI defaults
changed `cfg.data` and `cfg.train` into the scalar string `"denoising"`.

### Distributed readiness

Auxiliary configuration, tokenizer, and flow-configuration writes are not
rank-gated; only Trainer-managed saves receive Trainer's coordination. Custom
evaluation builds ordinary loaders and bypasses normal Accelerator preparation.
Region accumulators are not reduced across ranks when used with sharded loaders.

These are static integration findings, not demonstrated multi-GPU failures.
Perform a two-rank train/eval/save/reload test before claiming distributed
readiness. Preserve HF Trainer's existing distributed behavior wherever possible.

## 3. Evaluation correctness

The evaluation harness has useful structure, but some reported numbers cannot
yet be trusted.

| Finding | Evidence and impact | Correction |
| --- | --- | --- |
| Contact predictions depend on batch padding. | For the same sequence alone versus alongside a longer sequence, real-residue attention differed by only `7.45e-9`, but APC scores differed by `0.01236` and only 15 of the top 20 contacts agreed. | Exclude padded/special query and key positions before symmetrization and APC. |
| CDR-contact precision can actually be overall precision. | Missing CDR annotations cause an unrestricted fallback. The structure dataset supplies no CDR mask, yet the default configuration requests this metric. | Omit or reject metrics whose required annotations are unavailable. |
| Missing coordinates count as negative contacts. | Candidate selection does not exclude residues lacking observed coordinates. A sample with entirely missing coordinates still contributed five negative predictions. | Exclude unobserved residues from candidate pairs and evaluated length. |
| Structure parsing can silently select the wrong chains. | If either requested chain is absent, both selections are replaced by the first two protein chains, potentially substituting an unrelated chain. | Require explicit chain assignments or fail clearly. |
| Region annotations depend on training masking mode. | Annotation-column configuration reaches evaluation only when weighted training masking is enabled. | Load evaluation annotations independently of training corruption policy. |
| Some evaluation controls are ineffective. | Sample-limit lookup uses metric display names instead of configuration names; region `mode` and `seed` are unused. | Implement the advertised behavior or remove the options. |

References: [contact extraction](../src/mutable/eval/metrics/contact.py#L106),
[CDR fallback](../src/mutable/eval/metrics/contact.py#L206),
[contact candidate selection](../src/mutable/eval/metrics/contact.py#L295),
[chain fallback](../src/mutable/eval/structure_parser.py#L190),
[annotation configuration](../src/mutable/train.py#L329), and
[sample-limit lookup](../src/mutable/eval/evaluator.py#L146).

Sequence-distance exclusions also apply across heavy/light chains, where
distance in the concatenated token sequence has no corresponding within-chain
meaning. This should be separated from within-chain contact filtering.

Silently emitting a differently defined metric is worse than omitting it.
Distinguish what the reconstruction metrics measure: masked reconstruction uses
a teacher-forced decoder with a clean preceding prefix. It does not establish
successful free-running generation or useful mutation control.

The evaluator leaves the model in evaluation mode, but ordinary HF
`Trainer.training_step` restores training mode. The audit did not establish a
persistent training-mode corruption bug from that behavior.

## 4. Architecture: retain the core, establish the missing evidence

The encoder -> fixed latent bottleneck -> decoder design is a reasonable
research architecture. Pre-norm transformers, RoPE, SwiGLU, cross-attention, and
conditional flow blocks are individually defensible choices.

The Phase 1 objective is full-sequence teacher-forced reconstruction, consistent
with the general [BART approach](https://aclanthology.org/2020.acl-main.703/).
**Information weighting changes which encoder positions are masked; it does not
directly weight reconstruction loss.** Weighted masking is disabled in the
shipped default preset and requires correctly configured annotations to obtain
the intended weighting.

### Highest-value architectural experiments

1. **Define the conditional generative problem precisely.** Decide whether the
   flow should generate multiple latent outcomes for a fixed germline and
   intensity. A deterministic ODE from one fixed starting point produces one
   latent outcome. Conditional noise-to-data flow is a standard alternative,
   but its source distribution must match between training and sampling.
   [Flow Matching paper](https://arxiv.org/abs/2210.02747)
2. **Prove the bottleneck carries sequence-specific information before freezing
   it.** Compare reconstruction using correct, shuffled, and zeroed latents.
   Measure free-running reconstruction, separator/EOS validity, chain lengths,
   and mutation-site fidelity. Low teacher-forced loss alone can conceal an
   overly capable decoder that underuses its latents.
3. **Benchmark a simpler conditional encoder-decoder.** Reuse the existing
   components to establish whether the latent-flow stage improves diversity,
   conditioning, or held-out quality. This is more informative than immediately
   adding architectural complexity.
4. **Make mutation intensity an explicit data contract.** The implementation
   currently uses amino-acid difference fraction. That is not a nucleotide SHM
   rate, and the ODE timestep is not an evolutionary clock. Junction differences,
   indels, and uncertain germline assignments require deliberate treatment.
5. **Treat optimal transport accurately.** `optimal_transport_plan()` is never
   called by training. The implementation uses a straight interpolation formula,
   but does not perform minibatch OT coupling. Arbitrarily re-pairing genuine
   germline-mutated examples would not be a sound correction.
   [Conditional flow matching and minibatch OT](https://arxiv.org/abs/2302.00482)

These are research questions, not claims that a different architecture has
already been shown to outperform Mutable.

### What already works

- Encoder padding masks reach both encoder attention and the bottleneck.
- Causal decoding without an explicit decoder mask passed the future-token
  perturbation check.
- A tiny Phase 1 model produced nonzero encoder and bottleneck gradients; a
  one-example optimization probe reduced cross-entropy from approximately
  `3.5079` to `0.1559` in 60 updates. This establishes a functioning optimization
  path, not generalization or latent utility.
- The implemented interpolation and target velocity are algebraically
  consistent. Euler and RK4 use conventional update formulas. The principal flow
  defect is connecting these components to the wrong inference source.

The current presets are sufficient for the proposed experiments:

| Preset | Phase 1 parameters | Flow-network parameters |
| --- | ---: | ---: |
| Small | 5,954,496 | 6,355,904 |
| Base | 23,881,280 | 15,483,456 |
| Large | 117,778,944 | 36,609,024 |

Flow-network counts exclude the frozen backbone and decoder head.

### Numerical and implementation contracts

- **Explicit BF16 conversion:** `.bfloat16().generate()` fails because sinusoidal
  features remain FP32 while projection weights become BF16. BF16 rotary
  position calculation also makes positions 256 and 257 indistinguishable.
  Preserve FP32 positional calculations and cast features appropriately at the
  projection boundary. These were reproduced with explicit model conversion,
  not established as failures of ordinary Trainer AMP training.
  [Embedding implementation](../src/mutable/modules/embedding.py#L55)
- **AdaLN initialization:** generic model initialization overwrites the intended
  zero initialization of adaptive normalization projections. All 512 weights of
  a tiny model's examined projection were nonzero. Preserve specialized
  initialization after generic initialization. This is a contract defect, not
  proof that optimization cannot work.
  [AdaLN initialization](../src/mutable/modules/layers.py#L264)
- **Gradient checkpointing:** support is advertised, but enabling it raises an
  incompatibility error. Implement block checkpointing or remove the claim
  before recommending this option for larger runs.
  [Base model](../src/mutable/models/base.py#L19)

## 5. Data preparation requirements

The repository consumes prepared CSVs. It does not provide a complete,
documented path from repertoire data to validated training pairs.

Before a meaningful pilot, require:

- A typed paired-data schema with stable sample identifiers and provenance.
- Validation of chain presence, sequence alphabet, annotation lengths/values,
  finite intensity, and supported lengths. Default-mode tokenization currently
  permits a null chain to become the literal string `"None"`, producing residue
  `N` and an unknown token instead of rejecting the row.
- Explicit germline-reference and mutation-intensity definitions, including
  junctions, indels, and uncertain assignments.
- Deduplication and split manifests that account for clone/donor relationships.
- Accepted/rejected row counts and basic dataset summaries.
- Small real-data fixtures exercising CSV ingestion through model input
  construction, including leading-zero annotations.
- Held-out generation evaluation covering reconstruction, validity,
  conditioning, diversity, and memorization.

This is missing protection against leakage, not evidence that existing external
datasets already leak. Their preparation state was not available for the audit.
Keep a simple CSV/HF Datasets path for the first validated dataset; use typed
Parquet or additional preprocessing/caching where data integrity or measured
throughput justifies it.

## 6. Design consistency with OPLM

OPLM's strongest contribution is its operational contracts. The following links
are pinned to the reference checkout's revision.

| OPLM pattern | Recommendation for Mutable |
| --- | --- |
| [Defaults -> preset -> YAML -> dotlist loading](https://github.com/briney/oplm/blob/b45c7b5fc6324755a99539d051095e028392fc6c/src/oplm/config.py#L585), with reloadable serialization | Adopt the same OmegaConf precedence and replay behavior. |
| [Unknown model-key rejection](https://github.com/briney/oplm/blob/b45c7b5fc6324755a99539d051095e028392fc6c/src/oplm/config.py#L549) | Reject misspelled or unsupported configuration instead of silently filtering it. |
| [Thin CLI adapter](https://github.com/briney/oplm/blob/b45c7b5fc6324755a99539d051095e028392fc6c/src/oplm/cli.py#L62) | Keep command handling thin and share one loader. Click itself is adequate. |
| [Documented data contracts](https://github.com/briney/oplm/blob/b45c7b5fc6324755a99539d051095e028392fc6c/docs/DATA_TOOLING.md#L12) and [deterministic evaluation](https://github.com/briney/oplm/blob/b45c7b5fc6324755a99539d051095e028392fc6c/src/oplm/data/sequence/loaders.py#L143) | Share train/eval sequence preparation and make validation masking reproducible. |
| [FP32 rotary calculations](https://github.com/briney/oplm/blob/b45c7b5fc6324755a99539d051095e028392fc6c/src/oplm/model/rope.py#L94) | Reuse the numerical policy. |
| Tested dependencies and CI | Add actual CLI lifecycle tests and a supported dependency range. |

OPLM uses raw-sequence Parquet shards and collator tokenization. It does not
supply the antibody-specific preparation pipeline Mutable needs.

Retain HF Trainer initially and wire its checkpoint, resume, and distributed
contracts correctly. Copying OPLM's custom trainer, distributed checkpoint
machinery, sweeps, and Slurm support would expand the work substantially without
resolving Mutable's immediate correctness defects.

## 7. Cleanup and streamlining

The repository's main problem is incomplete integration rather than pervasive
overengineering. These cuts have clear justification:

1. **Reuse:** consolidate duplicated train/eval annotation construction; both
   copies already contain the same defects.
   [Evaluation dataset](../src/mutable/eval/datasets/eval_dataset.py#L89)
2. **Reuse:** remove duplicated config discovery from `model-size`; share the
   training loader. [CLI](../src/mutable/cli.py#L350)
3. **Reuse:** use the existing chain-ID helper and shared loss accumulation
   instead of duplicate implementations.
   [Evaluator](../src/mutable/eval/evaluator.py#L44),
   [classification metrics](../src/mutable/eval/metrics/classification.py#L150)
4. **Delete:** remove unused `_MASK_KEYS`, `flow_config_path`, and the uncalled
   mutation-position utility. Repository-wide searches found no callers of the
   utility; account for its public export when removing it.
5. **Remove unsupported flexibility:** remove or implement ineffective options
   such as region `per_position`, region seed, and decoder `use_cache`.
6. **Dependencies:** remove unused direct requirements `pandas`, `evaluate`, and
   `tqdm`. This removes direct declarations, not necessarily transitive installs.
   SciPy becomes removable if the OT helper is retired; that helper currently
   has tests and public exports but no production callers.
7. **Native feature:** count parameters on the meta device, as OPLM does, instead
   of allocating two complete models.

Estimated reduction: **200-350 lines and three direct dependencies**, with
another dependency conditional on retiring the OT API. This is a cleanup
estimate, not a measured patch.

Documentation also needs repair: the README is effectively empty; AGENTS/CLAUDE
describe denoising columns that differ from shipped defaults (`sequence_heavy`
and `sequence_light` versus `heavy` and `light`); and the only CI workflow
publishes packages without running tests. Keep AGENTS.md and CLAUDE.md identical
when correcting their shared instructions.

## 8. Training-readiness assessment and acceptance criteria

| Milestone | Assessment |
| --- | --- |
| Tiny tensor-level training | Already works; smoke tests and optimization probes establish this. |
| Functional Phase 1 -> Phase 2 -> generation | Blocked by correctness, evaluation, and persistence defects. |
| Scientifically credible pilot | Also requires validated data, bottleneck evidence, and generation-focused evaluation. |
| OPLM-like operational readiness | Requires demonstrated resume, mixed precision, two-rank execution, and measured data throughput. |

Assuming validated input pairs already exist, several focused engineering days
is a reasonable estimate for repairing the functional pipeline and its
regression tests. Budget roughly **one to three weeks for a defensible pilot**,
including data validation and architecture checks, excluding substantial dataset
construction and training compute. This is an engineering estimate, not a
measured schedule; unknown external data preparation may dominate the work.

### First acceptance test

Exercise one real CLI lifecycle using nondefault tiny model configurations:

```text
validated data -> Phase 1 train/eval -> save/reload
-> Phase 2 train/eval -> resume -> save/reload -> valid generated pairs
```

The lifecycle should establish that:

- The intended annotations survive on-disk ingestion and tokenization.
- Decoder outputs remain causal with explicit padding masks.
- Frozen backbone representations remain deterministic during flow training.
- Flow training and generation use the same source distribution.
- Every checkpoint contains enough configuration to reconstruct both networks.
- Evaluation emits the metric used for checkpoint selection.
- Resume restores optimizer/scheduler state and global step.
- Generation uses previous predictions, stops correctly, and exports valid,
  whitespace-free heavy/light sequences.

Only after this passes should GPU mixed precision and a two-rank lifecycle be
validated. Benchmark data loading before adding a new streaming or caching
framework.

### Why the existing green suite is insufficient

The tests bypass much of the lifecycle: generation tests primarily check token
shapes, checkpoint round-trip tests cover Phase 1, and trainer smoke tests
manually supply objects that the CLI fails to construct correctly. Causality
tests do not cover the explicit-mask combination. Dataset tests predominantly
use already typed dictionaries rather than actual CSV ingestion.

The baseline audit command, after satisfying project dependencies, was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest tests/ -q --disable-warnings
```

Additional probes used tiny CPU models, temporary CSV files, decoder hooks, an
oracle velocity network, padding-invariance comparisons, configuration replay,
and Click's CLI runner. They were temporary audit diagnostics, not committed
regression tests. Their observed results are recorded above so this report does
not depend on machine-local temporary files.
