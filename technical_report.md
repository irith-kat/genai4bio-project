# scAlphaGenome — Technical Report

This technical report documents the scAlphaGenome decoder model, training setup, evaluation, and comparison to the AGCell baseline. For each architectural and experimental decision I provide an explicit rationale and practical trade-offs.

## Summary

- Goal: predict cell-type-specific gene expression (raw log-CPM) from AlphaGenome (AG) sequence-level features combined with scVI cell-type embeddings.
- Main model: scAlphaGenomeDecoder — uses an AG tokenizer to embed sequence tracks, a hypernetwork that converts scVI embeddings to rich queries and FiLM parameters, then a stack of cross-attention layers that attend queries to AG keys/values, producing per-cell-type predictions.
- Baseline: AGCellBaseline — an MLP applied to AlphaGenome cell-type-conditioned AG tracks (one track per supported cell type).
- Training targets: per-gene z-scored expression across cell types; model outputs are denormalized back to raw log-CPM for evaluation.
- Loss: mean squared error (MSE) on z-scored targets. Primary evaluation metric: Pearson correlation (r) per cell type across held-out genes.

## Architecture — high level

scAlphaGenomeDecoder components and responsibilities:

- AG tokenizer (AGTokenizer): converts an AG track (8192 bins) into a sequence of d_model tokens. This lets the model use attention over local sequence bins instead of a single pooled representation. Rationale: sequence information (local peaks, motifs) is spatially distributed and attention over tokens allows downstream queries to select relevant bins per cell type.

- Hypernetwork (SCVIHyperNet): maps scVI embeddings (per-cell-type latent means) into:
  1. Dense query vectors (one per cell type), in the same d_model space used by the cross-attention layers.
  2. Per-layer FiLM (feature-wise linear modulation) parameters used to condition the cross-attention layers.

  Rationale: a hypernetwork enables parameter-efficient conditioning. Rather than concatenating scVI to every layer or learning a monolithic projection, the hypernetwork produces specialized layer-wise modulation that tailors attention computations to each cell type. This is especially useful when the number of cell types varies across organs or when we want the same AG encoder to generalize across many CTs.

- Cross-attention stack: queries (per-CT) attend to AG tokens (keys/values). Each layer is optionally modulated by FiLM params produced by the hypernetwork.

  Rationale: cross-attention lets a learned query select relevant sequence regions (bins) for each cell type. This directly implements the biological intuition: different cell types respond to different regulatory elements in the same locus. Using layered cross-attention increases the model's capacity to perform iterative retrieval and integration of sequence signals.

- Organ embedding: a small learned embedding per organ that is added to queries. Rationale: organs can systematically shift which regulatory features are informative; a per-organ bias helps the model adapt to organ-specific contexts without retraining separate decoders.

- Output head: LayerNorm → MLP → scalar prediction (z-scored). Denormalized at evaluation.

## Why these design choices? — explicit defense

- Why attention (cross-attention on queries → AG tokens)?
  - Biological motivation: regulatory information is spatial and combinatorial. Cross-attention lets a per-CT query dynamically read out relevant bins rather than relying on fixed pooling or global summaries.
  - Modularity: the same AG tokenizer + KV memory can be reused for many CT queries cheaply.
  - Interpretability: attention weights can be inspected to locate which bins influenced a CT's prediction.

- Why not just an MLP or CNN on pooled AG features?
  - MLP/CNNs with global pooling tend to collapse spatial detail. While CNNs capture local motifs, cross-attention gives flexible, content-based routing where different queries can read different sets of bins.

- Why hypernetwork + FiLM conditioning from scVI?
  - Efficient conditioning: the hypernetwork compresses per-CT latent info into lightweight modulation parameters rather than increasing encoder size linearly with CT types.
  - Expressive contextualization: FiLM-style modulations let scVI embeddings influence layer-normalized activations in a multiplicative/additive manner — proven useful in conditional generation and perception tasks.
  - Generalization: by separating sequence memory (AG tokenizer) from learned conditioning (hypernetwork), the model can generalize to unseen CTs (different scVI embeddings) without retraining the AG encoder.

- Why use scVI latent embeddings as conditioning signals?
  - scVI compressed embeddings summarize the transcriptional identity of each cell type. They capture both discrete identity and graded similarity between CTs; this is exactly the information needed to query sequence-level regulatory potentials.
  - Practical: scVI embeddings are low-dimensional and robust; they reduce noise versus using raw single-cell profiles.

- Why z-score per-gene targets (training on labels_z) and MSE on z-scores?
  - Z-scoring isolates the cross-CT variation (relative expression pattern) from per-gene baseline mean and variance. The model's purpose is to predict which CTs express a gene relative to others; removing per-gene mean simplifies optimization and focuses capacity on cell-type ranking.
  - MSE on z-scores is equivalent to a Gaussian log-likelihood on standardized targets. It's stable, differentiable, and aligns with our denormalization scheme where outputs are rescaled back to raw log-CPM for interpretability.
  - Alternative losses (rank-based, contrastive) were considered but MSE is simple and directly optimizes per-gene profile reconstruction, which maps cleanly to downstream Pearson and R² metrics.

- Why Pearson r as a primary metric?
  - Pearson r (across val genes per CT, or per gene across CTs) measures linear association between predicted and true expression profiles, ignoring scale and mean shifts. This matches our modeling decision to standardize targets and to evaluate how well the model ranks cell types per gene (or genes per CT) rather than matching absolute calibration only.
  - It is widely used in gene-expression prediction literature, easy to interpret, and complementary to MSE/RMSE and R².

- Why include baseline AGCellBaseline (MLP on agcell tracks)?
  - Apples-to-apples: AGCellBaseline uses the same AlphaGenome-derived inputs but with AG's own cell-type conditioned tracks. If our decoder cannot outperform a model trained on AG's CT-conditioned outputs, it suggests our conditioning or architecture adds no value.
  - Simplicity: MLP baseline provides a strong, easy-to-train comparator and helps quantify the quality advantage (not only coverage).

## Training & evaluation choices — defended

- Gene-split (train/val on genes): training and evaluation are done on disjoint gene sets. This evaluates generalization to new genes (loci) rather than memorization.
- Per-organ models vs multi-organ: we add an organ embedding so a single decoder can adapt across organs; this is a trade-off between sharing statistical strength and organ-specific specialization.
- Clip gradients and use Adam: standard practice for stability on relatively small batch sizes and deep modules.

## Metrics reported and how to interpret

- Pearson r per CT (mean and distribution): primary quality metric. High mean r indicates the model recovers which CTs are high/low for a gene.
- R² relative to null (per-gene mean predictor): quantifies the fraction of cross-CT variance explained.
- RMSE / MAE (raw log-CPM): captures absolute calibration and typical prediction errors in log-CPM units.
- Concordance correlation coefficient (CCC): complements Pearson by penalizing mean and variance mismatches.

Why this combination? Pearson + R² captures ranking and variance explained; RMSE/MAE check that the model's denormalized values are in a biologically meaningful range; CCC ensures agreement.

## Baseline comparison and coverage advantage

- Baseline is limited to AG-supported CTs (agcell). Our model's conditioning via scVI embeddings allows predictions for any CT in scVI space (coverage advantage). When restricted to overlapping CTs we compare with the same metrics to show quality advantage.

- Interpreting head-to-head tables: scAlphaGenome outperforming AGCellBaseline on mean Pearson r for overlapping CTs indicates that cross-attention + hypernetwork conditioning provides a stronger mapping from sequences to CT profiles than training an MLP directly on AG's CT-conditioned tracks.

## Limitations and practical notes

- Predictions are only as good as the input AG tracks and scVI embeddings. If scVI embeddings fail to separate CT identities or if AG features miss relevant regulatory signals near a gene, performance will be limited.
- Model complexity vs interpretability: attention offers interpretability at the bin level but hypernetwork parameters are opaque. We recommend attention-based inspections and perturbation analyses for mechanistic claims.
- Metric caveats: Pearson r is robust to scale but can be inflated by outliers. Complementary metrics (RMSE/CCC) are reported for more complete assessment.