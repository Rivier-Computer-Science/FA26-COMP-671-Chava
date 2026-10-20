# Held-Out Experiment Results

This results record was regenerated from the current checked-out code after local project files were deleted. It uses the UCI archive, the current fixed split (seed 42), and three training seeds for each TextCNN condition. The raw message-level error CSV and model checkpoints are local, ignored outputs.

## Main comparison

| Model | Clean PR-AUC | Clean spam recall | Clean ham FPR | Perturbed PR-AUC | Perturbed spam recall | Perturbed ham FPR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.9915 | 0.9908 | 0.0473 | 0.9895 | 0.9908 | 0.0501 |
| TextCNN, plain | 0.9705 ± 0.0057 | 0.9694 ± 0.0140 | 0.0566 ± 0.0045 | 0.9606 ± 0.0013 | 0.9694 ± 0.0140 | 0.1085 ± 0.0252 |
| TextCNN, augmented | 0.9708 ± 0.0061 | 0.9633 ± 0.0183 | 0.0515 ± 0.0037 | 0.9612 ± 0.0009 | 0.9602 ± 0.0191 | 0.0960 ± 0.0301 |

Figure 1 plots test spam recall for clean and deterministically perturbed inputs. The baseline's recall remains near 0.991 in both conditions, but its perturbed ham FPR is 0.0501, just above the 5% validation target. That target constrained validation threshold selection; it is not guaranteed on held-out or future data. TF-IDF has the highest average precision and recall in this experiment, while both CNN variants have higher perturbed false-positive rates.

The predeclared augmentation target was a five percentage-point improvement in perturbed spam recall over the plain TextCNN. It was not met: the augmented mean recall is 0.9602 versus 0.9694 for the plain CNN, a decrease of about 0.92 percentage points. Augmentation lowers the CNN mean perturbed ham FPR from 0.1085 to 0.0960 (1.25 points) but the residual false-positive rate is still nearly twice the TF-IDF rate. This is a mixed result and does not support claiming that augmentation improved overall spam detection.

## Evaluation protocol

The UCI SMS Spam Collection has 5,574 labeled messages. Duplicate normalized texts were kept within one partition. The deterministic grouped, stratified split contains 3,918 training examples (3,391 ham, 527 spam), 828 validation examples (717 ham, 111 spam), and 828 test examples (719 ham, 109 spam). Validation selects the decision threshold subject to ham FPR no greater than 5%; the test set is reserved for final metrics. The neural summaries report mean ± standard deviation over seeds 17, 42, and 73. The TF-IDF estimator is a single fit, so no seed interval is reported.

Metrics include average precision (PR-AUC), spam precision and recall, macro-F1, ham false-positive rate, and confusion counts. Spam is the positive class. The synthetic perturbation applies reproducible character edits to held-out messages. The augmentation is applied only to the augmented model's training examples.

## Failure analysis and limits

In the clean test condition, the baseline produced 34 ham false positives and one spam false negative. Under perturbation it produced 36 false positives and one false negative. The held-out ham FPR is therefore close to, and under perturbation slightly above, the target selected on validation. A reproduced inference on “Congratulations, claim your prize now” with the augmented TextCNN seed-42 checkpoint assigned spam probability 0.4763, just under its 0.4839 threshold, and predicted ham. Message-level error rows contain raw text and may contain private or identifying content; they are kept out of the tracked repository.

The experiment uses a historical English-language corpus, one random split, three neural seeds, and a limited synthetic corruption function. It does not establish current deployment performance or resilience to realistic adversarial editing. Exact duplicate grouping does not remove sender, campaign, or temporal overlap, since those fields are not available. Further evaluation should use time-separated and external data, more seeds, uncertainty intervals, and privacy-reviewed errors.

## Responsible use

False positives can hide legitimate messages. A live system would need reversible quarantine, user review, appeal or correction, and ongoing monitoring. The educational prototype should not automatically block or delete communication. Raw messages and error rows should not be published without redaction and privacy review.
