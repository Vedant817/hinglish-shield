# Compute options: Tiny Aya Global baseline, Locate, Repair

- Written: 2026-11-06 IST by ML Research Lead. Prices fetched 2026-11-06 IST from the URLs below.
- Nothing has been signed up for or spent. New accounts and any paid spend need Vedant's OK via Lexsi.
- All cost and time figures below are **estimates**, not measurements.

## Workload estimate (assumptions)

- Model: CohereLabs/tiny-aya-global, about 3.35B params, bf16 weights about 7 GB; with KV cache for batch 16 x 512 tokens, fits a 16 GB card, comfortable on 24 GB.
- Prompts: 3 languages x (harmful + benign) x about 300 = about 1,800 generations per condition, max_new_tokens 256.
- Conditions: base on all splits (1,800); refuse-everything control (no GPU); steering grid on dev only, about 3 layers x 4 alphas + 2 controls = 14 conditions x about 540 dev prompts = about 7,500; final frozen arms on test, about 3 x 900 = 2,700. Total about 12,000 generations, about 3M output tokens upper bound.
- Activation caching for direction extraction: calibration prompts only, residual stream at all layers, a few hundred prompts; minutes of GPU, a few GB of disk.
- Judging: WildGuard (7B) over about 12,000 responses; fits on 24 GB in bf16, or run after unloading Tiny Aya.
- Throughput assumption: batched HF generate on L4/A10-class gives roughly 300 to 600 output tok/s for a 3B model; T4 roughly 2x to 3x slower and no native bf16 (fp16 needed, logged as deviation, overflow risk to check).
- Resulting GPU time estimate: **about 4 to 10 GPU-hours on a 24 GB card**, about 10 to 25 hours on T4. Add 30% for setup, reruns and cold starts.

## Options

| Option | GPU (VRAM) | Price | New account? | Est. cost for workload | Source |
| --- | --- | --- | --- | --- | --- |
| Kaggle Notebooks | T4 x2 (2 x 16 GB, separate cards) | Free, about 30 GPU-h/week quota, 12 h sessions | Yes unless Vedant already has one (phone verification) | $0; fits in about 1 week of quota | kaggle.com/docs/efficient-gpu-usage (search snippet; direct fetch timed out). P100 retired 2026-09-15 per third-party guides only, unverified |
| Google Colab free | T4 typically (not guaranteed) | Free, unpublished dynamic limits, max 12 h | Google account likely exists | $0, but unreliable for multi-hour batch | research.google.com/colaboratory/faq.html |
| HF ZeroGPU (Spaces) | Half RTX Pro 6000 (48 GB) | Free 5 min/day; PRO 40 min/day; then $1 per 10 GPU-min | No (Vedant817 exists). Hosting needs account older than 30 days; Gradio only | Not suitable for batch: 4 to 10 h would be about $24 to $60 in overage credits | huggingface.co/docs/hub/en/spaces-zerogpu |
| HF Spaces GPU | L4 24 GB $0.80/h; A10G small 24 GB $1.00/h; T4 small $0.40/h; L40S 48 GB $1.80/h | per hour | No (existing HF account); can use HF $20 hackathon credits (ADR-1) | L4: about $4 to $10 | huggingface.co/pricing |
| HF Inference Endpoints | T4 $0.50/h, L4 $0.70 to $0.80/h, A10G $1.00/h | per minute | No account, but needs card on file | Serving only; activation caching and steering hooks not practical. Not recommended | huggingface.co/docs/inference-endpoints/en/pricing |
| Modal | L4 $0.000222/s (about $0.80/h); A10 about $1.10/h; T4 about $0.59/h; L40S about $1.95/h | per second; Starter includes $30/month free compute | Yes | L4: about $4 to $10, likely fully inside $30 free credit | modal.com/pricing |
| RunPod Pods | L4 24 GB $0.59/h secure ($0.44 community); A40 48 GB $0.59/h ($0.35); RTX A5000 24 GB $0.27/h ($0.16) | per second | Yes, prepaid credit | A5000 or L4: about $2 to $7 | runpod.io/pricing (page dated 2026-11-04) |
| Lambda | A10 24 GB $1.29/h; A6000 48 GB $1.09/h; A100 40 GB $1.99/h | per hour; listed prices take effect 2026-11-16 UTC, current ones need login | Yes | A6000: about $5 to $15 | lambda.ai/pricing |
| Hackathon credits | HF $20 credits (ADR-1); $200 only if shortlisted | credit | No | Covers HF Spaces L4 plan with margin | Notion HQ / ADR-1 (not re-verified today) |

Modal hourly figures are per-second list prices x 3,600. Region selection or non-preemptible runs cost 1.15x to 3x more on Modal.

## Recommendation

**Default: HF Spaces GPU (1x L4, $0.80/h) or HF Jobs on Vedant's existing HF account, paid from the $20 hackathon credits.** No new account, uses credits we already have, 24 GB is enough for Tiny Aya plus WildGuard sequentially, and the estimate (about $4 to $10) leaves headroom. Confirm the credits are actually on the account before starting.

- Free parallel path: Kaggle T4 x2, if Vedant already has a phone-verified Kaggle account. Good for smoke and dev sweeps; fp16 deviation must be logged.
- If HF credits are not available: Modal Starter (about $30 free compute/month) or RunPod A5000/L4 (cheapest paid, about $2 to $7). Both need a new account and Vedant's OK.
- Not recommended: ZeroGPU (quota model fits demos, not batch), Inference Endpoints (no hook access), Lambda (pricier, login needed for current prices).

## Unverified

- Kaggle quota and P100 retirement (direct page fetch timed out; third-party sources only).
- Colab free GPU type and limits (Google does not publish them).
- HF $20 credit availability on Vedant's account and whether it applies to Spaces hardware or Jobs.
- Throughput numbers are assumptions until the first GPU smoke run measures tok/s.
