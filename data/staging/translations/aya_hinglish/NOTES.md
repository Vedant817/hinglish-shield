# Aya HI -> Hinglish sample

- Rows: 50
- Method: `llm_authored_by_agent`
- Detail: Hinglish produced by Data Engineer A agent using Aya Hindi prompt plus Aya literal_translation English. Not a deterministic IAST/rule-based transliterator. Not native-speaker verified. Pending Vedant review.
- Replaced source_ids: 258, 334, 356, 399, 439, 450, 484, 502
- Reason: Replaced rows that were operational violence how-to, riot-incitement how-to, population-reduction, sexual-embarrassment how-to, or detailed physical-harm how-to. Also excluded Non-consensual sexual / Self-Harm / Graphic material categories and weapons/CSAM keyword hits upstream.
- Fallback hinglish ids (lighter polish): 3, 6, 11, 13, 15, 18, 19, 21
- Split path: `data/staging/translations/aya_hinglish/`
- No em dashes.
