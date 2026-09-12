# Engineering Decision Log

1. **Selected Brand: @Uber_Support over AmazonHelp**
   - *Rationale:* Amazon's operational scope (Kindle, Prime Video, AWS, Packages) is too broad for an auditable, cohesive taxonomy in a 4-day sprint. Uber presents a distinct, high-stakes operational trade-off: severe physical safety hazards vs routine fare disputes.

2. **Inner Join on `in_response_to_tweet_id` for Grounding Pairs**
   - *Rationale:* We discarded unreplied tweets because training retrieval memory on customer inquiries without verified resolutions creates dead ends for grounding.

3. **Inverted Index of Customer Queries (Not Brand Replies)**
   - *Rationale:* When a new customer writes in, we want semantic search over *what previous customers asked*, retrieving what Uber did in response. Indexing brand replies directly degrades retrieval performance because company replies are heavily templated DMs.

4. **Deterministic Policy Layer Over Single-Prompt LLM Reasoning**
   - *Rationale:* We banned prompting the LLM with "Decide whether to escalate." LLMs are overconfident and hard to audit. Instead, escalation is calculated by an explicit function with boolean gates and metric thresholds.

5. **Split Model Architectures for Drafting vs Judging**
   - *Rationale:* Claude 3 Haiku is used for drafting to model realistic production latency and token economics. Claude 3.5 Sonnet is used for the critic and judge to eliminate intra-model self-preference bias.

6. **Hard-Coded Zero-Tolerance Policy on Safety Intents**
   - *Rationale:* Even if the classifier has a 0.99 confidence score and the retrieval match is 0.95, `safety_incident` and `driver_conduct` intents are hard-coded to `ESCALATE`. Autonomous agents should not mediate assault or accident claims.

7. **Normalizing Embeddings for Inner Product FAISS Index**
   - *Rationale:* Using `normalize_embeddings=True` with `IndexFlatIP` mathematically replicates Cosine Similarity while running in single-digit milliseconds on CPU.

8. **280-Character Budget Constraint in Generation Prompt**
   - *Rationale:* Many LLM customer support demos generate lengthy 4-paragraph emails. Twitter rejects replies over 280 characters. Imposing this boundary forces the model into authentic public customer support patterns.

9. **Quadratic-Weighted Cohen's Kappa for Judge Validation**
   - *Rationale:* Standard percentage agreement treats a rating discrepancy between 4 and 5 the same as a discrepancy between 1 and 5. Quadratic weighting penalizes severe misalignments between the human labeler and the LLM judge.

10. **Rejection of Generic Vector DBs (Chroma/Pinecone) in Favor of Flat FAISS**
    - *Rationale:* Zero external infrastructure dependencies. A single binary file loads in memory in 80ms, making live interview code walkthroughs friction-free.

11. **Explicit `other_unroutable` Intent**
    - *Rationale:* Public Twitter data is saturated with random gibberish, spam, and unparseable rants. Without an explicit garbage bin class, models force random customer tweets into legitimate support categories.

12. **Conservative Asymmetric Escalation Bias**
    - *Rationale:* In customer support operations, a False Auto-Handle (sending an automated message to someone who was assaulted) is catastrophic. A False Escalation (sending a simple coupon question to a human) merely costs a few cents. Our thresholds prioritize safety over deflection rate.