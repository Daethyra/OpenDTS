# CyberSentinel

### A package of modules for automating the LLM training process for the /project/data folder

---

#### - Overview -

- Intention:
  - Identify the intention to commit violence based on hatred via binary classification
  - Fine tune OpenDTS' threat-intelligence decision making via similarity queries
- Module status:
  - ~`preprocessor.py` is in an unfinished state~
- Functionality:
  1) Configure API keys
  2) Load, clean and split the data
  3) Tokenize
  4) Send data to OpenAI for embedding
  5) Store results in Pinecone index
  6) Perform similarity queries

* Origin:
  * This was the original set of modules planned for OpenDTS
  * @Daethyra (ME!) decided augmenting AutoGPT makes the most sense (duh)
  * Serve as a notification robot to alert at-risk minority demographics
