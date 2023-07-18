# implicit-hate-corpus

**NEW (as of 11/06/2021):**

It is common for Twitter to ban accounts linked with offensive or hateful behavior. Unsurprisingly, more than 60% of the tweets in this dataset were no longer publicly accessible via the Twitter API as of 11/06/2021. That is why we now provide `implicit_hate_v1_stg1_posts.tsv`,  `implicit_hate_v1_stg2_posts.tsv`, and  `implicit_hate_v1_stg3_posts.tsv`. Each gives an alternative view of the data in which we provide the full text from the tweets themselves (`post`) but *not* the tweet ID according to the Twitter API guidelines.

-----------------------------

This folder also includes:

* `implicit_hate_v1_stg1.tsv`: The Stage-1 annotations (high-level; §4.2.1 in the paper) with the following columns:
  * `ID`: (str) The unique identifier for the post. 
    * If the text is from the Social Bias Inference Corpus ([Sap et al. 2020](https://homes.cs.washington.edu/~msap/social-bias-frames/)), the ID will include the prefix `SAP_`. To recover the text in this case, you can join the dataframe from `implicit_hate_v1_SAP_posts.tsv` below. 
    * Else, if the text is from a tweet, then the ID represents the unique 64-bit Tweet ID. To recover the text in this case, you should query the Twitter API. At the time of publication, the full-archive search endpoint was only available with the [Academic API](https://developer.twitter.com/content/developer-twitter/en/docs/projects/overview#product-track). Because the API is constantly evolving, we do not include any scripts for hydrating tweets, but many libraries are available (e.g. [tweepy](https://docs.tweepy.org/en/stable/))
  * `class`: (str) The high-level label in {`explicit_hate`, `implicit_hate`, `not_hate`}
* `implicit_hate_stg2.tsv`: the Stage-2 annotations (fine-grained implicit hate; §4.2.2 in the paper) with the following columns:
  * `ID`: (str) The unique identifier for the post. 
  * `class`: (str) The fine-grained implicit hate label in {`white_grievance`, `incitement`, `inferiority`, `irony`, `stereotypical`, `threatening`, `other`}
  * `extra_implicit_class`: (str) A secondary fine-grained implicit hate label in {`white_grievance`, `incitement`, `inferiority`, `irony`, `stereotypical`, `threatening`, `other`, **None**}
* `implicit_hate_v1_stg3.tsv`: the Stage-3 annotations (target and implied statement explanations; §4.2.4 in the paper) with the following columns:
  * `ID`: (str) The unique identifier for the post (with multiple annotations per post in this file). 
  * `target`: (str) Free-text annotation for the group being targeted (e.g. `Black people`, `Immigrants`, etc.)
  * `implied_statement`: (str) Free-text annotation for the implicit or hidden underlying meaning of the post made explicit (e.g. `people in minority groups are all in gangs`)
* `implicit_hate_v1_SAP_posts.tsv`:
  * `ID`: (str) The unique identifier for the text.  
  * `post`: (str) The text given in the post under evaluation.

You can read each of the files using pandas as follows, taking care to note the tab delimiter:

```python
import pandas as pd
df = pd.read_csv("stg2_file_ID.tsv", delimiter='\t')
```

For more information, please see our paper:

ElSherief, M., Ziems, C., Muchlinski, D., Anupindi, V., Seybolt, J., De Choudhury, M., & Yang, D. (2021). [Latent Hatred: A Benchmark for Understanding Implicit Hate Speech](#). In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)_.

### Note:

The link to the data access form: https://forms.gle/QxCpEbVp91Z35hWFA

The link to the Box directory is: https://gatech.box.com/s/6juaf8g5cpficlc7s6tikugyt64x404n

The password for the Box link is: `implicitH8`

