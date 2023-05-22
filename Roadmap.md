- GPT-4 "Completions" documentation: **https://platform.openai.com/docs/api-reference/chat/create**

# Choose target audience
- Wealthier people who can pay for the subscription to gain insights over their local area 
- Corporations?
- Law enforcement
- Developers who can make use of the API
    - PaidServices: 
    - API endpoint for developers to access the extremist data
    - Offer extremist data package
        - What's included?
        - Fine tuned model
          - trained on my local computer, then deployed to Azure
            - is it possible to constantly update this model during runtime?
        - Access to my embeddings index
        - Access to list of "Red Flag" (keywords/phrases(how would this be accomplished in the first place? potentially infeasible?))
===
    - Free services:
    - Templates:
        - index
          - landing page. will have a heat map(s) of geographically predicted domestic terror
          - Access to current knowledge of threats to physical safety
        - about
        - 300s,400s,500s response pages

===

- Formulate directory structure by describing what functionalities are required for the project, and then divide those into small modules of pseudocode.
- 'Cyclic Self Updating'
  - The program requires a locally fine-tuned model based on my extremist data
  - Tweepy-filtered-stream data
  - Extremist data links: 
        {
            https://github.com/CartographerLabs/Pinpoint
            https://github.com/sjtuprog/fox-news-comments
            https://github.com/FelisNigelus/WorstExtremistViolenceGlobalIncidentDatabase
        }

  - Available data:
    - Pinecone index of embeddings from real-life examples of extremist domestic terror threats, actions, and manifestos. Plus early warning signs someone may show, known as "Red Flags" 
      - Twitter is the primary source for real-time information on potential extremist threats, as increasingly, it is the young men going out and commiting mass murder #Links([**https://www.washingtonpost.com/education/interactive/school-shootings-database/**, **https://github.com/washingtonpost/data-school-shootings/blob/master/school-shootings-data.csv**])
      - These embeddings will be used to process their comparison against incoming data for potential threats of, or declarations of, domestic terrorism

===

# Preliminary actions
- Use TensorFlow to train a fine-tuned clustering machine learning model on local device
  - Train model on "shooter manifestos", bigotry Twitter accounts(Andy Ngo, ), Transcripts of Nick Fuentes' hateful videos
  - Upload fine-tuned model to Azure for processing liklihood of committing violence
  - Call model to process potential threats of violence 
    - Then, update pertinent website heatmaps
    - To address bias, we will supply jokes and sarcastic statements that actually do not show any intention of harm based on the context of the accounts' "Following" list and their top 3 replied to that would trick a model into producing biased results.

## Draft pseudocode input for GPT-4
- What functionality can be taken from online resources?
  - See: **https://docs.pinecone.io/docs/openai**
  - See: **https://platform.openai.com/docs/api-reference/chat/create**
    - What functionality goes in which modules? 
    - What requests/data will be sent/pulled to/from Pinecone?
      - *When* do we need to pull embeddings from the Pinecone index to assess potential threats of violence?
        - Is this even required? How do we compare the incoming filtered streaming data from Twitter to assess potential threats of violence?
    - How will OpenAI's API actually be useful?
      - Fine tune the model:
        - **https://platform.openai.com/docs/guides/fine-tuning**
        - During which processes related to OpenAI (Ex. Will Ada-002 do all of the 'heavy lifting', or can GPT-4 be used to improve accuracy, or perhaps 3.5-turbo to save costs?)
        - When are embeddings created from Ada-002 in the module pipeline? How many times are embeddings requested for in a single 'cycle'
        - How would the embeddings be made use of?

