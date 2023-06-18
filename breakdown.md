# OpenDTS System Breakdown

OpenDTS (Domestic Threat Scanner) is designed to identify and analyze potential domestic terrorism threats in real-time. Here's how the system works:

### Collecting Information:

- Gathers information about potential threats from sources like Twitter.
- Automatically redacts all personally identifiable information (PII) using the `scrubadub` library before further processing.

### Analyzing the Information:

- Uses OpenAI's text embeddings model to generate embeddings for tweet text.
- Analyzes the sentiment of these texts to identify possible threats of violence.
- Passes information through OpenAI and Pinecone for analysis.

### Storing the Information:

- Stores tweet ID, vector embedding, and original text in a Pinecone index.
- The database is designed to handle a large amount of data and quickly access it for similarity queries.

### Processing the Information:

- Efficiently handles and processes incoming data.
- Respects rate limit policies and privacy policies on all platforms interacted with.
- Queries Pinecone index for similar tweets and logs potential threats based on similarity scores.

### Visualizing the Threats:

- Plans to develop a dashboard that visually represents the data, making it easier to understand the potential threats.

The main goal of OpenDTS is to provide communities with early warnings about potential violence, helping them prepare and protect themselves.

Also see:
- [Readme](readme.md)
- [Roadmap](roadmap.md)
