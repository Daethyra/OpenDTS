# OpenDTS Project Roadmap

OpenDTS (Domestic Threat Scanner) is being developed as a real-time domestic terrorism threat detector. It aims to identify and analyze potential extremist threats by processing real-time Twitter data and other sources. The goal is to provide insights for better understanding and mitigation of these threats.

## Data Collection

- Collect real-time Twitter data.
- Download and update data links related to extremist activities.
- Ensure data is up-to-date.

## Data Preparation

- Clean and organize the collected data.
- Encode categorical data.
- Redact personally identifiable information (PII) from the tweet text using the `scrubadub` library.

## Building and Training Models

- Generate embeddings for tweet text using OpenAI's text embeddings model.
- Upsert tweet ID, vector embedding, and original text to a Pinecone index.
- Query Pinecone index for similar tweets and log potential threats based on similarity scores.

## System Development

- Develop an API endpoint for developers to access the processed data.
- Create a web application with a dynamic heatmap to visualize predicted threats.

## Testing and Deployment

- Test models for accuracy, bias, and performance.
- Conduct security audits.
- Deploy fine-tuned models to a cloud platform for real-time processing of potential threats.

## User Engagement

- Develop strategies to keep users engaged.
- Create a community for discussions.
- Provide regular updates and newsletters.

## Legal Compliance

- Comply with data privacy laws.
- Ensure the platform is legal and respectful of privacy.

## Maintenance and Future Planning

- Monitor the system regularly.
- Update models and system based on data changes and performance improvements.
- Plan for long-term sustainability.

By the end of this journey, OpenDTS will be a comprehensive platform providing valuable insights into potential extremist threats.

Also see:
- [Readme](readme.md)
- [System Breakdown](breakdown.md)
