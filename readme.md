# OpenDTS: Real-Time Domestic Terrorism Threat Detector

#### TLDR: It uses a multi-layered approach with techniques to predict if online posts, especially tweets, indicate an intention to commit mass harm.

###### OpenDTS (Domestic Threat Scanner) is a project intended to designed to provide analytical insight into hostile sentiment towards user-defined minority demographics.

---

## Developer's Thoughts

The application will do its best to accurately discern true intent to harm versus sarcasm, frustration without the intent to harm, and will make key decisions on multiple different layers comprised of entire models. Like the Ada-002 from OpenAI, for example. 

Ideally, the first actual deployment will be a heatmap serving real-time processing data for whatever I choose. 

- OpenDTS is the manifestation of my attempt to help others protect themselves
- I want it to be a comprehensive open-source cyberspace threat-intelligence platform


*All information below is outdated as of 7/17/23*
---

## Features

- **Data Collection**: OpenDTS collects real-time Twitter data and updates data links related to extremist activities to ensure the data is always up-to-date.

- **Data Preparation**: The system prepares the data for analysis, which involves cleaning, organizing, and encoding categorical data.

- **Anomaly Detection**: OpenDTS uses OpenAI's text embeddings model to generate embeddings for tweet text. It then upserts the tweet ID, vector embedding, and original text to a Pinecone index. The system queries the Pinecone index for similar tweets and logs potential threats based on similarity scores.

- **Data Redaction**: The system redacts personally identifiable information (PII) from the tweet text using the `scrubadub` library.

- **Data Visualization**: The project plans to develop a dynamic heatmap web application that visualizes the predicted threats, providing users an intuitive understanding of the situation.

--

## Roadmap

The usage of OpenAI is intended to serve a comprehensive platform for cyberspace threat-intel. 

The roadmap includes 
1. data collection(✔),
2. data sanitization,
3. a TensorFlow module set for model building(✔),
4. environment development(✔),
5. testing,
6. deployment,
7. and finally to provide online services via web/mobile app and an API service for developers. 

For more details, check the project's [roadmap](roadmap.md).

## Acknowledgements

This project leverages technologies from OpenAI and Pinecone. The text embeddings model from OpenAI is used for sentiment analysis, and Pinecone is used for storing and accessing data efficiently. This project uses 'AI', or machine learning.

### Disclaimer

Given the sensitive nature of the data, OpenDTS does its best to remove all PII during preprocessing.

For more information on how OpenDTS works, refer to the [system breakdown](breakdown.md).

---
