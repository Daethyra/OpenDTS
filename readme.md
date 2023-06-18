# OpenDTS: Real-Time Domestic Terrorism Threat Detector

OpenDTS (Domestic Threat Scanner) is an advanced system designed to identify potential domestic terror threats online in real-time. It uses state-of-the-art techniques such as sentiment analysis and embeddings to predict if online posts, especially tweets, indicate an intention to commit mass harm.

## Features

- **Data Collection**: OpenDTS collects real-time Twitter data and updates data links related to extremist activities to ensure the data is always up-to-date.

- **Data Preparation**: The system prepares the data for analysis, which involves cleaning, organizing, and encoding categorical data.

- **Anomaly Detection**: OpenDTS uses OpenAI's text embeddings model to generate embeddings for tweet text. It then upserts the tweet ID, vector embedding, and original text to a Pinecone index. The system queries the Pinecone index for similar tweets and logs potential threats based on similarity scores.

- **Data Redaction**: The system redacts personally identifiable information (PII) from the tweet text using the `scrubadub` library.

- **Data Visualization**: The project plans to develop a dynamic heatmap web application that visualizes the predicted threats, providing users an intuitive understanding of the situation.

## Technology

OpenDTS leverages OpenAI for generating embeddings and Pinecone for efficient data storage and retrieval. It uses the OpenAI's text embeddings model for sentiment analysis and anomaly detection. The system is designed to handle and process a large amount of data efficiently.

## Roadmap

OpenDTS aims to be a comprehensive platform for insights into potential extremist threats. The roadmap includes data collection, preparation, model building, system development, testing, deployment, user engagement, legal compliance, maintenance, and future planning. For more details, check the project's [roadmap](roadmap.md).

## Acknowledgements

This project leverages technologies from OpenAI and Pinecone. The text embeddings model from OpenAI is used for sentiment analysis, and Pinecone is used for storing and accessing data efficiently.

## Developer's Note

This project reflects the developer's commitment to independent learning and self-taught skills in artificial intelligence and machine learning. It aims to provide insights that can be used to better understand and mitigate domestic terror threats. Feedback and contributions are welcome.

## Disclaimer

Given the sensitive nature of the data, OpenDTS complies with all relevant data privacy laws and is respectful of everyone's privacy.

For more information on how OpenDTS works, refer to the [system breakdown](breakdown.md).
