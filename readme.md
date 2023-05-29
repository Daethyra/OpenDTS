# OpenDTS: Real-Time Domestic Terrorism Threat Detector

OpenDTS (Domestic Threat Scanner) is a state-of-the-art anomaly detection system designed to identify potential domestic terror threats online in real-time. The project uses advanced techniques, such as sentiment analysis, to predict if online posts indicate an intention to commit mass harm. This project was started from scratch and is the developer's first venture into the realm of Large Language Models (LLMs) and machine learning.

## Features
- **Data Collection**: OpenDTS collects data from various sources, focusing on real-time Twitter data using Tweepy. It downloads and updates data links related to extremist activities periodically, ensuring the data is always up-to-date.
- **Data Preparation**: The system prepares the data for analysis, which involves cleaning, organizing, and encoding categorical data.
- **Anomaly Detection**: Leveraging the power of OpenAI's Ada-002 model, OpenDTS performs sentiment analysis on the collected data and compares the results with a unique vector database hosted on Pinecone.
- **Data Visualization**: The project plans to develop a dynamic heatmap web application that visualizes the predicted threats, providing users an intuitive understanding of the situation.

## Technology
While OpenDTS currently leverages the power of OpenAI's Ada-002 model for sentiment analysis and Pinecone for efficient data storage and retrieval, the project has plans to incorporate TensorFlow in the near future. TensorFlow will be used to build and train a classification model that assigns a binary value to the cosine similarity output of a given tweet's embeddings. This model will essentially categorize tweets as 'likely to commit violence' (True, or 1) or 'not indicating intention to act' (False, or 0) based on their cosine similarity to known extremist content. For this, a specific threshold for 'likelihood to commit violence' will be defined. This means that even if a tweet contains direct references to violence, it may still be classified as False, or 0, if it doesn't indicate an intention to act. The system is designed to handle and process a large amount of data efficiently. Once fine-tuned, the models will be deployed to Azure, a cloud platform, where they'll process potential threats of violence in real-time.


## Roadmap
OpenDTS is on its way to becoming a comprehensive platform that provides valuable insights into potential extremist threats. The roadmap includes data collection, preparation, model building, system development, testing, deployment, user engagement, legal compliance, maintenance, and future planning. For more details, check the project's [roadmap](roadmap.md).

## Acknowledgements
This project leverages technologies from OpenAI and Pinecone. The Ada-002 model from OpenAI is used for sentiment analysis, and the vector database from Pinecone is used for storing and accessing data efficiently.

## Developer's Note
This project reflects my commitment to independent learning and self-taught skills in artificial intelligence and machine learning. I hope it provides insights that can be used to better understand and mitigate domestic terror threats. Your feedback and contributions are always welcome.

## Disclaimer
Given the sensitive nature of the data, OpenDTS complies with all relevant data privacy laws and is respectful of everyone's privacy.

For more information on how OpenDTS works, refer to the [system breakdown](breakdown.md).


### Please note:
All [releases](https://github.com/Daethyra/OpenDTS/releases) are VERY OLD and only remain to share where the project has been, and how far it has come.
