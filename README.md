# OpenDTS (Domestic Threat Scanner)

OpenDTS is a comprehensive threat detection system that aims to identify and track potential threats of extremist violence against minority groups in the United States. By leveraging advanced sentiment analysis techniques, OpenDTS scans tweets and news articles for potential threats of violence and stores this information in a database for further analysis.

The primary goal of OpenDTS is to provide neighborhoods and communities with actionable insights and early warning signs of potential extremist violence, allowing them to better prepare and protect themselves from such threats.

## Features

- Sentiment analysis using OpenAI's GPT-3.5-turbo model
- Twitter and NewsAPI data collection
- NoSQL database for storing and tracking potential threats
- Message queue system for scalable data processing
- Interactive dashboard for visualizing threat data (coming soon)

## Getting Started

### Prerequisites

- Python 3.7 or higher
- An OpenAI API key
- A Twitter Developer API key
- A NewsAPI API key
- RabbitMQ server

### Installation

1. Clone the repository:
git clone https://github.com/yourusername/OpenDTS.git

2. Change to the project directory:
cd OpenDTS

3. Install the required dependencies:
pip install -r requirements.txt

4. Create a `.env` file in the project directory and add the following environment variables:
API_KEY_OPENAI=your_openai_api_key
API_KEY_TWITTER=your_twitter_api_key
API_KEY_NEWSAPI=your_newsapi_api_key
RABBITMQ_URL=your_rabbitmq_url

5. Run the main script to start the pipeline:
python main.py

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.
