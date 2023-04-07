# OpenDTS
## Open Domestic Threat Scanner - Real-time sentiment analyzer and neighborhood alert system.

OpenDTS is a project that aims to provide a systematic and efficient way of identifying potential threats of extremist violence against minority groups in the United States. The project utilizes sentiment analysis to scan news outlets and tweets to identify potential threats of violence. Headlines and tweets are saved to a database, then piped into OpenAI's moderation endpoint. 

The intention behind the creation of OpenDTS is to help neighborhoods across America prepare for real threats to their safety by providing actionable insight and  warning signs of extremist violence.

===


*End goals:*
- Scan mainstream News outlets, Twitter, and Scholarly for data, based on a keyword list created by the user. 
- Keyword list receives *real-time* updates when necessary to stay up to date with "dog whistles".
-- NewsAPI: Scrape all categories and log headlines related to queer & BIPOC issues. User may choose to manually review the headlines, or have the machine perform sentiment analysis.
-- Twitter: Scrape users' profiles if a tweet matches the keyword list, and the tweet passes the user-set threshold for an intention of violence. Scan for repeated negative sentiment. Log each tweet that passes the threshold. Store number of offenses. 
- Perform predictive sentiment analysis for the intention of violence against queer and BIPOC folk based on temperature averages over time using the Chat3.5-turbo natural language processing model.
