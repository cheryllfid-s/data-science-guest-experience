# Improving Theme Park Guest Experience with Data Analytics

## Project Overview
This project aims to develop a data-driven system that maps and analyzes guest journeys within the theme park industry, from pre-visit planning to post-visit feedback. By leveraging advanced analytics, machine learning, and predictive modeling, the system will gain deeper insights into guest preferences and pain points, helping to improve overall guest experiences consistently. This project centers on Universal Studios Singapore, utilizing business metrics to identify opportunities for improvement within their amusement park.

## Prerequisites
- Python 3.10 or higher
- Git
- [Docker](https://www.docker.com/get-started) and Docker Compose
- bert.pt downloaded from [OneDrive](https://nusu-my.sharepoint.com/personal/e0929810_u_nus_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments%2Fbert%5Fmodel%2Ept&parent=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments&ga=1) 

Once you have downloaded bert.pt, move the file into the model/ directory. The following visual has been provided:
```bash
models/
└── bert_model.pt
```

## Running the Docker
### 1. Clone the Repository
Clone the GitHub repository and navigate into the project folder
```bash
git clone https://github.com/darderrdur17/data-science-guest-experience.git
cd data-science-guest-experience
```
### 2. Build and Start the Docker Container
This command builds and starts the docker container
```bash
docker-compose up --build
```
### 3. Stop the Docker Container
Once you're done, you can stop and remove the container with the following command
```bash
docker-compose down
```

## Running the Project
The entry point of this project is:
```bash
Scripts/main.py
```

