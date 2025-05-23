# Improving Theme Park Guest Experience with Data Analytics

## Project Overview
This project aims to develop a data-driven system that maps and analyzes guest journeys within the theme park industry, from pre-visit planning to post-visit feedback. By leveraging advanced analytics, machine learning, and predictive modeling, the system will gain deeper insights into guest preferences and pain points, helping to improve overall guest experiences consistently. This project centers on Universal Studios Singapore, utilizing business metrics to identify opportunities for improvement within their amusement park.

## Prerequisites
- Python 3.10 or higher
- Git
- [Docker](https://www.docker.com/get-started)
- `bert.pt` downloaded from [OneDrive](https://nusu-my.sharepoint.com/personal/e0929810_u_nus_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments%2Fbert%5Fmodel%2Ept&parent=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments&ga=1) 

Once `bert.pt` is downloaded, follow the first step below and clone the repository. After that, move `bert.pt` into the models/ directory. The following visual has been provided:
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
### 2. Build the Docker Container
This command builds and starts the docker container
```bash
docker-compose build
```

### 3. Run the Docker and Choose an Entry Point
You can choose which analysis module to run using `docker-compose`.

#### Available options:
- `main.py`: Runs all components
- `main_A.py`: Runs Guest Journey Analysis and Segmentation
- `main_B.py`: Runs Experience Optimization and Predictive Modeling

#### To run `main.py`:
```bash
docker-compose up uss_main
```

#### To run `main_A.py`:
```bash
docker-compose up uss_a
```

#### To run `main_B.py`:
```bash
docker-compose up uss_b
```

### 4. Stop the Docker Container
Once you're done, you can stop and remove the container with the following command
```bash
docker-compose down
```
