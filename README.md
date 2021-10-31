# Kappa Backend

This is the RESTful API for Kappa built using Flask. Kappa is a Cluster-based Comic Recommender System.

## Demo
You can try the full application **[here](kappa.zsyihab.tech)**, please remember that you need to input a minimum of five comics.

## Container
There is a Docker images available to grab.  
`docker pull teszerrakt/kappa-api`

## Available Request
- `GET /` return `WELCOME TO KAPPA`
- `POST /api/kmeans` return a list of recommended comics processed by K-Means.
- `POST /api/dbscan` return a list of recommended comics processed by DBSCAN.

### Input
For the `POST` request it expect input with `id` and `rating`.  
For example:   
`[  
    {  
        "id": 44489,  
        "rating": 5
    },  
    {
        "id": 72025
        "rating": 5,  
    }  
  ]`  
    
  
  
