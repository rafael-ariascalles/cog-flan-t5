#sudo docker run --gpus all -v ~/cache:/root/.cache/ --rm --name api -d -it -p 8095:5000 stakapi
sudo docker run --gpus --env-file .env all --rm --name api -d -it -p 8095:5000 stakapi